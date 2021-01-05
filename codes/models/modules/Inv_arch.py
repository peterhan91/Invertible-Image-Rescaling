import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data.fastmri import transforms


class InvBlockExp(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)

    def jacobian(self, x, rev=False):
        return self.last_jac


class FourierDownsampling(nn.Module):
    def __init__(self, mask_func, seed):
        super(FourierDownsampling, self).__init__()
        self.mask_func = mask_func
        self.seed = seed

    def forward(self, x, rev=False):
        if x.size(-1) != 1:
            img = torch.stack((x, torch.zeros_like(x)), -1)
        assert img.size(-1) == 2
        
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 2 * np.log(1/4.)
            
            # x shape [bs, ch, H, W] e.g. [16, 1, 144, 144]
            kspace = transforms.fft2(img) # [bs, ch, H, W, 2]
            center_kspace, _ = transforms.apply_mask(kspace, self.mask_func, 
                                                    seed=self.seed, cuda=True)
            periph_kspace, _ = transforms.apply_mask(kspace, self.mask_func, 
                                                    rev=True, seed=self.seed, cuda=True)
            img_LF = transforms.complex_abs(transforms.ifft2(center_kspace)) # [bs, ch, H, W]
            img_HF = transforms.complex_abs(transforms.ifft2(periph_kspace)) 
            return torch.cat((img_LF, img_HF), dim=1) # [bs, ch*2, H, W]
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 2 * np.log(4.)
            
            # if rev: img shape [bs, ch*2, H, W, 2]
            center_kspace = transforms.fft2(img.narrow(1, 0, 1)) # shape   
            periph_kspace = transforms.fft2(img.narrow(1, 1, 1)) # shape [bs, ch, H, W, 2]
            out = transforms.complex_abs(transforms.ifft2(center_kspace + periph_kspace))
            return out  # [bs, ch, H, W]

    def jacobian(self, x, rev=False):
        return self.last_jac


class InvRescaleNet(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, 
                    block_num=[], down_num=2, Haar=True, mask_func=None, seed=None):
        super(InvRescaleNet, self).__init__()

        operations = []

        current_channel = channel_in
        for i in range(down_num):
            if Haar:
                b = HaarDownsampling(current_channel)
                current_channel *= 4
            else:
                b = FourierDownsampling(mask_func, seed)
                current_channel *= 2
            operations.append(b)

            for j in range(block_num[i]):
                b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                operations.append(b)

        self.operations = nn.ModuleList(operations)

    def forward(self, x, rev=False, cal_jacobian=False):
        out = x
        jacobian = 0

        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)

        if cal_jacobian:
            return out, jacobian
        else:
            return out

