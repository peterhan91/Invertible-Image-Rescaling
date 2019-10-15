import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil

class SimpleNetBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='kaiming'):
        super(SimpleNetBlock, self).__init__()

        act = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        body = []
        expand = 6

        body.append(nn.Conv2d(channel_in, channel_in * expand, 1, padding=1//2))
        body.append(act)
        body.append(nn.Conv2d(channel_in * expand, channel_in, 1, padding=1//2))
        body.append(act)
        body.append(nn.Conv2d(channel_in, channel_out, 3, padding=3//2))
        body.append(act)

        if init == 'xavier':
            mutil.initialize_weights_xavier(body[:-1], 0.1)
        else:
            mutil.initialize_weights(body[:-1], 0.1)
        mutil.initialize_weights(body[-1], 0)

        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


class THNetBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='kaiming'):
        super(THNetBlock, self).__init__()

        act = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        body = []
        hidden_channel = 64

        body.append(nn.Conv2d(channel_in, hidden_channel, 1, padding=1//2))
        body.append(act)
        body.append(nn.Conv2d(hidden_channel, hidden_channel, 3, padding=3//2))
        body.append(act)
        body.append(nn.Conv2d(hidden_channel, channel_out, 1, padding=1//2))
        body.append(act)

        if init == 'xavier':
            mutil.initialize_weights_xavier(body[:-1], 0.1)
        else:
            mutil.initialize_weights(body[:-1], 0.1)
        mutil.initialize_weights(body[-1], 0)

        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)

class RRDBNetBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='kaiming', gc=32, bias=True):
        super(RRDBNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5


def subnet(net_structure, init='kaiming'):
    def constructor(channel_in, channel_out):
        if net_structure == 'SimpleNet':
            if init == 'xavier':
                return SimpleNetBlock(channel_in, channel_out, init)
            else:
                return SimpleNetBlock(channel_in, channel_out)
        elif net_structure == 'THNet':
            if init == 'xavier':
                return THNetBlock(channel_in, channel_out, init)
            else:
                return THNetBlock(channel_in, channel_out)
        elif net_structure == 'RRDBNet':
            if init == 'xavier':
                return RRDBNetBlock(channel_in, channel_out, init)
            else:
                return RRDBNetBlock(channel_in, channel_out)
        else:
            return None

    return constructor