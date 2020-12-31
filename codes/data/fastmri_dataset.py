import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util

class FASTMRIDataset(data.Dataset):
    '''
    Read subsampled (Low Quality, here is LQ) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''
    def __init__(self, opt, mask_func, transform):
        super(FASTMRIDataset, self).__init__()
        self.opt = opt
        self.transform = transform
        self.mask_func = mask_func
        self.seed = self.opt['seed']

        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None  # environment for lmdb

        self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def __getitem__(self, index):
        if self.data_type == 'lmdb':
            if (self.GT_env is None) or (self.LQ_env is None):
                self._init_lmdb()
        GT_path, LQ_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        # get GT image
        GT_path = self.paths_GT[index]
        if self.data_type == 'lmdb':
            resolution = [int(s) for s in self.sizes_GT[index].split('_')]
        else:
            resolution = None
        img_GT = util.read_img(self.GT_env, GT_path, resolution)
        # modcrop in the validation / test phase
        if self.opt['phase'] != 'train':
            img_GT = util.modcrop(img_GT, scale)
        # change color space if necessary
        if self.opt['color']:
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]
        if img_GT.ndim == 2:
            img_GT = np.expand_dims(img_GT, axis=2)
        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, C = img_GT.shape
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),
                                    interpolation=cv2.INTER_LINEAR)
            
            # randomly crop - GT img shape: [H, W, 1]
            rnd_h = random.randint(0, max(0, H - GT_size))
            rnd_w = random.randint(0, max(0, W - GT_size))
            img_GT = img_GT[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]

            # augmentation - flip, rotate
            img_GT = util.augment([img_GT], self.opt['use_flip'], self.opt['use_rot'])[0]

        # transform() - subsample k space - img_LF
        # input img shape [H, W, 1] or [H, W, 3]
        img_LF, img_GT = self.transform(img_GT, self.mask_func, self.seed)

        if LQ_path is None:
            LQ_path = GT_path
        return {'LQ': img_LF, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)