'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data
import numpy as np


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=True)


def create_dataset(dataset_opt):
    mode = dataset_opt['mode'] # mode ~ which dataset to use
    if mode == 'LQ':
        from data.LQ_dataset import LQDataset as D
        dataset = D(dataset_opt)
    elif mode == 'LQGT':
        from data.LQGT_dataset import LQGTDataset as D
        dataset = D(dataset_opt)
    elif mode == 'FastMRI':
        from data.fastmri_dataset import FASTMRIDataset as D
        from data.fastmri import subsample, transforms
        # Create a mask function
        mask_func = subsample.RandomMaskFunc(
            center_fractions=[0.08],
            accelerations=[4]
        )
        class DataTransform:
            def __call__(self, target, mask_func, seed=None):
                # Preprocess the data here
                # target shape: [H, W, 1] or [H, W, 3]
                if target.shape[2] == 1:
                    img = np.concatenate((target, np.zeros_like(target)), axis=2)
                assert img.shape[-1] == 2
                img = transforms.to_tensor(img)
                kspace = transforms.fft2(img) 

                center_kspace, _ = transforms.apply_mask(kspace, mask_func, seed=seed)
                img_LF = transforms.complex_abs(transforms.ifft2(center_kspace))
                img_LF = img_LF.unsqueeze(0)
                # img_LF tensor should have shape [H, W, ?]
                target = transforms.to_tensor(np.transpose(target, (2, 0, 1)))  # target shape [1, H, W]
                return img_LF, target
        dataset = D(dataset_opt, mask_func, transform=DataTransform())
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
