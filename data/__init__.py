'''create dataset and dataloader'''
import logging
import torch.utils.data

GDRIVE = False

def create_dataloader(dataset, dataset_opt):
    '''create dataloader '''
    phase = dataset_opt['phase']
    if phase == 'train':
        b_s = dataset_opt['batch_size']
        s_f = dataset_opt['use_shuffle']
        n_w = 0 if GDRIVE else dataset_opt['n_workers']
        d_p = True
    else:
        b_s = 1
        s_f = False
        n_w = 0
        d_p = False
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=b_s,
        shuffle=s_f,
        num_workers=n_w,
        drop_last=d_p,
        pin_memory=True)


def create_dataset(dataset_opt, rsi=True):
    '''create dataset'''
    mode = dataset_opt['mode']
    if mode == 'LR':
        from data.LR_dataset import LRDataset as D
    elif mode == 'LRHR':
        if rsi:
            from data.LRHR_dataset import NPYGeo as D
        else:
            from data.LRHR_dataset import LRHRDataset as D
    elif mode == 'LRHRseg_bg':
        from data.LRHR_seg_bg_dataset import LRHRSeg_BG_Dataset as D
    else:
        raise NotImplementedError(
            'Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(
        dataset.__class__.__name__, dataset_opt['name']))
    return dataset
