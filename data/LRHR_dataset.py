import os.path
import sys
import random
import copy
import pickle
from pathlib import Path

import lmdb
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data

# local or gdrive
GDRIVE = False
DRIVE_ROOT = '/content/gdrive/My Drive' if GDRIVE else os.getenv('HOME')
sys.path.append(os.path.join(DRIVE_ROOT, 'code/BasicSR/codes'))

import data.util as util


class LRHRDataset(data.Dataset):
    '''
    Read LR and HR image pairs.
    If only HR image is provided, generate LR image on-the-fly.
    The pair is ensured by 'sorted' function,
    so please check the name convention.
    '''

    def __init__(self, opt, rsi=True):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        self.rsi = rsi
        self.dynamic_range = 2 * 8 - 1  # pixel value [0, dynamic_range]
        self.paths_LR = None
        self.paths_HR = None
        self.LR_env = None  # environment for lmdb
        self.HR_env = None

        # read image list from subset list txt
        if opt['subset_file'] is not None and opt['phase'] == 'train':
            with open(opt['subset_file']) as f:
                self.paths_HR = sorted([
                    os.path.join(opt['dataroot_HR'], line.rstrip('\n'))
                    for line in f
                ])
            if opt['dataroot_LR'] is not None:
                raise NotImplementedError(
                    'Now subset only supports generating LR on-the-fly.')
        else:  # read image list from lmdb or image files
            self.HR_env, self.paths_HR = util.get_image_paths(
                opt['data_type'], opt['dataroot_HR'])
            self.LR_env, self.paths_LR = util.get_image_paths(
                opt['data_type'], opt['dataroot_LR'])

        assert self.paths_HR, 'Error: HR path is empty.'
        if self.paths_LR and self.paths_HR:
            num_lr = len(self.paths_LR)
            num_hr = len(self.paths_HR)
            assert num_lr == num_hr, 'LR: {} != HR: {}.'.format(num_lr, num_hr)

        self.random_scale_list = [1]

    def __getitem__(self, index):
        HR_path, LR_path = None, None
        scale = self.opt['scale']
        # expected HR
        HR_size = self.opt['HR_size']
        # get HR image ==========
        HR_path = self.paths_HR[index]
        img_HR = util.read_img(self.HR_env, HR_path)
        # modcrop in the validation / test phase
        # multiple of scale
        if self.opt['phase'] != 'train':
            img_HR = util.modcrop(img_HR, scale)
        # change color space if necessary
        # BGR, YCbCr, Gray
        if self.opt['color']:
            img_HR = util.channel_convert(img_HR.shape[2], self.opt['color'],
                                          [img_HR])[0]

        # get LR image ===========
        if self.paths_LR:
            LR_path = self.paths_LR[index]
            img_LR = util.read_img(self.LR_env, LR_path)
        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_HR.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, HR_size)
                W_s = _mod(W_s, random_scale, scale, HR_size)
                img_HR = cv2.resize(np.copy(img_HR), (W_s, H_s),
                                    interpolation=cv2.INTER_LINEAR)
                # force to 3 channels,
                #  GRAY2BGR results in B=G=R, gray image in RGB-color space
                if img_HR.ndim == 2:
                    img_HR = cv2.cvtColor(img_HR, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_HR.shape
            # using matlab imresize
            img_LR = util.imresize_np(img_HR, 1 / scale, True)
            if img_LR.ndim == 2:
                img_LR = np.expand_dims(img_LR, axis=2)

        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_HR.shape
            if H < HR_size or W < HR_size:
                img_HR = cv2.resize(np.copy(img_HR), (HR_size, HR_size),
                                    interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LR = util.imresize_np(img_HR, 1 / scale, True)
                if img_LR.ndim == 2:
                    img_LR = np.expand_dims(img_LR, axis=2)

            H, W, C = img_LR.shape
            LR_size = HR_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
            rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
            img_HR = img_HR[rnd_h_HR:rnd_h_HR + HR_size, rnd_w_HR:rnd_w_HR +
                            HR_size, :]
            # augmentation - flip, rotate
            img_LR, img_HR = util.augment([img_LR, img_HR],
                                          self.opt['use_flip'],
                                          self.opt['use_rot'])

        # change color space if necessary
        if self.opt['color']:
            img_LR = util.channel_convert(
                C, self.opt['color'],
                [img_LR])[0]  # TODO during val no definition

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_HR.shape[2] == 3:
            img_HR = img_HR[..., [2, 1, 0]]
            img_LR = img_LR[..., [2, 1, 0]]
        img_HR = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()

        if LR_path is None:
            LR_path = HR_path
        return {
            'LR': img_LR,
            'HR': img_HR,
            'LR_path': LR_path,
            'HR_path': HR_path
        }

    def __len__(self):
        return len(self.paths_HR)


###########################
# .npy img in LMDB database
###########################


class NPYGeo(data.Dataset):
    """This is for Multispectral RSIs
    including satellites (IKONOS, QuickBird, WorldView2),
    for IKONOS and QuickBird, bands are (B, G, R, NIR)
    for WorldView2, bands are(cost, B, G, Y, R, red_edge, NIR, NIR2)"""

    def __init__(self, dataset_opt, scale=4):
        super().__init__()
        #
        self.opt = dataset_opt
        self.scale = scale  # HR_size / LR_size
        # set type
        self.set_type = self.opt['phase']  # key is added in opt parse function
        if self.set_type not in ['train', 'val', 'test']:
            raise NotImplementedError(
                'training set type should be train , val or test')
        # Data fmt
        self.data_fmt = self.opt['data_type']
        if self.data_fmt != 'lmdb':
            raise NotImplementedError('Only accept lmdb-format file')
        # dataset name
        self.dataname = self.opt['name']
        if self.dataname not in ['IKONOS', 'QuickBird', 'WorldView2']:
            raise NotImplementedError('Unrecognized data name')
        # path
        #  self.train_type = opt['train_type']
        #  self.dataset_type = self.opt['dataroot_HR'].find('reduced')
        self.root = Path(DRIVE_ROOT).joinpath(self.opt['dataroot_HR'])
        self.keys_cache = '_keys_cache.pkl'
        # misc
        self.img_types = ['MUL', 'PAN']
        self.dynamic_range = 2**11 - 1  # pixel value [0, dynamic_range]
        # keys of HR and LR
        self.interpolate = 'bic'  # method generating LR images
        self.LR_suffix = '{}LRx{}'.format(self.interpolate, self.scale)
        self.HR_suffix = 'HR'
        self.resolutions = [
            self.HR_suffix, self.LR_suffix
        ] if self.opt['dataroot_HR'].find('reduced') != -1 else [
            self.HR_suffix
        ]
        # names of images
        self.env, self.names = self.get_names(self.set_type)
        self.names_dict = self.get_names_dict(self.names)

    def __len__(self):
        factor = len(self.img_types) * len(self.resolutions)
        return int(len(self.names) / factor)

    def __getitem__(self, index):
        # imgs is a dict,
        # when reduced dataset:
        # {'HR': {'MUL': ..., 'PAN': ...}, 'bicLRx4': {'MUL': ..., 'PAN': ...}}
        # when full dataset:
        # {'HR': {'MUL': ..., 'PAN': ...}}
        imgs = {}
        for HL in self.resolutions:
            if HL not in imgs:
                imgs[HL] = {}
            for MP in self.img_types:
                name = self.names_dict[HL][MP][index]
                img = self.get_img(self.env, name)
                img = self.basic_process(img)
                img = self.augment(img)
                # To Tensor and CHW
                img = torch.from_numpy(
                    np.ascontiguousarray(np.transpose(img,
                                                      (2, 0, 1)))).float()
                imgs[HL][MP] = [img, name]
        return imgs

    def get_names(self, set_type='train'):
        """All names for images"""
        lmdb_path = self.root.joinpath('.'.join([set_type, self.data_fmt]))
        keys_path = lmdb_path.joinpath(self.keys_cache)
        env = lmdb.open(str(lmdb_path),
                        readonly=True,
                        lock=False,
                        readahead=False,
                        meminit=False)
        if keys_path.is_file():
            with open(keys_path, 'rb') as f_keys:
                keys = pickle.load(f_keys)
        else:
            with env.begin(write=False) as txn:
                keys = [key.decode('ascii') for key, _ in txn.cursor()]
            with open(keys_path, 'wb') as f_keys:
                pickle.dump(keys, f_keys)
        names = sorted([key for key in keys if not key.endswith('.meta')])
        return env, names

    def get_names_dict(self, names):
        """Dictionarize names for convenince and sort for alignment"""
        names_dict = {
            self.LR_suffix: {img_type: []
                             for img_type in self.img_types},
            self.HR_suffix: {img_type: []
                             for img_type in self.img_types}
        }
        # group
        for name in names:
            if name.endswith('meta'):
                continue
            else:
                img_type = name.split('_')[0]
                LH = self.LR_suffix if name.endswith(
                    self.LR_suffix) else self.HR_suffix
                names_dict[LH][img_type].append(name)
        # sort
        for _, LHv in names_dict.items():
            for _, typev in LHv.items():
                typev.sort(
                    key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2])))
        return names_dict

    def get_img(self, env, name):
        """Image names"""
        with env.begin(write=False) as txn:
            buf = txn.get(name.encode('ascii'))
            buf_meta = txn.get(
                (name + '.meta').encode('ascii')).decode('ascii')
        img_flat = np.frombuffer(buf, dtype=np.uint16)
        shapes = [int(s) for s in buf_meta.split(',')]
        img = img_flat.reshape(shapes)
        return img

    def basic_process(self, img, norm_type='dynamic_range'):
        """Input: CHW, B G R NIR, uint16
        norm_type: channels-wise, mM, dynamic_range
        Output: HWC, BGRNIR, float32 [0, 1]"""
        img_ = copy.deepcopy(img)
        img_ = img_.astype(np.float32).transpose(1, 2, 0)
        H, W, C = img_.shape
        if norm_type == 'channel-wise':
            m = img_.reshape(-1, C).min(axis=0)
            M = img_.reshape(-1, C).max(axis=0)
        elif norm_type == 'mM':
            m, M = img_.min(), img_.max()
        elif norm_type == 'dynamic_range':
            m, M = 0, self.dynamic_range
        else:
            print('Unknown norm_type!')
            raise NotImplementedError
        img_ = (img_ - m) / (M - m + 1e-16)
        return img_

    def augment(self, img):
        """Input: HWC imgs"""
        hflip_ = self.opt['use_flip'] and random.random() < 0.5
        vflip_ = self.opt['use_flip'] and random.random() < 0.5
        rot_ = self.opt['use_rot'] and random.random() < 0.5
        img_ = copy.deepcopy(img)
        if hflip_:
            img_ = np.flip(img_, 1)
        if vflip_:
            img_ = np.flip(img_, 0)
        if rot_:
            img_ = np.rot90(img_, axes=(0, 1))
        return img_

    def display_process(self, img, p=(0.02, 0.98)):
        """Input: HWC, BGRNIR, float32 [0, 1]
        Channel-wise clip; range transform [0, 1]"""
        img_ = copy.deepcopy(img)
        H, W, C = img_.shape
        # get idxs of min and max
        N = H * W
        idxs = [int(N * ele) for ele in p]
        # min and max
        img_sort = np.sort(img_.reshape(-1, C), axis=0)
        mM = img_sort[idxs, :]
        # clip
        img_flat = np.clip(img_.reshape(-1, C), mM[0], mM[1])
        # range transform
        img_flat = (img_flat - mM[0]) / (mM[1] - mM[0] + 1e-16)
        img_ = img_flat.reshape(H, W, C)
        return img_

    def display_group(self, imgs, names):
        """imgs: torch.tensor, CHW, float, [0, 1]"""
        assert len(imgs) == len(names) == 4, '4 imgs in a group'
        fig, ax = plt.subplots(2, 2)
        for i, (img, name) in enumerate(zip(imgs, names)):
            img = img.numpy().transpose(1, 2, 0)
            img = dataset.display_process(img, p=(0.02, 0.98))
            H, W, C = img.shape
            print(H, W, C)
            if C != 1:
                img = img[..., 2::-1]
            else:
                img = img[..., 0]
            ax[i // 2][i % 2].imshow(img, cmap='gray')
            ax[i // 2][i % 2].set_title(name)
        plt.show()


if __name__ == '__main__':
    opt = {
        'name': 'QuickBird',
        'phase': 'val',
        'data_type': 'lmdb',
        'use_flip': True,
        'use_rot': True,
        'n_workers': 4,
        'batch_size': 16,
        'use_shuffle': True
    }
    dataset = NPYGeo(opt, scale=4)
    print(len(dataset))
    print(dataset.LR_suffix)
    print(dataset.names_dict.keys())
    print(dataset.names_dict['bicLRx4'].keys())
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt['batch_size'],
                                             shuffle=opt['use_shuffle'],
                                             num_workers=opt['n_workers'],
                                             drop_last=True,
                                             pin_memory=True)
    for i, dict_2folds in enumerate(dataloader):
        if i == 1:
            exit()
        #  print(type(data))
        #  print(data.keys())
        #  print(data['bicLRx4'].keys())
        for HL, dict_fold in dict_2folds.items():
            for MP, img_name_list in dict_fold.items():
                print(HL, MP)
                print(type(img_name_list))
                print(len(img_name_list))
                print(type(img_name_list[0]))
                print(img_name_list[0].size())
                print(type(img_name_list[1]))
                print(img_name_list[1])
    img_dict = dataset.__getitem__(50)
    print(img_dict.keys())
    hr_mul = img_dict['HR']['MUL']
    hr_pan = img_dict['HR']['PAN']
    lr_mul = img_dict['bicLRx4']['MUL']
    lr_pan = img_dict['bicLRx4']['PAN']
    imgs = [hr_mul[0], hr_pan[0], lr_mul[0], lr_pan[0]]
    names = [hr_mul[1], hr_pan[1], lr_mul[1], lr_pan[1]]
    dataset.display_group(imgs, names)
