"""Crop/Split optical RSIs and save as .npy format
Author: Cheney
Email: zhoucslyx@gmail.com
Date: 20190416
"""
import os
import shutil
#  import concurrent.futures
import sys

import gdal
import numpy as np

sys.path.append(os.path.expanduser('~/code/BasicSR/codes/data/'))
from util import imresize_np

#  print(sys.argv[0])
#  print(os.path.realpath(__file__))
#  print(os.path.abspath(__file__))
#  print(__name__)
#  print(__package__)
#  print(__file__)
#  sys.path.append(os.path.expanduser('~/code/BasicSR/codes'))


class OpticalRSI(object):
    """TIF/tif-format optical RSI"""

    def __init__(self, dataname):
        """Basic info"""
        if dataname not in ('QuickBird', 'IKONOS'):
            raise NameError(f'Data names: QuickBird, IKONOS, '
                            f'but found {dataname}!')
        self.prefix = 'opticalRSI'
        self.dataname = dataname
        # data root
        self.dataroot = os.path.expanduser(
            f'~/data/{self.prefix}/{self.dataname}')
        # image format
        if dataname == 'QuickBird':
            self.img_fmt = 'TIF'
        elif dataname == 'IKONOS':
            self.img_fmt = 'tif'
        # set type
        self.set_types = ['MUL', 'PAN']
        # image name
        self.img_names = [
            f'{self.dataname}_Sample_{ele}' for ele in self.set_types
        ]
        # bands with wavelength from short to long
        # usually only Blue, Green, Red, NearInfraRed are used
        self.bands = ('coast', 'blue', 'green', 'yellow', 'red', 'red_edge',
                      'NIR', 'NIR2')
        # pixel resolution MUL/PAN
        self.mp_ratio = 4

    def read_img(self, set_type):
        """Read img as array"""
        if set_type not in self.set_types:
            raise NameError('MUL or PAN?')
        img_name = self.img_names[self.set_types.index(set_type)]
        path = os.path.join(self.dataroot, f'{img_name}.{self.img_fmt}')
        # gdal treat a tif image as a dataset
        dataset = gdal.Open(path, gdal.GA_ReadOnly)
        channels, height, width = (dataset.RasterCount, dataset.RasterYSize,
                                   dataset.RasterXSize)
        # gdal default order: (channels, height, width)
        # if channels == 1
        img = dataset.ReadAsArray()
        if channels == 1:
            img = img.reshape(channels, height, width)
        return img, (channels, height, width)

    def get_write_name(self, top, left, set_type, writeroot=None):
        """Write name for sub_img"""
        # root
        if writeroot is None:
            writeroot = os.path.join(self.dataroot, f'{set_type}_sub')
            os.makedirs(writeroot, exist_ok=True)
        # name
        img_name = '{:s}_{:d}_{:d}.npy'.format(set_type, top, left)
        return os.path.join(writeroot, img_name)


def write_img(img, img_name):
    """np.array"""
    with open(img_name, 'wb') as f_img:
        np.save(f_img, img)


def crop_img(img, top, left, crop_size=256):
    """np.array"""
    _, height, width = img.shape
    assert height >= crop_size and width >= crop_size, 'img < crop_size!'
    top_mM = max(0, min(top, height - crop_size))
    left_mM = max(0, min(left, width - crop_size))
    sub_img = img[:, top_mM:top_mM + crop_size, left_mM:left_mM + crop_size]
    return sub_img, (top_mM, left_mM)


def crop_flow(dataname='QuickBird', set_type='MUL', reduced=True):
    """Crop imags."""
    # dataset
    dataset = OpticalRSI(dataname)
    # Crop info
    factor = 1 if set_type == 'MUL' else dataset.mp_ratio
    if reduced:
        crop_sz = 256 * factor
        step = 200 * factor
    else:
        crop_sz = 64 * factor
        step = 50 * factor
    # img
    img, (channels, height, width) = dataset.read_img(set_type)
    # crop key node
    h_steps = (height - crop_sz) // step + 1
    w_steps = (width - crop_sz) // step + 1
    #  yx = [[y, x] for y in range(h_steps) for x in range(w_steps)]
    # Concurrent
    #  args = ((ele, step, img, crop_sz, dataset, set_type) for ele in yx)
    #  with concurrent.futures.ProcessPoolExecutor() as executor:
    #  executor.map(crop_save, args)
    for y_top in range(h_steps):
        for x_left in range(w_steps):
            top = y_top * step
            left = x_left * step
            sub_img, (top_mM, left_mM) = crop_img(img, top, left, crop_sz)
            save_name = dataset.get_write_name(top_mM, left_mM, set_type)
            write_img(sub_img, save_name)


def main(dataname='QuickBird'):
    for set_type in ('MUL', 'PAN'):
        print("==================================")
        print(f"Processing {set_type}...")
        crop_flow(dataname, set_type, reduced=False)
    print("Done!")
    return


# ==============split train val test=====================


def split(dataname='QuickBird'):
    """Split dataset into train val test"""
    # dataset info
    dataroot = os.path.expanduser(f'~/data/opticalRSI/{dataname}/')
    img_types = ['MUL', 'PAN']
    paths = [os.path.join(dataroot, f'{ele}_sub') for ele in img_types]
    img_lists = [
        sorted(os.listdir(path),
               key=lambda ele: list(map(int,
                                        ele.split('.')[0].split('_')[1:])))
        for path in paths
    ]
    assert len(img_lists[0]) == len(img_lists[1]), 'num_MUL != num_PAN'
    # split info
    proportions = [0, 0.6, 0.8, 1]
    set_types = ['train', 'val', 'test']
    # split index
    num_img = len(img_lists[0])
    idxs = list(range(num_img))
    np.random.shuffle(idxs)
    # split
    key_nodes = [int(num_img * prop) for prop in proportions]
    for i, set_type in enumerate(set_types):
        this_idx = idxs[key_nodes[i]:key_nodes[i + 1]]
        for j, img_type in enumerate(img_types):
            save_path = os.path.join(dataroot, set_type, img_type)
            os.makedirs(save_path, exist_ok=True)
            # copy
            for ele in this_idx:
                img_name = img_lists[j][ele]
                shutil.copy(os.path.join(paths[j], img_name), save_path)


# =====================bic_LR============================


def generate_LR(img_HR, scale=4):
    """generate LR image,
    size of img_HR should be multiple of scale"""
    assert scale > 1, "scale should be greater than 1!"
    H_s, W_s, _ = img_HR.shape
    # maybe the assert is not necessary
    temp = (H_s % scale == 0) and (W_s % scale == 0)
    assert temp, "img size should be multiple of {}".format(scale)
    # using matlab imresize
    img_LR = imresize_np(img_HR, 1 / scale, True)
    return img_LR


def bic_LR(dataname='QuickBird'):
    # dataset info
    dataroot = os.path.expanduser('~/data/opticalRSI/{}/'.format(dataname))
    set_types = ['train', 'val', 'test']
    img_types = ['MUL', 'PAN']
    # scale info
    scale = 4
    interpolate = 'bic'
    suffix = '{}LRx{}'.format(interpolate, scale)

    for set_type in set_types:
        for img_type in img_types:
            # save dir
            save_dir = os.path.join(dataroot, set_type,
                                    '_'.join([img_type, suffix]))
            os.makedirs(save_dir, exist_ok=True)
            # image names
            img_folder = os.path.join(dataroot, set_type, img_type)
            names = os.listdir(img_folder)
            # load, resize and save
            for name in names:
                print('Processing {}'.format(name))
                # load
                full_path = os.path.join(img_folder, name)
                img_HR = np.load(full_path)
                data_type = img_HR.dtype
                m, M = img_HR.min(), img_HR.max()
                # preprocess, CHW -> HWC, uint16 -> float32, range[0, 1]
                img = img_HR.transpose(1, 2, 0).astype(np.float32)
                img = (img - m) / (M - m + 1e-16)
                # resize
                img_LR = generate_LR(img, scale)
                # postprocess
                img = m + (M - m) * img_LR
                img = img.transpose(2, 0, 1).astype(data_type)
                # save
                save_name = os.path.join(save_dir, name)
                np.save(save_name, img)
    return


if __name__ == '__main__':
    dataname = 'IKONOS'
    #  main(dataname)
    #  split(dataname)
    bic_LR(dataname)
