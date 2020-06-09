"""
Ref: github/xinntao
Modified: github/wasacheney
"""

import os.path
import pickle

import lmdb
import numpy as np


def main(dataname, reduced=True):
    # Dataset info
    #  img_fmt = 'TIF'; storage_fmt = 'npy'
    dataroot = os.path.expanduser('~/data/opticalRSI/{}'.format(dataname))
    scale = 4
    LR_suffix = '_bicLRx{}'.format(scale)
    if reduced:
        img_types = [
            'MUL', 'MUL{}'.format(LR_suffix), 'PAN', 'PAN{}'.format(LR_suffix)
        ]
    else:
        img_types = ['MUL', 'PAN']
    set_types = ['train', 'val', 'test']
    # Load and Save
    for set_type in set_types:
        # get img full path
        img_list = []
        for img_type in img_types:
            img_folder = os.path.join(dataroot, set_type, img_type)
            names = os.listdir(img_folder)
            full_paths = [os.path.join(img_folder, name) for name in names]
            img_list.extend(full_paths)
        # Create lmdb env
        lmdb_save_path = os.path.join(dataroot, f'{set_type}.lmdb')
        env = lmdb.open(lmdb_save_path, map_size=37580963840)
        # Load, preprocess and save
        print('Create lmdb: {}'.format(lmdb_save_path))
        with env.begin(write=True) as txn:
            for i, path in enumerate(img_list):
                print(f'processing image {i:d}')
                # img, key
                img = np.load(path)
                basename = os.path.basename(path).split('.')[0]
                if path.find(LR_suffix) != -1:
                    basename += LR_suffix
                key = basename.encode('ascii')
                # shape, meta key
                C, H, W = img.shape
                meta = f'{C:d}, {H:d}, {W:d}'
                meta_key = '.'.join([basename + '.meta']).encode('ascii')
                # The encode is only essential in Python 3
                txn.put(key, img)
                txn.put(meta_key, meta.encode('ascii'))
        print('Finish creating lmdb.')
        # Create keys_cache_file
        keys_cache_file = os.path.join(lmdb_save_path, '_keys_cache.pkl')
        env = lmdb.open(lmdb_save_path,
                        readonly=True,
                        lock=False,
                        readahead=False,
                        meminit=False)
        with env.begin(write=False) as txn:
            print(f'Create lmdb keys cache: {keys_cache_file}')
            keys = [key.decode('ascii') for key, _ in txn.cursor()]
            pickle.dump(keys, open(keys_cache_file, "wb"))
        print('Finish creating lmdb keys cache.')


if __name__ == '__main__':
    dataname = 'IKONOS'
    main(dataname, reduced=False)
