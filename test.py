"""This is SISR test
Reference: github/xinntao
Modified: github/wasacheney"""

#  import argparse
import logging
import os
import json
#  import time
#  from collections import OrderedDict

import options.options as option
import utils.util as util
from data import create_dataloader, create_dataset
#  from data.util import bgr2ycbcr
from models import create_model


def get_options(json_path):
    """options"""
    #  parser = argparse.ArgumentParser()
    #  parser.add_argument(
    #  '-opt', type=str, required=True, help='Path to options JSON file.')
    #  opt = option.parse(parser.parse_args().opt, is_train=False)
    is_train = False
    opt = option.parse(json_path, is_train)
    util.mkdirs((path for key, path in opt['path'].items()
                 if not key == 'pretrain_model_G'))
    opt = option.dict_to_nonedict(opt)

    util.setup_logger(None,
                      opt['path']['log'],
                      'test',
                      level=logging.INFO,
                      screen=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))
    return opt, logger


def get_ckpts_path(opt,
                   exp_path=os.path.expanduser('~/code/BasicSR/experiments/')):
    """Return all ckpts as a dict"""
    # prefixed
    prefix = opt['name']
    # all case
    paradigms = ['spsf_reduced']
    inits = ['random', 'psnr', 'esrgan']
    # loss weights
    ABGs = [(1, 0, 0), (0, 1, 0.01), (1, 1, 0.01)]
    # lr for G and D
    all_lrs = [(1e-4, 0), (1e-5, 0), (1e-4, 1e-4), (1e-4, 1e-5), (1e-5, 1e-4)]
    # result_dict =
    # {setting_reduced1:
    #   {ckpt_name1: [sam, psnr, scc, qindex, ssim, ergas],
    #    ckpt_name2: [sam, psnr, scc, qindex, ssim, ergas]},
    #  setting_full1:
    #   {ckpt_name1: [d_lambda, d_s, qnr],
    #    ckpt_name2: [d_lambda, d_s, qnr],
    #    ckpt_name3: [d_labmda, d_s, qnr]}}
    result_dict = {}
    for paradigm in paradigms:
        for init in inits:
            for alpha, beta, gamma in ABGs:
                if (alpha, beta, gamma) == (1, 0, 0):
                    lrs = all_lrs[:2]
                else:
                    lrs = all_lrs[2:]
                for lrG, lrD in lrs:
                    # settings as path name
                    storage_folder = '_'.join([
                        prefix, paradigm, init,
                        str(alpha),
                        str(beta),
                        str(gamma),
                        str(lrG),
                        str(lrD)
                    ])
                    # setting key, not the full path
                    result_dict[storage_folder] = {}
                    # full dir containing ckpts
                    ckpt_path = os.path.join(exp_path, storage_folder,
                                             'models')
                    # ckpt_name key
                    for ckpt in filter(lambda s: s.find('G') != -1,
                                       os.listdir(ckpt_path)):
                        result_dict[storage_folder][ckpt] = ()
    return result_dict


def get_dataloader(opt, logger):
    """Create test dataset and dataloader"""
    # test loaders
    test_loaders = []
    #  print(opt['datasets'])
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info('Number of test images in [{:s}]: {:d}'.format(
            dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)
    return test_loaders


def valid(opt, val_loader, model, logger, show=False):
    """Common image quality assessment"""
    range_max = val_loader.dataset.dynamic_range
    crop_size = opt['scale']
    # XXX
    opt['dataset_type'] = 'full'
    if opt['dataset_type'] == 'reduced':
        avg_sam = 0.
        avg_psnr = 0.
        avg_scc = 0.
        avg_qindex = 0.
        avg_ssim = 0.
        avg_ergas = 0.
    elif opt['dataset_type'] == 'full':
        avg_D_lambda = 0.0
        avg_D_s = 0.0
        avg_qnr = 0.0
    idx = 0
    for val_data in val_loader:
        idx += 1
        # forward
        model.feed_data(val_data)
        model.test()
        visuals = model.get_current_visuals()
        # Generated image
        sr_img = util.tensor2img(visuals['SR'],
                                 dynamic_range=range_max)  # uint
        # if show, it is not necessary to caculate scores
        if show:
            print(f'{val_loader.dataset.root}/{idx}.npy')
            util.save_img(sr_img, f'{val_loader.dataset.root}/_full_{idx}.npy')
            continue
        cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
        # "Reference" image
        if opt['dataset_type'] == 'reduced':
            gt_img = util.tensor2img(visuals['HRx'],
                                     dynamic_range=range_max)  # uint
            cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:
                                    -crop_size, :]
            now_sam = util.calculate_sam(cropped_sr_img, cropped_gt_img)
            now_psnr = util.calculate_psnr(cropped_sr_img, cropped_gt_img,
                                           range_max)
            now_scc = util.calculate_scc(cropped_sr_img, cropped_gt_img)
            now_qindex = util.calculate_qindex(cropped_sr_img,
                                               cropped_gt_img,
                                               block_size=8)
            now_ssim = util.calculate_ssim(cropped_sr_img, cropped_gt_img,
                                           range_max)
            now_ergas = util.calculate_ergas(cropped_sr_img,
                                             cropped_gt_img,
                                             scale=4)
            avg_sam += now_sam
            avg_psnr += now_psnr
            avg_scc += now_scc
            avg_qindex += now_qindex
            avg_ssim += now_ssim
            avg_ergas += now_ergas
            logger.info(f'# Validation image [{idx}] --- '
                        f'# SAM: {now_sam:.4e} # PSNR: {now_psnr:.4e} '
                        f'# SCC: {now_scc:.4e} # Q-index: {now_qindex:.4e} '
                        f'# SSIM: {now_ssim:.4e} # ERGAS: {now_ergas:.4e}')
        elif opt['dataset_type'] == 'full':
            # lr_x
            lr_x_img = util.tensor2img(visuals['LRx'], dynamic_range=range_max)
            # lr_p
            lr_p_img = util.tensor2img(visuals['LRp'], dynamic_range=range_max)
            now_qnr, now_D_lambda, now_D_s = util.calculate_qnr(
                sr_img,
                lr_x_img,
                lr_p_img,
                satellite=val_loader.dataset.dataname,
                scale=4,
                block_size=32,
                p=1,
                q=1,
                alpha=1,
                beta=1)
            avg_D_lambda += now_D_lambda
            avg_D_s += now_D_s
            avg_qnr += now_qnr
            logger.info(f'# Validation image [{idx}] --- '
                        f'# D_lambda: {now_D_lambda:.4e} '
                        f'# D_s: {now_D_s:.4e} # QNR: {now_qnr:.4e}')
    if show:
        return
    if opt['dataset_type'] == 'reduced':
        avg_sam /= idx
        avg_psnr /= idx
        avg_scc /= idx
        avg_qindex /= idx
        avg_ssim /= idx
        avg_ergas /= idx
        # log
        logger.info(
            f'# Validation Avg --- # SAM: {avg_sam:.4e} # PSNR: {avg_psnr:.4e} '
            f'# SCC: {avg_scc:.4e} # Q-index: {avg_qindex:.4e} '
            f'# SSIM: {avg_ssim:.4e} # ERGAS: {avg_ergas:.4e}')
        return avg_sam, avg_psnr, avg_scc, avg_qindex, avg_ssim, avg_ergas
    if opt['dataset_type'] == 'full':
        avg_D_lambda = avg_D_lambda / idx
        avg_D_s = avg_D_s / idx
        avg_qnr = avg_qnr / idx
        # log
        logger.info(
            '# Validation Avg --- # D_lambda: {:.4e} # D_s: {:.4e} # QNR: {:.4e}'
            .format(avg_D_lambda, avg_D_s, avg_qnr))
        return avg_D_lambda, avg_D_s, avg_qnr


def test(opt, logger, test_loader, model, show=False):
    """For every test set"""
    #  test_start_time = time.time()
    # Dataset info
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    # Save dir for SR
    #  dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
    #  util.mkdir(dataset_dir)
    scores = valid(opt, test_loader, model, logger, show)
    return scores


def test_flow():
    """Test quantitatively"""
    # opt
    json_path = 'options/test/test_ESRGAN.json'
    opt, logger = get_options(json_path)
    # ckpt path, some options will also be corrected
    exp_dir = os.path.expanduser('~/code/BasicSR/experiments/')
    result_dict = get_ckpts_path(opt, exp_path=exp_dir)
    for setting, ckpt_dict in result_dict.items():
        setting_list = setting.split('_')
        opt['train_type'] = setting_list[4]
        opt['dataset_type'] = setting_list[5]
        opt['paradigm'] = f"{opt['train_type']}_{opt['dataset_type']}"
        # XXX use full dataset only
        opt['datasets']['val']['dataroot_HR'] = opt['datasets']['val'][
            'dataroot_LR'] = "data/opticalRSI/QuickBird/full/"
        opt['datasets']['val']['data_type'] = 'lmdb'
        tmp = os.path.join(exp_dir, setting, 'models')
        for ckpt in ckpt_dict:
            # create model and load ckpt
            opt['path']['pretrain_model_G'] = os.path.join(tmp, ckpt)
            model = create_model(opt, rsi=True)
            # test dataloader
            test_loaders = get_dataloader(opt, logger)
            for test_loader in test_loaders:
                scores = test(opt, logger, test_loader, model)
                result_dict[setting][ckpt] = scores
    # store
    store_path = os.path.join(opt['path']['results_root'], 'result_dict.json')
    with open(store_path, 'w') as f_result:
        json.dump(result_dict, f_result)


def test_show():
    """Test qualitatively"""
    # opt
    json_path = 'options/test/test_ESRGAN.json'
    opt, logger = get_options(json_path)
    # ckpt path, some options will also be corrected
    exp_dir = os.path.expanduser('~/code/BasicSR/experiments/')
    #  result_dict = get_ckpts_path(opt, exp_path=exp_dir)
    # spuf full
    #  result_dict = {
    #  'RRDB_ESRGAN_x4_QuickBird_spuf_full_esrgan_0_1_0.01_0.0001_0.0001': {
    #  '2840_G_0.7558312431086864_qnr.pth': ()
    #  }
    #  }
    # spsf full
    #  result_dict = {
    #  'RRDB_ESRGAN_x4_QuickBird_spsf_reduced_esrgan_0_1_0.01_0.0001_1e-05': {
    #  '2300_G_0.9058441273171588_ssim.pth': ()
    #  }
    #  }
    #  result_dict = {
    #  'RRDB_ESRGAN_x4_QuickBird_spsf_reduced_esrgan_0_1_0.01_0.0001_1e-05': {
    #  '2300_G_34.92932099830986_psnr.pth': ()
    #  }
    #  }
    result_dict = {
        'RRDB_ESRGAN_x4_QuickBird_spsf_reduced_psnr_1_1_0.01_0.0001_1e-05': {
            '2460_G_35.818132392418725_psnr.pth': ()
        }
    }

    for setting, ckpt_dict in result_dict.items():
        setting_list = setting.split('_')
        opt['train_type'] = setting_list[4]
        opt['dataset_type'] = 'full'  # setting_list[5]
        opt['paradigm'] = f"{opt['train_type']}_{opt['dataset_type']}"
        # XXX use full dataset only
        opt['datasets']['val']['dataroot_HR'] = opt['datasets']['val'][
            'dataroot_LR'] = "code/BasicSR/data_samples/IKONOS/full/"  #"data/opticalRSI/QuickBird/full/"
        opt['datasets']['val']['data_type'] = 'lmdb'
        tmp = os.path.join(exp_dir, setting, 'models')
        for ckpt in ckpt_dict:
            # create model and load ckpt
            opt['path']['pretrain_model_G'] = os.path.join(tmp, ckpt)
            model = create_model(opt, rsi=True)
            # test dataloader
            test_loaders = get_dataloader(opt, logger)
            for test_loader in test_loaders:
                #  print(dir(test_loader.dataset))
                #  print(test_loader.dataset.root)
                #  break
                scores = test(opt, logger, test_loader, model, show=True)
                #  result_dict[setting][ckpt] = scores
    # store
    #  store_path = os.path.join(opt['path']['results_root'], 'result_dict.json')
    #  with open(store_path, 'w') as f_result:
    #  json.dump(result_dict, f_result)


if __name__ == '__main__':
    # evaluation score
    #  test_flow()
    # generate image
    test_show()
