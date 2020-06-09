"""BasicSR but for pansharpening
Author: Cheney
E-mail: zhoucslyx@gmail.com
Reference: xinn/BasicSR"""

import logging
#  import time
import math
import sys
import os.path
import glob
import random
#  import pdb

import torch
import visdom

# local or gdrive
GDRIVE = False
DRIVE_ROOT = '/content/gdrive/My Drive' if GDRIVE else os.getenv('HOME')
sys.path.append(os.path.join(DRIVE_ROOT, 'code/BasicSR/codes'))

import options.options as option
from data import create_dataloader, create_dataset
from models import create_model
from utils import util


def resume_logger(opt):
    #  parser = argparse.ArgumentParser()
    #  parser.add_argument(
    #  '-opt', type=str, required=True, help='Path to option JSON file.')
    #  opt = option.parse(json_path, is_train=True)
    #  opt = option.dict_to_nonedict(
    #  opt)  # Convert to NoneDict, which return None for missing key.
    # train from scratch OR resume training
    if opt['path']['resume_state']:  # resuming training
        resume_state = torch.load(opt['path']['resume_state'])
    else:  # training from scratch
        resume_state = None
        util.mkdir_and_rename(
            opt['path']['experiments_root'])  # rename old folder if exists
        util.mkdirs((path for key, path in opt['path'].items()
                     if not key == 'experiments_root'
                     and 'pretrain_model' not in key and 'resume' not in key))
    # config loggers. Before it, the log will not work
    util.setup_logger(None,
                      opt['path']['log'],
                      'train',
                      level=logging.INFO,
                      screen=True)
    util.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    return resume_state, logger


def init_vis(opt):
    '''Initialize visdom
    Visdom: no tex-format syntax, but html-format syntax'''
    vis = visdom.Visdom()
    archG = opt['network_G']['which_model_G']
    archR = opt['network_R']['which_model_R']
    archD = opt['network_D']['which_model_D']
    if opt['dataset_type'] == 'reduced':
        vis_titles = {
            'settings': 'settings',
            'netG': f'netG_{archG}',
            'netR': f'netR_{archR}',
            'netD': f'netD_{archD}',
            'lr': 'Learning Rate',
            'val_psnr': 'Avg_PSNR',
            'val_ssim': 'Avg_SSIM'
        }
    elif opt['dataset_type'] == 'full':
        vis_titles = {
            'settings': 'settings',
            'netG': f'netG_{archG}',
            'netR': f'netR_{archR}',
            'netD': f'netD_{archD}',
            'lr': 'Learning Rate',
            'val_no_ref': 'Avg_D_lambda_s_qnr',
        }
    vis_legends = {
        #  'netG': [
        #  r'$l_{\mathrm{pixel}}$', r'$l_{\mathrm{feat}}$',
        #  r'$l_{\mathrm{GAN}}$'
        #  ],
        'netG': ['pixel', 'feature', 'GAN'],
        #  'netR': [r'$l_{\mathrm{pixel}}$'],
        'netR': ['pixel'],
        #  '$l_{\mathrm{feat}}'],
        #  'netD': [
        #  r'$l_{\mathrm{real}}$', r'$l_{\mathrm{fake}}$',
        #  r'$D_{\mathrm{real}}$', r'$D_{\mathrm{fake}}$'
        #  ],
        'netD': ['real', 'fake', 'D_real', 'D_fake'],
        'lr': ['netG', 'netR', 'netD'],
        'val_no_ref': ['D_lambda', 'D_s', 'QNR']
    }
    vis_plots = {}
    for key, title in vis_titles.items():
        if key == 'settings':
            texts = [
                f"Paradigm: {opt['paradigm']}", f"Model: {opt['name']}",
                f"Scale: {opt['scale']}",
                f"HR_size: {opt['datasets']['train']['HR_size']}",
                f"BatchSize: {opt['datasets']['train']['batch_size']}",
                f"NumIters: {opt['train']['niter']}",
                f"Order: {opt['train']['order']}",
                f"Pretrained: {opt['path']['init']}",
                f"""Loss hyperparas: {opt['train']['pixel_weight']},
                {opt['train']['feature_weight']},
                {opt['train']['gan_weight']}""",
                f"""LR: {opt['train']['lr_G']},
                {opt['train']['lr_R']},
                {opt['train']['lr_D']}"""
            ]
            # visdom text supports html-format string
            vis_plots[key] = vis.text('<br /> '.join(texts),
                                      opts=dict(title=title))
        elif key in ('val_psnr', 'val_ssim'):
            vis_plots[key] = vis.line(X=torch.zeros((1, )).cpu(),
                                      Y=torch.zeros((1, )).cpu(),
                                      opts=dict(xlabel='Iteration',
                                                ylabel=title,
                                                title=title))
        else:
            if key in ('lr', 'val_no_ref'):
                ylabel = title
            else:
                ylabel = 'Loss'
            vis_plots[key] = util.create_vis_plot(vis, 'Iteration', ylabel,
                                                  title, vis_legends[key])
    return vis, vis_plots


def get_dataloader(opt, logger):
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(
                math.ceil(len(train_set) / dataset_opt['batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                len(train_set), train_size))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            logger.info('Total epochs: {:d} and total iters: {:,d}'.format(
                total_epochs, total_iters))
            train_loader = create_dataloader(train_set, dataset_opt)
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            logger.info('Number of val images in [{:s}]: {:d}'.format(
                dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError(
                'Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None
    return train_loader, val_loader, total_iters, total_epochs


def train(opt, train_data, current_step, epoch, model, logger, vis, vis_plots):
    """ trainning with a batch of data and logging & visdom"""
    # update learning rate
    model.update_learning_rate()
    # feed data
    model.feed_data(train_data)
    # training order
    if opt['train']['order'] == 'G_RD':
        model.optimize_parameters_G_RD(current_step)
    elif opt['train']['order'] == 'GR_D':
        model.optimize_parameters_GR_D(current_step)
    elif opt['train']['order'] == 'GD_R':
        model.optimize_parameters_GD_R(current_step)
    else:
        raise NotImplementedError(
            f"Unrecognized order: {opt['train']['order']}")

    # log
    if current_step % opt['logger']['print_freq'] == 0:
        logs = model.get_current_log()
        # logger
        message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
            epoch, current_step, model.get_current_learning_rate())
        for k, v in logs.items():
            message += '{:s}: {:.4e} '.format(k, v)
        logger.info(message)
        # visdom
        if opt['use_vis_logger'] and 'debug' not in opt['name']:
            # G
            if opt['train']['pixel_weight'] == 0:
                logs['l_g_pix'] = 0
            if opt['train']['feature_weight'] == 0:
                logs['l_g_feat'] = 0
            if opt['train']['gan_weight'] == 0:
                logs['l_g_gan'] = 0
            loss_g = [logs['l_g_pix'], logs['l_g_feat'], logs['l_g_gan']]
            util.update_vis(vis, vis_plots['netG'], current_step, *loss_g)
            # R
            if opt['train']['pixel_weight'] == 0 or opt['train_type'] == 'spsf':
                logs['l_r_pix'] = 0
            loss_r = [logs['l_r_pix']]
            util.update_vis(vis, vis_plots['netR'], current_step, *loss_r)
            # D
            if opt['train']['gan_weight'] == 0:
                loss_d = [0, 0, 0, 0]
            else:
                loss_d = [
                    logs['l_d_real'], logs['l_d_fake'], logs['D_real'],
                    logs['D_fake']
                ]
            util.update_vis(vis, vis_plots['netD'], current_step, *loss_d)
            # learning rate
            if opt['train_type'] == 'spuf':
                lrs = [
                    model.optimizer_G.param_groups[0]['lr'],
                    model.optimizer_R.param_groups[0]['lr'],
                    model.optimizer_D.param_groups[0]['lr']
                ]
            elif opt['train_type'] == 'spsf':
                lrs = [
                    model.optimizer_G.param_groups[0]['lr'], 0,
                    model.optimizer_D.param_groups[0]['lr']
                ]
            util.update_vis(vis, vis_plots['lr'], current_step, *lrs)


def valid(opt, val_loader, current_step, epoch, model, logger, vis, vis_plots):
    """Validation -> get PSNR and SSIM or D_labmda D_s QNR"""
    range_max = val_loader.dataset.dynamic_range
    crop_size = opt['scale']
    # calculate psnr and ssim
    if opt['dataset_type'] == 'reduced':
        avg_psnr = 0.0
        avg_ssim = 0.0
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
        cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
        # "Reference" image
        if opt['dataset_type'] == 'reduced':
            gt_img = util.tensor2img(visuals['HRx'],
                                     dynamic_range=range_max)  # uint
            cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:
                                    -crop_size, :]
            avg_psnr += util.calculate_psnr(cropped_sr_img, cropped_gt_img,
                                            range_max)
            avg_ssim += util.calculate_ssim(cropped_sr_img, cropped_gt_img,
                                            range_max)
        elif opt['dataset_type'] == 'full':
            # lr_x
            lr_x_img = util.tensor2img(visuals['LRx'], dynamic_range=range_max)
            # lr_p
            lr_p_img = util.tensor2img(visuals['LRp'], dynamic_range=range_max)
            now_qnr, now_D_lambda, now_D_s = util.qnr(sr_img,
                                                      lr_x_img,
                                                      lr_p_img,
                                                      satellite='QuickBird',
                                                      scale=4,
                                                      block_size=32,
                                                      p=1,
                                                      q=1,
                                                      alpha=1,
                                                      beta=1)
            avg_D_lambda += now_D_lambda
            avg_D_s += now_D_s
            avg_qnr += now_qnr
        #  if idx == 10:
        #  break
    if opt['dataset_type'] == 'reduced':
        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx
        # log
        logger.info('# Validation # PSNR: {:.4e} # SSIM: {:.4e}'.format(
            avg_psnr, avg_ssim))
        # logger for validation only
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info(
            '<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e} ssim: {:.4e}'.format(
                epoch, current_step, avg_psnr, avg_ssim))
        # visdom
        if opt['use_vis_logger'] and 'debug' not in opt['name']:
            util.update_vis(vis, vis_plots['val_psnr'], current_step, avg_psnr)
            util.update_vis(vis, vis_plots['val_ssim'], current_step, avg_ssim)
        return avg_psnr, avg_ssim
    if opt['dataset_type'] == 'full':
        avg_D_lambda = avg_D_lambda / idx
        avg_D_s = avg_D_s / idx
        avg_qnr = avg_qnr / idx
        # log
        logger.info(
            '# Validation # D_lambda: {:.4e} # D_s: {:.4e} # QNR: {:.4e}'.
            format(avg_D_lambda, avg_D_s, avg_qnr))
        # logger for validation only
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info(
            '<epoch:{:3d}, iter:{:8,d}> D_lambda: {:.4e} D_s: {:.4e} QNR: {:.4e}'
            .format(epoch, current_step, avg_D_lambda, avg_D_s, avg_qnr))
        # visdom
        if opt['use_vis_logger'] and 'debug' not in opt['name']:
            util.update_vis(vis, vis_plots['val_no_ref'], current_step,
                            *(avg_D_lambda, avg_D_s, avg_qnr))
        return avg_D_lambda, avg_D_s, avg_qnr


def storage(opt, val_loader, current_step, model, store=None):
    """store generated images or not, and this takes lots of time!
    store = None, nothing would happen
    store = (this_psnr, highest_psnr, this_ssim, highest_ssim), if one
    this > high, 5 images would be stored
    or sotre = (this_D_lambda, highest_D_labmda, this_D_s, highest_D_s,
    this_qnr, highest_qnr)"""
    if store is None:
        return
    if not isinstance(store, (list, tuple)):
        raise TypeError('store should be a list or tuple')
    if len(store) not in (4, 6):
        raise ValueError('store dimension should be 4 or 6')
    if len(store) == 4 and store[0] <= store[1] and store[2] <= store[3]:
        return
    if len(store) == 6 and store[0] >= store[1] and store[2] >= store[
            3] and store[4] <= store[5]:
        return
    range_max = val_loader.dataset.dynamic_range
    # calculate psnr and ssim
    idx = 0
    for val_data in val_loader:
        idx += 1
        # forward
        model.feed_data(val_data)
        model.test()
        visuals = model.get_current_visuals()
        sr_img = util.tensor2img(visuals['SR'],
                                 dynamic_range=range_max)  # uint
        # store
        if idx <= 5:
            if opt['dataset_type'] == 'reduced':
                img_name = val_data['bicLRx4']['MUL'][1][0]
            elif opt['dataset_type'] == 'full':
                img_name = val_data['HR']['MUL'][1][0]
            img_dir = os.path.join(opt['path']['val_images'], img_name)
            util.mkdir(img_dir)
            # Save SR images for reference
            save_img_path = os.path.join(
                img_dir, '{:s}_{:d}'.format(img_name, current_step))
            util.save_img(sr_img, save_img_path)


def main(opt, resume_state, logger):
    # resume
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))
        option.check_resume(opt)  # check resume options
    logger.info(option.dict2str(opt))
    # Initialize visdom
    if opt['use_vis_logger'] and 'debug' not in opt['name']:
        vis, vis_plots = init_vis(opt)
    else:
        vis, vis_plots = None, None
    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)
    # create train and val dataloader
    train_loader, val_loader, total_iters, total_epochs = get_dataloader(
        opt, logger)
    # cudnn
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    # create model
    model = create_model(opt)
    # resume training
    if resume_state:
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0
    # training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(
        start_epoch, current_step))

    # highest_psnr, highest_ssim for model saving
    if opt['dataset_type'] == 'reduced':
        highest_psnr = highest_ssim = 0
    elif opt['dataset_type'] == 'full':
        lowest_D_labmda = lowest_D_s = 1
        highest_qnr = 0
    for epoch in range(start_epoch, total_epochs):
        for _, train_data in enumerate(train_loader):
            # current step
            current_step += 1
            if current_step > total_iters:
                break
            # training
            train(opt, train_data, current_step, epoch, model, logger, vis,
                  vis_plots)
            # validation
            if current_step % opt['train']['val_freq'] == 0:
                # it will write generated images to disk
                if opt['dataset_type'] == 'reduced':
                    this_psnr, this_ssim = valid(opt, val_loader, current_step,
                                                 epoch, model, logger, vis,
                                                 vis_plots)
                    # storage images take lots of time
                    #  storage(opt,
                    #  val_loader,
                    #  current_step,
                    #  model,
                    #  store=(this_psnr, highest_psnr, this_ssim,
                    #  highest_ssim))
                elif opt['dataset_type'] == 'full':
                    this_D_lambda, this_D_s, this_qnr = valid(
                        opt, val_loader, current_step, epoch, model, logger,
                        vis, vis_plots)
                    # storage images take lots of time
                    #  storage(opt,
                    #  val_loader,
                    #  current_step,
                    #  model,
                    #  store=(this_D_lambda, lowest_D_labmda, this_D_s,
                    #  lowest_D_s, this_qnr, highest_qnr))
            # save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if opt['dataset_type'] == 'reduced':
                    if this_psnr > highest_psnr:
                        logger.info(
                            'Saving models and training states with highest psnr.'
                        )
                        # remove the old
                        old_model = glob.glob(opt['path']['models'] +
                                              '/*_psnr.pth')
                        old_state = glob.glob(opt['path']['training_state'] +
                                              '/*_psnr.state')
                        for ele in old_model:
                            os.remove(ele)
                        for ele in old_state:
                            os.remove(ele)
                        # save the new
                        model.save(current_step, this_psnr, 'psnr')
                        model.save_training_state(epoch, current_step, 'psnr')
                        highest_psnr = this_psnr
                    if this_ssim > highest_ssim:
                        logger.info(
                            'Saving models and training states with highest ssim.'
                        )
                        # remove the old
                        old_model = glob.glob(opt['path']['models'] +
                                              '/*_ssim.pth')
                        old_state = glob.glob(opt['path']['training_state'] +
                                              '/*_ssim.state')
                        for ele in old_model:
                            os.remove(ele)
                        for ele in old_state:
                            os.remove(ele)
                        model.save(current_step, this_ssim, 'ssim')
                        model.save_training_state(epoch, current_step, 'ssim')
                        highest_ssim = this_ssim
                elif opt['dataset_type'] == 'full':
                    if this_D_lambda < lowest_D_labmda:
                        logger.info(
                            'Saving models and training states with lowest D_lambda.'
                        )
                        # remove the old
                        old_model = glob.glob(opt['path']['models'] +
                                              '/*_D_lambda.pth')
                        old_state = glob.glob(opt['path']['training_state'] +
                                              '/*_D_lambda.state')
                        for ele in old_model:
                            os.remove(ele)
                        for ele in old_state:
                            os.remove(ele)
                        # save the new
                        model.save(current_step, this_D_lambda, 'D_lambda')
                        model.save_training_state(epoch, current_step,
                                                  'D_lambda')
                        lowest_D_labmda = this_D_lambda
                    if this_D_s < lowest_D_s:
                        logger.info(
                            'Saving models and training states with lowest D_s.'
                        )
                        # remove the old
                        old_model = glob.glob(opt['path']['models'] +
                                              '/*_D_s.pth')
                        old_state = glob.glob(opt['path']['training_state'] +
                                              '/*_D_s.state')
                        for ele in old_model:
                            os.remove(ele)
                        for ele in old_state:
                            os.remove(ele)
                        model.save(current_step, this_D_s, 'D_s')
                        model.save_training_state(epoch, current_step, 'D_s')
                        lowest_D_s = this_D_s
                    if this_qnr > highest_qnr:
                        logger.info(
                            'Saving models and training states with higest QNR'
                        )
                        # remove the old
                        old_model = glob.glob(opt['path']['models'] +
                                              '/*_qnr.pth')
                        old_state = glob.glob(opt['path']['training_state'] +
                                              '/*_qnr.state')
                        for ele in old_model:
                            os.remove(ele)
                        for ele in old_state:
                            os.remove(ele)
                        model.save(current_step, this_qnr, 'qnr')
                        model.save_training_state(epoch, current_step, 'qnr')
                        highest_qnr = this_qnr
    # save the last state
    #  logger.info('Saving the final model.')
    #  model.save('latest')
    logger.info('End of training.')


def train_flow():
    """Adjustable arguments"""
    #  json_path = os.path.join(
    #  DRIVE_ROOT, 'code/BasicSR/codes/options/train/train_ESRGAN.json')
    #  opt, resume_state, logger = get_options(json_path)
    #  main(opt, resume_state, logger)
    # for loop
    paradigms = ['spuf_reduced', 'spsf_reduced', 'spuf_full']
    inits = [
        ('random', None),
        ('psnr',
         os.path.join(
             DRIVE_ROOT,
             "code/BasicSR/experiments/pretrained_models/RRDB_PSNR_x4_no_0_10.pth"
         )),
        ('esrgan',
         os.path.join(
             DRIVE_ROOT,
             "code/BasicSR/experiments/pretrained_models/RRDB_ESRGAN_x4_no_0_10.pth"
         ))
    ]
    # loss weights
    ABGs = [(1, 0, 0), (0, 1, 0.01), (1, 1, 0.01)]
    # lr for G and D
    all_lrs = [(1e-4, 0), (1e-5, 0), (1e-4, 1e-4), (1e-4, 1e-5), (1e-5, 1e-4)]
    # init opt
    json_path = os.path.join(
        DRIVE_ROOT, 'code/BasicSR/codes/options/train/train_ESRGAN.json')
    is_train = True
    opt = option.parse(json_path, is_train)
    opt = option.dict_to_nonedict(opt)
    #  correct opt, and get resume_state, logger
    for paradigm in paradigms:
        opt['paradigm'] = paradigm
        train_type, dataset_type = paradigm.split('_')
        if train_type == 'spsf':
            opt['network_D']['which_model_D'] = 'discriminator_vgg_256'
        elif train_type == 'spuf':
            opt['network_D']['which_model_D'] = 'discriminator_vgg_64_256'
        opt['train_type'] = train_type
        opt['dataset_type'] = dataset_type
        opt['datasets']['train']['dataroot_HR'] = opt['datasets']['train'][
            'dataroot_LR'] = f"data/opticalRSI/QuickBird/{dataset_type}/"
        opt['datasets']['val']['dataroot_HR'] = opt['datasets']['val'][
            'dataroot_LR'] = f"data/opticalRSI/QuickBird/{dataset_type}/"
        for init in inits:
            opt['path']['init'] = init[0]
            opt['path']['pretrain_model_G'] = init[1]
            for alpha, beta, gamma in ABGs:
                opt['train']['pixel_weight'] = alpha
                opt['train']['feature_weight'] = beta
                opt['train']['gan_weight'] = gamma
                #
                if paradigm != 'spuf_full' or init[0] != 'psnr':
                    continue
                if (alpha, beta, gamma) == (1, 0, 0):
                    continue
                    lrs = all_lrs[:2]
                elif (alpha, beta, gamma) == (0, 1, 0.01):
                    continue
                else:  # (1, 1, 0.01)
                    lrs = all_lrs[2:]
                for lrG, lrD in lrs:
                    opt['train']['lr_G'] = lrG
                    opt['train']['lr_D'] = lrD
                    # correct path
                    storage_folder = '_'.join([
                        opt['name'], opt['paradigm'], opt['path']['init'],
                        str(opt['train']['pixel_weight']),
                        str(opt['train']['feature_weight']),
                        str(opt['train']['gan_weight']),
                        str(opt['train']['lr_G']),
                        str(opt['train']['lr_D'])
                    ])
                    if is_train:
                        experiments_root = os.path.join(
                            opt['path']['root'], 'experiments', storage_folder)
                        opt['path']['experiments_root'] = experiments_root
                        opt['path']['models'] = os.path.join(
                            experiments_root, 'models')
                        opt['path']['training_state'] = os.path.join(
                            experiments_root, 'training_state')
                        opt['path']['log'] = experiments_root
                        opt['path']['val_images'] = os.path.join(
                            experiments_root, 'val_images')
                    else:
                        results_root = os.path.join(opt['path']['root'],
                                                    'results', storage_folder)
                        opt['path']['results_root'] = results_root
                        opt['path']['log'] = results_root
                    resume_state, logger = resume_logger(opt)
                    # train happily
                    main(opt, resume_state, logger)


if __name__ == '__main__':
    train_flow()
