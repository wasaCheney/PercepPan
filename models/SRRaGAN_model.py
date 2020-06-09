#  import os
import logging
from collections import OrderedDict

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from models.base_model import BaseModel
from models.modules.loss import GANLoss, GradientPenaltyLoss
logger = logging.getLogger('base')


class SRRaGANModel(BaseModel):
    def __init__(self, opt):
        super(SRRaGANModel, self).__init__(opt)
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)  # G
        if self.is_train:
            self.netD = networks.define_D(opt).to(self.device)  # D
            self.netG.train()
            self.netD.train()
        self.load()  # load G, R and D if needed

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError(
                        'Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError(
                        'Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt,
                                              use_bn=False).to(self.device)

            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0,
                                   0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            # D_update_ratio and D_init_iters are for WGAN
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt[
                'D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt[
                'D_init_iters'] else 0

            if train_opt['gan_type'] == 'wgan-gp':
                self.random_pt = torch.Tensor(1, 1, 1, 1).to(self.device)
                # gradient penalty loss
                self.cri_gp = GradientPenaltyLoss(device=self.device).to(
                    self.device)
                self.l_gp_w = train_opt['gp_weigth']

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt[
                'weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters(
            ):  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning(
                        'Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params,
                                                lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1_G'],
                                                       0.999))
            self.optimizers.append(self.optimizer_G)
            # D
            wd_D = train_opt['weight_decay_D'] if train_opt[
                'weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=train_opt['lr_D'],
                                                weight_decay=wd_D,
                                                betas=(train_opt['beta1_D'],
                                                       0.999))
            self.optimizers.append(self.optimizer_D)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR(optimizer,
                                                 train_opt['lr_steps'],
                                                 train_opt['lr_gamma']))
            else:
                raise NotImplementedError(
                    'MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
        # print network
        self.print_network()

    def feed_data(self, data, need_HR=True):
        # LR
        self.var_L = data['LR'].to(self.device)

        if need_HR:  # train or val
            self.var_H = data['HR'].to(self.device)

            input_ref = data['ref'] if 'ref' in data else data['HR']
            self.var_ref = input_ref.to(self.device)

    def optimize_parameters(self, step):
        # G
        for p in self.netD.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()

        self.fake_H = self.netG(self.var_L)

        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                l_g_total += l_g_pix
            if self.cri_fea:  # feature loss
                real_fea = self.netF(self.var_H).detach()
                fake_fea = self.netF(self.fake_H)
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fea
            # G gan + cls loss
            pred_g_fake = self.netD(self.fake_H)
            pred_d_real = self.netD(self.var_ref).detach()

            l_g_gan = self.l_gan_w * (
                self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
            l_g_total += l_g_gan

            l_g_total.backward()
            self.optimizer_G.step()

        # D
        for p in self.netD.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()
        l_d_total = 0
        pred_d_real = self.netD(self.var_ref)
        pred_d_fake = self.netD(
            self.fake_H.detach())  # detach to avoid BP to G
        l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
        l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)

        l_d_total = (l_d_real + l_d_fake) / 2

        if self.opt['train']['gan_type'] == 'wgan-gp':
            batch_size = self.var_ref.size(0)
            if self.random_pt.size(0) != batch_size:
                self.random_pt.resize_(batch_size, 1, 1, 1)
            self.random_pt.uniform_()  # Draw random interpolation points
            interp = self.random_pt * self.fake_H.detach() + (
                1 - self.random_pt) * self.var_ref
            interp.requires_grad = True
            interp_crit, _ = self.netD(interp)
            l_d_gp = self.l_gp_w * self.cri_gp(
                interp, interp_crit)  # maybe wrong in cls?
            l_d_total += l_d_gp

        l_d_total.backward()
        self.optimizer_D.step()

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            # G
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            if self.cri_fea:
                self.log_dict['l_g_fea'] = l_g_fea.item()
            self.log_dict['l_g_gan'] = l_g_gan.item()
        # D
        self.log_dict['l_d_real'] = l_d_real.item()
        self.log_dict['l_d_fake'] = l_d_fake.item()

        if self.opt['train']['gan_type'] == 'wgan-gp':
            self.log_dict['l_d_gp'] = l_d_gp.item()
        # D outputs
        self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
        self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(
                self.netG.__class__.__name__,
                self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(
            net_struc_str, n))
        logger.info(s)
        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel):
                net_struc_str = '{} - {}'.format(
                    self.netD.__class__.__name__,
                    self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)

            logger.info(
                'Network D structure: {}, with parameters: {:,d}'.format(
                    net_struc_str, n))
            logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(
                        self.netF.__class__.__name__,
                        self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)

                logger.info(
                    'Network F structure: {}, with parameters: {:,d}'.format(
                        net_struc_str, n))
                logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(
                load_path_G))
            self.load_network(load_path_G, self.netG)
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading pretrained model for D [{:s}] ...'.format(
                load_path_D))
            self.load_network(load_path_D, self.netD)

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)


class PANRaGANModel(BaseModel):
    """Lpixel + alpha Lfeat + beta LGAN -> G;
    Lpixel -> R;
    LGAN -> D"""

    def __init__(self, opt):
        super().__init__(opt)
        # training paradigm
        self.train_type = opt['train_type']  # spuf, spsf
        # XXX only full dataset
        self.dataset_type = 'full'  # opt['dataset_type']  # reduced, full
        # satellite
        if opt['is_train']:
            self.satellite = opt['datasets']['train']['name']
        else:
            self.satellite = opt['datasets']['val']['name']
        if opt['is_train']:
            # train_opt
            train_opt = opt['train']
        # when to train netR
        if self.train_type == 'spuf':
            self.netR_ksize = 3  # it should be odd
            #  self.R_begin = 10**8  # int(train_opt['niter'] * 2 / 3)
            #  self.R_begin + int(np.sqrt(train_opt['niter']))
            #  self.R_end = 10**8 + 1
            self.R_fixed_weights = self._fixed_parameters_for_R()

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)  # G
        if self.is_train:
            self.netG.train()

            if self.train_type == 'spuf':
                self.netR = networks.define_R(opt).to(self.device)  # R
                self.netR.train()

            self.netD = networks.define_D(opt).to(self.device)  # D
            self.netD.train()
        self.load()  # load G and R if needed

        # define losses, optimizer and scheduler
        if self.is_train:
            # G/R pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError(
                        'Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            # G/R feature loss
            if train_opt['feature_weight'] > 0:
                l_feat_type = train_opt['feature_criterion']
                if l_feat_type == 'l1':
                    self.cri_feat = nn.L1Loss().to(self.device)
                elif l_feat_type == 'l2':
                    self.cri_feat = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError(
                        'Loss type [{:s}] not recognized.'.format(l_feat_type))
                self.l_feat_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_feat = None
            #  if self.cri_fea:  # load VGG perceptual loss
            #  self.netF = networks.define_F(
            #  opt, use_bn=False).to(self.device)

            # G/D gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0,
                                   0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            # D_update_ratio and D_init_iters are for WGAN
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt[
                'D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt[
                'D_init_iters'] else 0

            if train_opt['gan_type'] == 'wgan-gp':
                self.random_pt = torch.Tensor(1, 1, 1, 1).to(self.device)
                # gradient penalty loss
                self.cri_gp = GradientPenaltyLoss(device=self.device).to(
                    self.device)
                self.l_gp_w = train_opt['gp_weight']

            # optimizers
            # G optim
            wd_G = train_opt['weight_decay_G'] if train_opt[
                'weight_decay_G'] else 0
            #  optim_params = [] # optim part of parameters of G
            #  for k, v in self.netG.named_parameters():
            #  if v.requires_grad:
            #  optim_params.append(v)
            #  else:
            #  logger.warning(
            #  'Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(
                #  optim_params,
                self.netG.parameters(),
                lr=train_opt['lr_G'],
                weight_decay=wd_G,
                betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)
            # R optim
            if self.train_type == 'spuf':
                wd_R = train_opt['weight_decay_R'] if train_opt[
                    'weight_decay_R'] else 0
                self.optimizer_R = torch.optim.Adam(
                    self.netR.parameters(),
                    lr=train_opt['lr_R'],
                    weight_decay=wd_R,
                    betas=(train_opt['beta1_R'], 0.999))
                self.optimizers.append(self.optimizer_R)
            # D optim
            wd_D = train_opt['weight_decay_D'] if train_opt[
                'weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=train_opt['lr_D'],
                                                weight_decay=wd_D,
                                                betas=(train_opt['beta1_D'],
                                                       0.999))
            self.optimizers.append(self.optimizer_D)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR(optimizer,
                                                 train_opt['lr_steps'],
                                                 train_opt['lr_gamma']))
            else:
                raise NotImplementedError(
                    'MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
        # print network
        self.print_network()

    def feed_data(self, data, need_HR=True):
        LR_type = 'bicLRx4'
        HR_type = 'HR'
        if self.dataset_type == 'reduced':
            self.lr_x = data[LR_type]['MUL'][0].to(self.device)
            self.lr_p = data[LR_type]['PAN'][0].to(self.device)
            if need_HR:  # train or val
                self.hr_x = data[HR_type]['MUL'][0].to(self.device)
                self.hr_p = data[HR_type]['PAN'][0].to(self.device)
        elif self.dataset_type == 'full':
            # in this case, only full-scale images are available
            self.lr_x = data[HR_type]['MUL'][0].to(self.device)
            self.lr_p = data[HR_type]['PAN'][0].to(self.device)

    def optimize_parameters_G_RD(self, step, before_fusion=False):
        """Fisrt G, then R D; wrong, but not used anymore"""
        # G
        for p in self.netR.parameters():
            p.requires_grad = False
        for p in self.netD.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()

        self.lr_x_rec, self.lr_p_rec = self.netR(
            self.netG(self.lr_x, self.lr_p))

        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            # pixel loss
            if self.cri_pix:
                l_g_pix_x = self.l_pix_w * self.cri_pix(
                    self.lr_x_rec, self.lr_x)
                l_g_pix_p = self.l_pix_w * self.cri_pix(
                    self.lr_p_rec, self.lr_p)
                l_g_pix = l_g_pix_x + l_g_pix_p
                l_g_total += l_g_pix
                #  print(l_g_total.requires_grad)
            # feature loss
            with torch.no_grad():
                real_feat, real_score = self.netD(self.lr_x,
                                                  self.lr_p,
                                                  before_fus=before_fusion)
            fake_feat, fake_score = self.netD(self.lr_x_rec,
                                              self.lr_p_rec,
                                              before_fus=before_fusion)
            if self.cri_feat:
                l_g_feat = self.l_feat_w * self.cri_feat(fake_feat, real_feat)
                l_g_total += l_g_feat
            #  print(l_g_feat.requires_grad)
            # gan loss
            l_g_gan = self.l_gan_w * (
                self.cri_gan(real_score - torch.mean(fake_score), False) +
                self.cri_gan(fake_score - torch.mean(real_score), True)) / 2
            l_g_total += l_g_gan
            #  print(l_g_gan.requires_grad)

            l_g_total.backward()
            self.optimizer_G.step()

        # New G to generate HR
        with torch.no_grad():
            self.hr_y = self.netG(self.lr_x, self.lr_p)
            self.lr_x_rec_d, self.lr_p_rec_d = self.netR(self.hr_y)

        # R
        self.hr_y.requires_grad = True
        for p in self.netR.parameters():
            p.requires_grad = True

        self.optimizer_R.zero_grad()

        self.lr_x_rec, self.lr_p_rec = self.netR(self.hr_y)
        l_r_total = 0
        # pixel loss
        l_r_pix_x = self.cri_pix(self.lr_x_rec, self.lr_x)
        l_r_pix_p = self.cri_pix(self.lr_p_rec, self.lr_p)
        l_r_pix = l_r_pix_x + l_r_pix_p
        l_r_total += l_r_pix

        l_r_total.backward()
        self.optimizer_R.step()

        # D
        for p in self.netD.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()

        _, fake_score = self.netD(self.lr_x_rec_d,
                                  self.lr_p_rec_d,
                                  before_fus=before_fusion)
        _, real_score = self.netD(self.lr_x,
                                  self.lr_p,
                                  before_fus=before_fusion)

        l_d_total = 0
        l_d_real = self.cri_gan(real_score - torch.mean(fake_score), True)
        l_d_fake = self.cri_gan(fake_score - torch.mean(real_score), False)

        l_d_total += (l_d_real + l_d_fake) / 2

        if self.opt['train']['gan_type'] == 'wgan-gp':
            batch_size = self.lr_x.size(0)
            if self.random_pt.size(0) != batch_size:
                self.random_pt.resize_(batch_size, 1, 1, 1)
            self.random_pt.uniform_()  # Draw random interpolation points
            interp_x = self.random_pt * self.lr_x_rec_d.detach() + (
                1 - self.random_pt) * self.lr_x
            interp_p = self.random_pt * self.lr_p_rec_d.detach() + (
                1 - self.random_pt) * self.lr_p
            interp_x.requires_grad = True
            interp_p.requires_grad = True
            _, interp_crit = self.netD(interp_x,
                                       interp_p,
                                       before_fus=before_fusion)
            l_d_gp_x = self.l_gp_w * self.cri_gp(
                interp_x, interp_crit)  # maybe wrong in cls?
            l_d_gp_p = self.l_gp_w * self.cri_gp(
                interp_p, interp_crit)  # maybe wrong in cls?
            l_d_gp = l_d_gp_x + l_d_gp_p
            l_d_total += l_d_gp

        l_d_total.backward()
        self.optimizer_D.step()

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            # G loss
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            if self.cri_feat:
                self.log_dict['l_g_feat'] = l_g_feat.item()
            self.log_dict['l_g_gan'] = l_g_gan.item()
        # R loss
        self.log_dict['l_r_pix'] = l_r_pix.item()
        # D loss
        self.log_dict['l_d_real'] = l_d_real.item()
        self.log_dict['l_d_fake'] = l_d_fake.item()
        if self.opt['train']['gan_type'] == 'wgan-gp':
            self.log_dict['l_d_gp'] = l_d_gp.item()
        # D outputs
        with torch.no_grad():
            self.log_dict['D_real'] = torch.mean(real_score)
            self.log_dict['D_fake'] = torch.mean(fake_score)

    def optimize_parameters_GR_D(self, step, before_fusion=False):
        """First G R, then D; wrong, but not used anymore"""
        # G
        for p in self.netR.parameters():
            p.requires_grad = False
        for p in self.netD.parameters():
            p.requires_grad = False
        # for R
        with torch.no_grad():
            self.hr_y_R = self.netG(self.lr_x, self.lr_p)

        self.optimizer_G.zero_grad()

        self.lr_x_rec, self.lr_p_rec = self.netR(
            self.netG(self.lr_x, self.lr_p))

        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            # pixel loss
            if self.cri_pix:
                l_g_pix_x = self.l_pix_w * self.cri_pix(
                    self.lr_x_rec, self.lr_x)
                l_g_pix_p = self.l_pix_w * self.cri_pix(
                    self.lr_p_rec, self.lr_p)
                l_g_pix = l_g_pix_x + l_g_pix_p
                l_g_total += l_g_pix
                #  print(l_g_total.requires_grad)
            # feature loss
            with torch.no_grad():
                real_feat, real_score = self.netD(self.lr_x,
                                                  self.lr_p,
                                                  before_fus=before_fusion)
            fake_feat, fake_score = self.netD(self.lr_x_rec,
                                              self.lr_p_rec,
                                              before_fus=before_fusion)
            if self.cri_feat:
                l_g_feat = self.l_feat_w * self.cri_feat(fake_feat, real_feat)
                l_g_total += l_g_feat
            #  print(l_g_feat.requires_grad)
            # gan loss
            l_g_gan = self.l_gan_w * (
                self.cri_gan(real_score - torch.mean(fake_score), False) +
                self.cri_gan(fake_score - torch.mean(real_score), True)) / 2
            l_g_total += l_g_gan
            #  print(l_g_gan.requires_grad)

            l_g_total.backward()
            self.optimizer_G.step()

        # R
        for p in self.netR.parameters():
            p.requires_grad = True
        self.hr_y_R.requires_grad = True

        self.optimizer_R.zero_grad()

        self.lr_x_rec, self.lr_p_rec = self.netR(self.hr_y_R)

        l_r_total = 0
        # pixel loss
        l_r_pix_x = self.cri_pix(self.lr_x_rec, self.lr_x)
        l_r_pix_p = self.cri_pix(self.lr_p_rec, self.lr_p)
        l_r_pix = l_r_pix_x + l_r_pix_p
        l_r_total += l_r_pix

        l_r_total.backward()
        self.optimizer_R.step()

        # D
        for p in self.netD.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()

        with torch.no_grad():
            self.lr_x_rec, self.lr_p_rec = self.netR(
                self.netG(self.lr_x, self.lr_p))
        _, fake_score = self.netD(self.lr_x_rec,
                                  self.lr_p_rec,
                                  before_fus=before_fusion)
        _, real_score = self.netD(self.lr_x,
                                  self.lr_p,
                                  before_fus=before_fusion)

        l_d_total = 0
        l_d_real = self.cri_gan(real_score - torch.mean(fake_score), True)
        l_d_fake = self.cri_gan(fake_score - torch.mean(real_score), False)

        l_d_total += (l_d_real + l_d_fake) / 2

        if self.opt['train']['gan_type'] == 'wgan-gp':
            batch_size = self.lr_x.size(0)
            if self.random_pt.size(0) != batch_size:
                self.random_pt.resize_(batch_size, 1, 1, 1)
            self.random_pt.uniform_()  # Draw random interpolation points
            interp_x = self.random_pt * self.lr_x_rec.detach() + (
                1 - self.random_pt) * self.lr_x
            interp_p = self.random_pt * self.lr_p_rec.detach() + (
                1 - self.random_pt) * self.lr_p
            interp_x.requires_grad = True
            interp_p.requires_grad = True
            _, interp_crit = self.netD(interp_x,
                                       interp_p,
                                       before_fus=before_fusion)
            l_d_gp_x = self.l_gp_w * self.cri_gp(
                interp_x, interp_crit)  # maybe wrong in cls?
            l_d_gp_p = self.l_gp_w * self.cri_gp(
                interp_p, interp_crit)  # maybe wrong in cls?
            l_d_gp = l_d_gp_x + l_d_gp_p
            l_d_total += l_d_gp

        l_d_total.backward()
        self.optimizer_D.step()

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            # G loss
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            if self.cri_feat:
                self.log_dict['l_g_feat'] = l_g_feat.item()
            self.log_dict['l_g_gan'] = l_g_gan.item()
        # R loss
        self.log_dict['l_r_pix'] = l_r_pix.item()
        # D loss
        self.log_dict['l_d_real'] = l_d_real.item()
        self.log_dict['l_d_fake'] = l_d_fake.item()
        if self.opt['train']['gan_type'] == 'wgan-gp':
            self.log_dict['l_d_gp'] = l_d_gp.item()
        # D outputs
        with torch.no_grad():
            self.log_dict['D_real'] = torch.mean(real_score)
            self.log_dict['D_fake'] = torch.mean(fake_score)

    def _fixed_parameters_for_R(self):
        """
        HRMS -> LRMS, fixed filter and downsampler
        HRMS -> PAN, fixed linear transform
        Ref: Remote Sensing Image Fusion via Sparse Representations
        Over Learned Dictionaries
        """
        if self.satellite == 'QuickBird':
            # HRMS -> PAN
            BGRNIR = (0.1139, 0.2315, 0.2308, 0.4239)
            # HRMS -> LRMS
            GNyq = (0.34, 0.32, 0.30, 0.22)
            # HRPAN -> LRPAN
            #  GNyqPan = 0.15
        elif self.satellite == 'IKONOS':
            BGRNIR = (0.1071, 0.2646, 0.2696, 0.3587)
            GNyq = (0.26, 0.28, 0.29, 0.28)
            #  GNyqPan = 0.17
        elif self.satellite == 'WorldView2':
            pass
            #  BGRNIR = None  # Parameters are unknown
            #  GNyq = (0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.27)
            #  GNyqPan = 0.11
        gaussian_kernels = []
        for ele in GNyq:
            gaussian_kernel = cv2.getGaussianKernel(self.netR_ksize,
                                                    1 / (2 * np.pi * ele))
            gaussian_kernels.append(np.outer(gaussian_kernel, gaussian_kernel))
        gaussian_kernels = np.expand_dims(np.stack(gaussian_kernels, axis=0),
                                          axis=1)  # for LM
        return torch.FloatTensor(gaussian_kernels).to(
            self.device), torch.FloatTensor(BGRNIR).reshape(1, -1, 1,
                                                            1).to(self.device)

    def optimize_parameters_GD_R(self, step, before_fusion=False):
        """Fixe R and train G D alternatively"""
        # R
        # initialization with customed weights
        if step in (0, 1) and self.train_type == 'spuf':
            for p, fixed_data in zip(self.netR.parameters(),
                                     self.R_fixed_weights):
                p.data = fixed_data
        if self.train_type == 'spuf':
            for p in self.netR.parameters():
                p.requires_grad = True
            with torch.no_grad():
                self.hr_y = self.netG(self.lr_x, self.lr_p)
            self.hr_y.requires_grad = True

            self.optimizer_R.zero_grad()

            self.lr_x_rec, self.lr_p_rec = self.netR(self.hr_y)

            l_r_total = 0
            # pixel loss
            if self.cri_pix:
                l_r_pix_x = self.cri_pix(self.lr_x_rec, self.lr_x)
                l_r_pix_p = self.cri_pix(self.lr_p_rec, self.lr_p)
                l_r_pix = l_r_pix_x + l_r_pix_p
                l_r_total += l_r_pix
                #  if step in range(self.R_begin, self.R_end + 1):
                #  l_r_total.backward()
                #  self.optimizer_R.step()
                #  else:
                #  pass

        # G
        if self.train_type == 'spuf':
            for p in self.netR.parameters():
                p.requires_grad = False
        for p in self.netD.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()

        self.hr_y = self.netG(self.lr_x, self.lr_p)
        if self.train_type == 'spuf':
            self.lr_x_rec, self.lr_p_rec = self.netR(self.hr_y)

        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            # pixel loss
            if self.cri_pix:
                if self.train_type == 'spuf':
                    l_g_pix_x = self.l_pix_w * self.cri_pix(
                        self.lr_x_rec, self.lr_x)
                    l_g_pix_p = self.l_pix_w * self.cri_pix(
                        self.lr_p_rec, self.lr_p)
                    l_g_pix = l_g_pix_x + l_g_pix_p
                elif self.train_type == 'spsf':
                    l_g_pix = self.l_pix_w * self.cri_pix(self.hr_y, self.hr_x)
                l_g_total += l_g_pix
                #  print(l_g_total.requires_grad)
            # feature loss
            with torch.no_grad():
                if self.train_type == 'spuf':
                    real_feat, real_score = self.netD(self.lr_x,
                                                      self.lr_p,
                                                      before_fus=before_fusion)
                elif self.train_type == 'spsf':
                    real_feat, real_score = self.netD(self.hr_x)
            if self.train_type == 'spuf':
                fake_feat, fake_score = self.netD(self.lr_x_rec,
                                                  self.lr_p_rec,
                                                  before_fus=before_fusion)
            elif self.train_type == 'spsf':
                fake_feat, fake_score = self.netD(self.hr_y)
            if self.cri_feat:
                l_g_feat = self.l_feat_w * self.cri_feat(fake_feat, real_feat)
                l_g_total += l_g_feat
            #  print(l_g_feat.requires_grad)
            # gan loss
            l_g_gan = self.l_gan_w * (
                self.cri_gan(real_score - torch.mean(fake_score), False) +
                self.cri_gan(fake_score - torch.mean(real_score), True)) / 2
            l_g_total += l_g_gan
            #  print(l_g_gan.requires_grad)
            #  if step in range(self.R_begin, self.R_end + 1):
            #  pass
            #  else:
            l_g_total.backward()
            self.optimizer_G.step()

        # D
        for p in self.netD.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()

        with torch.no_grad():
            self.hr_y = self.netG(self.lr_x, self.lr_p)
            if self.train_type == 'spuf':
                self.lr_x_rec, self.lr_p_rec = self.netR(self.hr_y)
        if self.train_type == 'spuf':
            _, fake_score = self.netD(self.lr_x_rec,
                                      self.lr_p_rec,
                                      before_fus=before_fusion)
            _, real_score = self.netD(self.lr_x,
                                      self.lr_p,
                                      before_fus=before_fusion)
        elif self.train_type == 'spsf':
            _, fake_score = self.netD(self.hr_y)
            _, real_score = self.netD(self.hr_x)

        l_d_total = 0
        l_d_real = self.cri_gan(real_score - torch.mean(fake_score), True)
        l_d_fake = self.cri_gan(fake_score - torch.mean(real_score), False)

        l_d_total += (l_d_real + l_d_fake) / 2

        if self.opt['train']['gan_type'] == 'wgan-gp':
            batch_size = self.lr_x.size(0)
            if self.random_pt.size(0) != batch_size:
                self.random_pt.resize_(batch_size, 1, 1, 1)
            self.random_pt.uniform_()  # Draw random interpolation points
            if self.train_type == 'spuf':
                interp_x = self.random_pt * self.lr_x_rec.detach() + (
                    1 - self.random_pt) * self.lr_x
                interp_p = self.random_pt * self.lr_p_rec.detach() + (
                    1 - self.random_pt) * self.lr_p
                interp_x.requires_grad = True
                interp_p.requires_grad = True
                _, interp_crit = self.netD(interp_x,
                                           interp_p,
                                           before_fus=before_fusion)
                l_d_gp_x = self.l_gp_w * self.cri_gp(
                    interp_x, interp_crit)  # maybe wrong in cls?
                l_d_gp_p = self.l_gp_w * self.cri_gp(
                    interp_p, interp_crit)  # maybe wrong in cls?
                l_d_gp = l_d_gp_x + l_d_gp_p
            elif self.train_type == 'spsf':
                interp_y = self.random_pt * self.hr_y.detach() + (
                    1 - self.random_pt) * self.hr_x
                interp_y.requires_grad = True
                _, interp_crit = self.netD(interp_y)
                l_d_gp_x = self.l_gp_w * self.cri_gp(
                    interp_y, interp_crit)  # maybe wrong in cls?
                l_d_gp = l_d_gp_x
            l_d_total += l_d_gp
        #  if step in range(self.R_begin, self.R_end + 1):
        #  pass
        #  else:
        l_d_total.backward()
        self.optimizer_D.step()

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            # G loss
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            if self.cri_feat:
                self.log_dict['l_g_feat'] = l_g_feat.item()
            self.log_dict['l_g_gan'] = l_g_gan.item()
        # R loss
        if self.cri_pix:
            if self.train_type == 'spuf':
                self.log_dict['l_r_pix'] = l_r_pix.item()
            elif self.train_type == 'spsf':
                self.log_dict['l_r_pix'] = 0
        # D loss
        self.log_dict['l_d_real'] = l_d_real.item()
        self.log_dict['l_d_fake'] = l_d_fake.item()
        if self.opt['train']['gan_type'] == 'wgan-gp':
            self.log_dict['l_d_gp'] = l_d_gp.item()
        # D outputs
        with torch.no_grad():
            self.log_dict['D_real'] = torch.mean(real_score)
            self.log_dict['D_fake'] = torch.mean(fake_score)

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.hr_y = self.netG(self.lr_x, self.lr_p)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LRx'] = self.lr_x.detach()[0].float().cpu()
        out_dict['LRp'] = self.lr_p.detach()[0].float().cpu()
        out_dict['SR'] = self.hr_y.detach()[0].float().cpu()
        if self.is_train and self.train_type == 'spuf':
            out_dict['LRxRec'] = self.lr_x_rec.detach()[0].float().cpu()
            out_dict['LRpRec'] = self.lr_p_rec.detach()[0].float().cpu()
        if need_HR:
            if self.dataset_type == 'reduced':
                out_dict['HRx'] = self.hr_x.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(
                self.netG.__class__.__name__,
                self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(
            net_struc_str, n))
        if self.is_train:
            logger.info(s)
        if self.is_train:
            # Reconstructor
            if self.train_type == 'spuf':
                s, n = self.get_network_description(self.netR)
                if isinstance(self.netR, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(
                        self.netR.__class__.__name__,
                        self.netR.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netR.__class__.__name__)

                logger.info(
                    'Network R structure: {}, with parameters: {:,d}'.format(
                        net_struc_str, n))
                logger.info(s)
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel):
                net_struc_str = '{} - {}'.format(
                    self.netD.__class__.__name__,
                    self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)

            logger.info(
                'Network D structure: {}, with parameters: {:,d}'.format(
                    net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(
                load_path_G))
            self.load_network(load_path_G, self.netG, strict=False)
        if self.train_type == 'spuf':
            load_path_R = self.opt['path']['pretrain_model_R']
            if self.opt['is_train'] and load_path_R is not None:
                logger.info('Loading pretrained model for R [{:s}] ...'.format(
                    load_path_R))
                self.load_network(load_path_R, self.netR)
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading pretrained model for D [{:s}] ...'.format(
                load_path_D))
            self.load_network(load_path_D, self.netD)

    def save(self, iter_step, score=0, identifier='psnr'):
        self.save_network(self.netG, 'G', iter_step, score, identifier)
        if self.train_type == 'spuf':
            self.save_network(self.netR, 'R', iter_step, score, identifier)
        self.save_network(self.netD, 'D', iter_step, score, identifier)
