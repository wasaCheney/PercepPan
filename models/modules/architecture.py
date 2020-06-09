import math
import os
import sys

import torch
import torch.nn as nn

import torchvision

# local or gdrive
GDRIVE = False
DRIVE_ROOT = '/content/gdrive/My Drive' if GDRIVE else os.getenv('HOME')
sys.path.append(os.path.join(DRIVE_ROOT, 'code/BasicSR/codes/models/modules'))

import block as B
import spectral_norm as SN

####################
# Generator
####################


class SRResNet(nn.Module):
    def __init__(self,
                 in_nc,
                 out_nc,
                 nf,
                 nb,
                 upscale=4,
                 norm_type='batch',
                 act_type='relu',
                 mode='NAC',
                 res_scale=1,
                 upsample_mode='upconv'):
        super(SRResNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc,
                                nf,
                                kernel_size=3,
                                norm_type=None,
                                act_type=None)
        resnet_blocks = [
            B.ResNetBlock(nf,
                          nf,
                          nf,
                          norm_type=norm_type,
                          act_type=act_type,
                          mode=mode,
                          res_scale=res_scale) for _ in range(nb)
        ]
        LR_conv = B.conv_block(nf,
                               nf,
                               kernel_size=3,
                               norm_type=norm_type,
                               act_type=None,
                               mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError(
                'upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [
                upsample_block(nf, nf, act_type=act_type)
                for _ in range(n_upscale)
            ]
        HR_conv0 = B.conv_block(nf,
                                nf,
                                kernel_size=3,
                                norm_type=None,
                                act_type=act_type)
        HR_conv1 = B.conv_block(nf,
                                out_nc,
                                kernel_size=3,
                                norm_type=None,
                                act_type=None)

        self.model = B.sequential(
            fea_conv, B.ShortcutBlock(B.sequential(*resnet_blocks, LR_conv)),
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x


class RRDBNet(nn.Module):
    def __init__(self,
                 in_nc,
                 out_nc,
                 nf,
                 nb,
                 gc=32,
                 upscale=4,
                 norm_type=None,
                 act_type='leakyrelu',
                 mode='CNA',
                 upsample_mode='upconv'):
        super(RRDBNet, self).__init__()
        self.in_nc = in_nc
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc,
                                nf,
                                kernel_size=3,
                                norm_type=None,
                                act_type=None)
        rb_blocks = [
            B.RRDB(nf,
                   kernel_size=3,
                   gc=32,
                   stride=1,
                   bias=True,
                   pad_type='zero',
                   norm_type=norm_type,
                   act_type=act_type,
                   mode='CNA') for _ in range(nb)
        ]
        LR_conv = B.conv_block(nf,
                               nf,
                               kernel_size=3,
                               norm_type=norm_type,
                               act_type=None,
                               mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError(
                'upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [
                upsample_block(nf, nf, act_type=act_type)
                for _ in range(n_upscale)
            ]
        HR_conv0 = B.conv_block(nf,
                                nf,
                                kernel_size=3,
                                norm_type=None,
                                act_type=act_type)
        HR_conv1 = B.conv_block(nf,
                                out_nc,
                                kernel_size=3,
                                norm_type=None,
                                act_type=None)

        self.model = B.sequential(
            fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x, p=None):
        x = self.model(x)
        # p is a style-free panchromatic image
        # similar to style transfer
        if p is not None:  # normalized or not?
            mu = x[:, :self.in_nc, :, :]
            sigma = x[:, self.in_nc:, :, :]
            # (p - mu_p) / sigma_p XXX
            x = sigma * p + mu
            # range [0, 1], XXX necessary or not?
            #  x = nn.functional.sigmoid(x)
        return x


####################
# Reconstructor
####################


class REC_CONVNet(nn.Module):
    """Reconstruct LM and PAN from the generated HM,
    the net could be more complicated or
    given and fixed (LM - DownSmth; Pan - linear combin)"""

    def __init__(self,
                 lm_nc,
                 pan_nc=1,
                 downscale=4,
                 norm_type=None,
                 act_type='leakyrelu',
                 mode='CNA'):
        super().__init__()
        assert downscale in (2, 4), 'downscale should be 2 or 4'
        self.downscale = downscale
        # Rec LM
        #  conv_LM0 = B.conv_block(
        #  1,
        #  1,
        #  kernel_size=4,
        #  stride=2,
        #  norm_type=None,
        #  act_type=act_type,
        #  mode=mode) if downscale == 4 else None
        #  conv_LM1 = B.conv_block(
        #  1,
        #  1,
        #  kernel_size=4,
        #  stride=2,
        #  norm_type=None,
        #  act_type=None,
        #  mode=mode)
        conv_LM0 = B.conv_block(lm_nc,
                                lm_nc,
                                kernel_size=3,
                                stride=1,
                                groups=lm_nc,
                                bias=False,
                                norm_type=None,
                                act_type=None,
                                mode=mode)
        down_LM1 = nn.Upsample(scale_factor=1 / downscale, mode='bilinear')
        self.LM = B.sequential(conv_LM0, down_LM1)

        # Rec Pan
        conv_PAN0 = B.conv_block(lm_nc,
                                 pan_nc,
                                 kernel_size=1,
                                 stride=1,
                                 bias=False,
                                 norm_type=None,
                                 act_type=None,
                                 mode=mode)
        self.PAN = B.sequential(conv_PAN0)

    def forward(self, x):
        #  batch_size, channels, height, width = x.size()
        #  lm_rec = self.LM(x.view(-1, 1, height, width)).view(
        #  batch_size, channels, height // self.downscale,
        #  width // self.downscale)
        lm_rec = self.LM(x)
        pan_rec = self.PAN(x)
        return lm_rec, pan_rec


####################
# Discriminator
####################


# VGG style Discriminator with input size 64*64 and 256*256
class Discriminator_VGG_64_256(nn.Module):
    def __init__(self,
                 lm_nc,
                 pan_nc=1,
                 base_nf=64,
                 downscale=4,
                 norm_type='batch',
                 act_type='leakyrelu',
                 mode='CNA'):
        super().__init__()
        assert downscale in (2, 4), 'downscale should be 2 or 4'
        # LM
        # input (bs, c, 64, 64)
        # output (bs, 32, 64, 64)
        conv_LM0 = B.conv_block(lm_nc,
                                base_nf // 2,
                                kernel_size=3,
                                norm_type=None,
                                act_type=act_type,
                                mode=mode)
        conv_LM1 = B.conv_block(base_nf // 2,
                                base_nf // 2,
                                kernel_size=3,
                                stride=1,
                                norm_type=norm_type,
                                act_type=act_type,
                                mode=mode)
        self.LM = B.sequential(conv_LM0, conv_LM1)
        # PAN
        # input (bs, 1, 256, 256)
        # output (bs, 32, 64, 64)
        conv_PAN0 = B.conv_block(pan_nc,
                                 base_nf // 4,
                                 kernel_size=3,
                                 norm_type=None,
                                 act_type=act_type,
                                 mode=mode)
        # size down -> 128
        conv_PAN1 = B.conv_block(base_nf // 4,
                                 base_nf // 4,
                                 kernel_size=4,
                                 stride=2,
                                 norm_type=norm_type,
                                 act_type=act_type,
                                 mode=mode) if downscale == 4 else None
        conv_PAN2 = B.conv_block(base_nf // 4,
                                 base_nf // 2,
                                 kernel_size=3,
                                 stride=1,
                                 norm_type=norm_type,
                                 act_type=act_type,
                                 mode=mode) if downscale == 4 else None
        # size down -> 64
        conv_PAN3 = B.conv_block(base_nf // 2,
                                 base_nf // 2,
                                 kernel_size=4,
                                 stride=2,
                                 norm_type=norm_type,
                                 act_type=act_type,
                                 mode=mode)
        self.PAN = B.sequential(conv_PAN0, conv_PAN1, conv_PAN2, conv_PAN3)
        # Fus(Concatenate(LM_output, PAN_output))
        # input (bs, 64, 64, 64)
        # output (bs, 64, 64, 64)
        conv_Fus = B.conv_block(base_nf,
                                base_nf,
                                kernel_size=3,
                                norm_type=None,
                                act_type=act_type,
                                mode=mode)
        self.Fus = B.sequential(conv_Fus)
        # head
        # input (bs, 64, 64, 64)
        # output (bs, 512, 4, 4)
        convs = []
        this_nf = base_nf
        while True:
            if this_nf < 512:
                nf123 = this_nf
                nf4 = 2 * this_nf
            #  elif this_nf == 512:
            #  nf123 = 512
            #  nf4 = 512
            else:
                break
            # size down
            size_down = B.conv_block(nf123,
                                     nf123,
                                     kernel_size=4,
                                     stride=2,
                                     norm_type=norm_type,
                                     act_type=act_type,
                                     mode=mode)
            # channels up
            size_keep = B.conv_block(nf123,
                                     nf4,
                                     kernel_size=3,
                                     stride=1,
                                     norm_type=norm_type,
                                     act_type=act_type,
                                     mode=mode)
            convs.extend([size_down, size_keep])
            this_nf *= 2
        # 8, 512
        conv_last = B.conv_block(512,
                                 512,
                                 kernel_size=4,
                                 stride=2,
                                 norm_type=norm_type,
                                 act_type=act_type,
                                 mode=mode)
        # 4, 512
        convs.append(conv_last)
        self.head = B.sequential(*convs)
        # classifier
        self.classifier = nn.Sequential(nn.Linear(512 * 4 * 4, 100),
                                        nn.LeakyReLU(0.2, True),
                                        nn.Linear(100, 1))

    def forward(self, x, p, before_fus=True):
        f_x = self.LM(x)
        f_p = self.PAN(p)
        f_xp = torch.cat((f_x, f_p), dim=1)
        f_fus = self.Fus(f_xp)
        #  print(f_fus.size())
        # For feature loss
        feature = f_xp if before_fus else f_fus
        # VGG-style head
        f_head = self.head(f_fus)
        #  print(f_head.size())
        f_head = f_head.view(x.size(0), -1)
        score = self.classifier(f_head)
        return feature, score


# VGG style Discriminator with input size 256*256
class Discriminator_VGG_256(nn.Module):
    def __init__(self,
                 lm_nc=4,
                 #  pan_nc=1,
                 base_nf=64,
                 #  downscale=4,
                 norm_type='batch',
                 act_type='leakyrelu',
                 mode='CNA'):
        super().__init__()
        # features
        # h and w, c
        # input 256, 3
        conv0 = B.conv_block(lm_nc,
                             base_nf,
                             kernel_size=3,
                             norm_type=None,
                             act_type=act_type,
                             mode=mode)
        conv1 = B.conv_block(base_nf,
                             base_nf,
                             kernel_size=4,
                             stride=2,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        # 128, 64
        conv2 = B.conv_block(base_nf,
                             base_nf * 2,
                             kernel_size=3,
                             stride=1,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        conv3 = B.conv_block(base_nf * 2,
                             base_nf * 2,
                             kernel_size=4,
                             stride=2,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        # 64, 128
        conv4 = B.conv_block(base_nf * 2,
                             base_nf * 4,
                             kernel_size=3,
                             stride=1,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        conv5 = B.conv_block(base_nf * 4,
                             base_nf * 4,
                             kernel_size=4,
                             stride=2,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        self.feature_shallow = B.sequential(conv0, conv1, conv2, conv3, conv4,
                                            conv5)
        # 32, 256
        conv6 = B.conv_block(base_nf * 4,
                             base_nf * 8,
                             kernel_size=3,
                             stride=1,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        conv7 = B.conv_block(base_nf * 8,
                             base_nf * 8,
                             kernel_size=4,
                             stride=2,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        # 16, 512
        conv8 = B.conv_block(base_nf * 8,
                             base_nf * 8,
                             kernel_size=3,
                             stride=1,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        conv9 = B.conv_block(base_nf * 8,
                             base_nf * 8,
                             kernel_size=4,
                             stride=2,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        # 8, 512
        conv10 = B.conv_block(base_nf * 8,
                              base_nf * 8,
                              kernel_size=3,
                              stride=1,
                              norm_type=norm_type,
                              act_type=act_type,
                              mode=mode)
        conv11 = B.conv_block(base_nf * 8,
                              base_nf * 8,
                              kernel_size=4,
                              stride=2,
                              norm_type=norm_type,
                              act_type=act_type,
                              mode=mode)
        # 4, 512
        self.feature_deep = B.sequential(conv6, conv7, conv8, conv9, conv10,
                                         conv11)

        # classifier
        self.classifier = nn.Sequential(nn.Linear(512 * 4 * 4, 100),
                                        nn.LeakyReLU(0.2, True),
                                        nn.Linear(100, 1))

    def forward(self, x):
        feature = self.feature_shallow(x)
        x = self.feature_deep(feature)
        x = x.view(x.size(0), -1)
        score = self.classifier(x)
        return feature, score


# VGG style Discriminator with input size 128*128
class Discriminator_VGG_128(nn.Module):
    def __init__(self,
                 in_nc,
                 base_nf,
                 norm_type='batch',
                 act_type='leakyrelu',
                 mode='CNA'):
        super(Discriminator_VGG_128, self).__init__()
        # features
        # h and w, c
        # input 128, 3
        conv0 = B.conv_block(in_nc,
                             base_nf,
                             kernel_size=3,
                             norm_type=None,
                             act_type=act_type,
                             mode=mode)
        conv1 = B.conv_block(base_nf,
                             base_nf,
                             kernel_size=4,
                             stride=2,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        # 64, 64
        conv2 = B.conv_block(base_nf,
                             base_nf * 2,
                             kernel_size=3,
                             stride=1,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        conv3 = B.conv_block(base_nf * 2,
                             base_nf * 2,
                             kernel_size=4,
                             stride=2,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        # 32, 128
        conv4 = B.conv_block(base_nf * 2,
                             base_nf * 4,
                             kernel_size=3,
                             stride=1,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        conv5 = B.conv_block(base_nf * 4,
                             base_nf * 4,
                             kernel_size=4,
                             stride=2,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        # 16, 256
        conv6 = B.conv_block(base_nf * 4,
                             base_nf * 8,
                             kernel_size=3,
                             stride=1,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        conv7 = B.conv_block(base_nf * 8,
                             base_nf * 8,
                             kernel_size=4,
                             stride=2,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        # 8, 512
        conv8 = B.conv_block(base_nf * 8,
                             base_nf * 8,
                             kernel_size=3,
                             stride=1,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        conv9 = B.conv_block(base_nf * 8,
                             base_nf * 8,
                             kernel_size=4,
                             stride=2,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        # 4, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5,
                                     conv6, conv7, conv8, conv9)

        # classifier
        self.classifier = nn.Sequential(nn.Linear(512 * 4 * 4, 100),
                                        nn.LeakyReLU(0.2, True),
                                        nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# VGG style Discriminator with input size 128*128, Spectral Normalization
class Discriminator_VGG_128_SN(nn.Module):
    def __init__(self):
        super(Discriminator_VGG_128_SN, self).__init__()
        # features
        # hxw, c
        # 128, 64
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.conv0 = SN.spectral_norm(nn.Conv2d(3, 64, 3, 1, 1))
        self.conv1 = SN.spectral_norm(nn.Conv2d(64, 64, 4, 2, 1))
        # 64, 64
        self.conv2 = SN.spectral_norm(nn.Conv2d(64, 128, 3, 1, 1))
        self.conv3 = SN.spectral_norm(nn.Conv2d(128, 128, 4, 2, 1))
        # 32, 128
        self.conv4 = SN.spectral_norm(nn.Conv2d(128, 256, 3, 1, 1))
        self.conv5 = SN.spectral_norm(nn.Conv2d(256, 256, 4, 2, 1))
        # 16, 256
        self.conv6 = SN.spectral_norm(nn.Conv2d(256, 512, 3, 1, 1))
        self.conv7 = SN.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 8, 512
        self.conv8 = SN.spectral_norm(nn.Conv2d(512, 512, 3, 1, 1))
        self.conv9 = SN.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 4, 512

        # classifier
        self.linear0 = SN.spectral_norm(nn.Linear(512 * 4 * 4, 100))
        self.linear1 = SN.spectral_norm(nn.Linear(100, 1))

    def forward(self, x):
        x = self.lrelu(self.conv0(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv7(x))
        x = self.lrelu(self.conv8(x))
        x = self.lrelu(self.conv9(x))
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.linear0(x))
        x = self.linear1(x)
        return x


class Discriminator_VGG_96(nn.Module):
    def __init__(self,
                 in_nc,
                 base_nf,
                 norm_type='batch',
                 act_type='leakyrelu',
                 mode='CNA'):
        super(Discriminator_VGG_96, self).__init__()
        # features
        # hxw, c
        # 96, 64
        conv0 = B.conv_block(in_nc,
                             base_nf,
                             kernel_size=3,
                             norm_type=None,
                             act_type=act_type,
                             mode=mode)
        conv1 = B.conv_block(base_nf,
                             base_nf,
                             kernel_size=4,
                             stride=2,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        # 48, 64
        conv2 = B.conv_block(base_nf,
                             base_nf * 2,
                             kernel_size=3,
                             stride=1,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        conv3 = B.conv_block(base_nf * 2,
                             base_nf * 2,
                             kernel_size=4,
                             stride=2,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        # 24, 128
        conv4 = B.conv_block(base_nf * 2,
                             base_nf * 4,
                             kernel_size=3,
                             stride=1,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        conv5 = B.conv_block(base_nf * 4,
                             base_nf * 4,
                             kernel_size=4,
                             stride=2,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        # 12, 256
        conv6 = B.conv_block(base_nf * 4,
                             base_nf * 8,
                             kernel_size=3,
                             stride=1,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        conv7 = B.conv_block(base_nf * 8,
                             base_nf * 8,
                             kernel_size=4,
                             stride=2,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        # 6, 512
        conv8 = B.conv_block(base_nf * 8,
                             base_nf * 8,
                             kernel_size=3,
                             stride=1,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        conv9 = B.conv_block(base_nf * 8,
                             base_nf * 8,
                             kernel_size=4,
                             stride=2,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        # 3, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5,
                                     conv6, conv7, conv8, conv9)

        # classifier
        self.classifier = nn.Sequential(nn.Linear(512 * 3 * 3, 100),
                                        nn.LeakyReLU(0.2, True),
                                        nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Discriminator_VGG_192(nn.Module):
    def __init__(self,
                 in_nc,
                 base_nf,
                 norm_type='batch',
                 act_type='leakyrelu',
                 mode='CNA'):
        super(Discriminator_VGG_192, self).__init__()
        # features
        # hxw, c
        # 192, 64
        conv0 = B.conv_block(in_nc,
                             base_nf,
                             kernel_size=3,
                             norm_type=None,
                             act_type=act_type,
                             mode=mode)
        conv1 = B.conv_block(base_nf,
                             base_nf,
                             kernel_size=4,
                             stride=2,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        # 96, 64
        conv2 = B.conv_block(base_nf,
                             base_nf * 2,
                             kernel_size=3,
                             stride=1,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        conv3 = B.conv_block(base_nf * 2,
                             base_nf * 2,
                             kernel_size=4,
                             stride=2,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        # 48, 128
        conv4 = B.conv_block(base_nf * 2,
                             base_nf * 4,
                             kernel_size=3,
                             stride=1,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        conv5 = B.conv_block(base_nf * 4,
                             base_nf * 4,
                             kernel_size=4,
                             stride=2,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        # 24, 256
        conv6 = B.conv_block(base_nf * 4,
                             base_nf * 8,
                             kernel_size=3,
                             stride=1,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        conv7 = B.conv_block(base_nf * 8,
                             base_nf * 8,
                             kernel_size=4,
                             stride=2,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        # 12, 512
        conv8 = B.conv_block(base_nf * 8,
                             base_nf * 8,
                             kernel_size=3,
                             stride=1,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        conv9 = B.conv_block(base_nf * 8,
                             base_nf * 8,
                             kernel_size=4,
                             stride=2,
                             norm_type=norm_type,
                             act_type=act_type,
                             mode=mode)
        # 6, 512
        conv10 = B.conv_block(base_nf * 8,
                              base_nf * 8,
                              kernel_size=3,
                              stride=1,
                              norm_type=norm_type,
                              act_type=act_type,
                              mode=mode)
        conv11 = B.conv_block(base_nf * 8,
                              base_nf * 8,
                              kernel_size=4,
                              stride=2,
                              norm_type=norm_type,
                              act_type=act_type,
                              mode=mode)
        # 3, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5,
                                     conv6, conv7, conv8, conv9, conv10,
                                     conv11)

        # classifier
        self.classifier = nn.Sequential(nn.Linear(512 * 3 * 3, 100),
                                        nn.LeakyReLU(0.2, True),
                                        nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


####################
# Perceptual Network
####################


# Assume input range is [0, 1]
class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        # pretrained=True will download the pretrained model on ImageNet
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1,
                                                            1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1,
                                                           1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(
            feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


# Assume input range is [0, 1]
class ResNet101FeatureExtractor(nn.Module):
    def __init__(self, use_input_norm=True, device=torch.device('cpu')):
        super(ResNet101FeatureExtractor, self).__init__()
        model = torchvision.models.resnet101(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1,
                                                            1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1,
                                                           1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.children())[:8])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


class MINCNet(nn.Module):
    def __init__(self):
        super(MINCNet, self).__init__()
        self.ReLU = nn.ReLU(True)
        self.conv11 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv21 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv22 = nn.Conv2d(128, 128, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv31 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv32 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv33 = nn.Conv2d(256, 256, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv41 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv42 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv43 = nn.Conv2d(512, 512, 3, 1, 1)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv51 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv52 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv53 = nn.Conv2d(512, 512, 3, 1, 1)

    def forward(self, x):
        out = self.ReLU(self.conv11(x))
        out = self.ReLU(self.conv12(out))
        out = self.maxpool1(out)
        out = self.ReLU(self.conv21(out))
        out = self.ReLU(self.conv22(out))
        out = self.maxpool2(out)
        out = self.ReLU(self.conv31(out))
        out = self.ReLU(self.conv32(out))
        out = self.ReLU(self.conv33(out))
        out = self.maxpool3(out)
        out = self.ReLU(self.conv41(out))
        out = self.ReLU(self.conv42(out))
        out = self.ReLU(self.conv43(out))
        out = self.maxpool4(out)
        out = self.ReLU(self.conv51(out))
        out = self.ReLU(self.conv52(out))
        out = self.conv53(out)
        return out


# Assume input range is [0, 1]
class MINCFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cpu')):
        super(MINCFeatureExtractor, self).__init__()

        self.features = MINCNet()
        self.features.load_state_dict(
            torch.load('../experiments/pretrained_models/VGG16minc_53.pth'),
            strict=True)
        self.features.eval()
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        output = self.features(x)
        return output


if __name__ == '__main__':
    test_model = REC_CONVNet(4)
    x = torch.rand([16, 4, 256, 256])
    lm_rec, pan_rec = test_model(x)
    print(test_model)
    print('LM Rec shape:', lm_rec.shape, 'PAN Rec shape:', pan_rec.shape)
    #  for name, para in test_model.named_parameters():
    #  print(name)
    #  print(para)
    print(list(test_model.parameters())[0].data.shape)
    print(list(test_model.parameters())[1].data.shape)
    #  test_model = Discriminator_VGG_64_256(4)
    #  x = torch.rand([16, 4, 64, 64])
    #  p = torch.rand([16, 1, 256, 256])
    #  feat, score = test_model(x, p, True)
    #  print(test_model)
    #  print(feat.size())
    #  print(score)
