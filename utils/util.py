import logging
import math
import os
import random
from datetime import datetime

from scipy import ndimage
import numpy as np
import cv2

import torch

from torchvision.utils import make_grid

###########
# visdom
###########


def create_vis_plot(vis, xlabel, ylabel, title, legend):
    num_lines = len(legend)
    #  print(num_lines)
    win = vis.line(X=torch.zeros((1, )).cpu(),
                   Y=torch.zeros((1, num_lines)).cpu(),
                   opts=dict(xlabel=xlabel,
                             ylabel=ylabel,
                             title=title,
                             legend=legend))
    #  print(win)
    return win


def update_vis(vis, window, xaxis, *args):
    yaxis = torch.Tensor([args]).cpu()
    vis.line(X=torch.Tensor([xaxis]).cpu(),
             Y=yaxis,
             win=window,
             update='append')
    return


####################
# miscellaneous
####################


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info(
            'Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    logs = logging.getLogger(logger_name)
    if logs.hasHandlers():
        logs.handlers.clear()
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
        datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    logs.setLevel(level)
    logs.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logs.addHandler(sh)


####################
# image convert
####################


def tensor2img(tensor, dynamic_range=255, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(8/4/3/1),H,W), 3D(C,H,W), or 2D(H,W), any range
    Output: 3D(H,W,C) or 2D(H,W), [0,255], (uint8 as default)
    '''
    assert dynamic_range in (255, 2047), 'Only 255 and 2047 are accepted!'
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)),
                           normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Epxect 4D, 3D or 2D tensor, but received {:d}D'.format(n_dim))
    img_np = (img_np * dynamic_range).round()
    if dynamic_range == 255:
        return img_np.astype(np.uint8)
    elif dynamic_range == 2047:
        return img_np.astype(np.uint16)


def save_img(img, img_path):
    """img write as raw numpy.array"""
    np.save(img_path, img)
    #  cv2.imwrite(img_path, img)


####################
# observation model
####################


def gaussian2d(N, std):
    t = np.arange(-(N - 1) // 2, (N + 2) // 2)
    t1, t2 = np.meshgrid(t, t)
    std = np.double(std)
    w = np.exp(-0.5 * (t1 / std)**2) * np.exp(-0.5 * (t2 / std)**2)
    return w


def kaiser2d(N, beta):
    t = np.arange(-(N - 1) // 2, (N + 2) // 2) / np.double(N - 1)
    t1, t2 = np.meshgrid(t, t)
    t12 = np.sqrt(t1 * t1 + t2 * t2)
    w1 = np.kaiser(N, beta)
    w = np.interp(t12, t, w1)
    w[t12 > t[-1]] = 0
    w[t12 < t[0]] = 0
    return w


def fir_filter_wind(Hd, w):
    """
    compute fir (finite impulse response) filter with window method
    Hd: 	desired freqeuncy response (2D)
    w: 		window (2D)
    """
    hd = np.rot90(np.fft.fftshift(np.rot90(Hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = h * w
    h = h / np.sum(h)
    return h


def GNyq2win(GNyq, downscale=4, N=41):
    """Generate a 2D convolutional window from a given GNyq
    GNyq: Nyquist frequency
    downscale: spatial size of PAN / spatial size of MS
    """
    # fir filter with window method
    fcut = 1 / downscale
    alpha = np.sqrt(((N - 1) * (fcut / 2))**2 / (-2 * np.log(GNyq)))
    H = gaussian2d(N, alpha)
    Hd = H / np.max(H)
    w = kaiser2d(N, 0.5)
    h = fir_filter_wind(Hd, w)
    return np.real(h)


def img_resize(img, satellite='QuickBird', downscale=4):
    # satellite GNyq
    downscale = int(downscale)
    if satellite == 'QuickBird':
        GNyq = [0.34, 0.32, 0.30, 0.22]  # Band Order: B,G,R,NIR
        GNyqPan = 0.15
    elif satellite == 'IKONOS':
        GNyq = [0.26, 0.28, 0.29, 0.28]  # Band Order: B,G,R,NIR
        GNyqPan = 0.17
    else:
        raise NotImplementedError('satellite: QuickBird or IKONOS')
    # lowpass
    img_ = img.squeeze()
    img_ = img_.astype(np.float64)
    if img_.ndim == 2:
        H, W = img_.shape
        lowpass = GNyq2win(GNyqPan, downscale, N=41)
        # img_ = ndimage.filters.correlate(img_, lowpass, mode='nearest')
        # img_ = cv2.resize(img_, 1/downscale, 'nearest')
    elif img_.ndim == 3:
        H, W, _ = img.shape
        lowpass = [GNyq2win(gnyq, downscale, N=41) for gnyq in GNyq]
        lowpass = np.stack(lowpass, axis=-1)
    img_ = ndimage.filters.correlate(img_, lowpass, mode='nearest')
    # downsampling
    output_size = (H // downscale, W // downscale)
    img_ = cv2.resize(img_, dsize=output_size, interpolation=cv2.INTER_NEAREST)
    return img_


####################
# metric
####################

# full reference


def calculate_sam(img1, img2):
    """SAM for 3D image, shape (H, W, C); uint or float[0, 1]"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    assert img1.ndim == 3 and img1.shape[
        2] > 1, "image n_channels should be greater than 1"
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    inner_product = (img1_ * img2_).sum(axis=2)
    img1_spectral_norm = np.sqrt((img1_**2).sum(axis=2))
    img2_spectral_norm = np.sqrt((img2_**2).sum(axis=2))
    # numerical stability
    cos_theta = (inner_product / (img1_spectral_norm * img2_spectral_norm +
                                  np.finfo(np.float64).eps)).clip(min=0, max=1)
    return np.mean(np.arccos(cos_theta))


def calculate_psnr(img1, img2, dynamic_range=255):
    """PSNR, img uint8 if 255; uint11 if 2047"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse < 1e-6:
        return np.inf
    return 20 * np.log10(dynamic_range /
                         (np.sqrt(mse) + np.finfo(np.float64).eps))


def calculate_scc(img1, img2):
    """SCC for 2D (H, W)or 3D (H, W, C) image; uint or float[0, 1].
    The value is in [-1., 1.] due to the appearance of covariance."""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    if img1_.ndim == 2:
        return np.corrcoef(img1_.reshape(1, -1), img2_.rehshape(1, -1))[0, 1]
    elif img1_.ndim == 3:
        ccs = [
            np.corrcoef(img1_[..., i].reshape(1, -1),
                        img2_[..., i].reshape(1, -1))[0, 1]
            for i in range(img1_.shape[2])
        ]
        return np.mean(ccs)
    else:
        raise ValueError('Wrong input image dimensions.')


def qindex(img1, img2, block_size=8):
    """Q-index for 2D (one-band) image, shape (H, W); uint or float [0, 1]
    This value is in [-1, 1] due to the appearance of covariance."""
    assert block_size > 1, 'block_size shold be greater than 1!'
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    window = np.ones((block_size, block_size)) / (block_size**2)
    # window_size = block_size**2
    # filter, valid
    pad_topleft = int(np.floor(block_size / 2))
    pad_bottomright = block_size - 1 - pad_topleft
    mu1 = cv2.filter2D(
        img1_, -1,
        window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu2 = cv2.filter2D(
        img2_, -1,
        window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(
        img1_**2, -1, window
    )[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu1_sq
    sigma2_sq = cv2.filter2D(
        img2_**2, -1, window
    )[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu2_sq
    sigma12 = cv2.filter2D(
        img1_ * img2_, -1, window
    )[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu1_mu2

    # all = 1, include the case of simga == mu == 0
    qindex_map = np.ones(sigma12.shape)
    # sigma == 0 and mu != 0
    idx = ((sigma1_sq + sigma2_sq) <= 1e-6) * ((mu1_sq + mu2_sq) > 1e-6)
    qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
    # sigma !=0 and mu == 0
    idx = ((sigma1_sq + sigma2_sq) > 1e-6) * ((mu1_sq + mu2_sq) <= 1e-6)
    qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
    # sigma != 0 and mu != 0
    idx = ((sigma1_sq + sigma2_sq) > 1e-6) * ((mu1_sq + mu2_sq) > 1e-6)
    qindex_map[idx] = ((2 * mu1_mu2[idx]) *
                       (2 * sigma12[idx])) / ((mu1_sq + mu2_sq)[idx] *
                                              (sigma1_sq + sigma2_sq)[idx])
    return np.mean(qindex_map)


def calculate_qindex(img1, img2, block_size=8):
    """Q-index for 2D (H, W) or 3D (H, W, C) image; uint or float [0, 1]"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return qindex(img1, img2, block_size)
    elif img1.ndim == 3:
        qindexs = [
            qindex(img1[..., i], img2[..., i], block_size)
            for i in range(img1.shape[2])
        ]
        return np.array(qindexs).mean()
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2, dynamic_range=255):
    """SSIM for 2D (one-band) image, shape (H, W);
    uint8 if 225; uint16 if 2047
    The value could small than zero"""
    C1 = (0.01 * dynamic_range)**2
    C2 = (0.03 * dynamic_range)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, dynamic_range=255):
    '''SSIM for 2D (H, W) or 3D (H, W, C) image;
    uint8 if 225; uint16 if 2047'''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2, dynamic_range)
    elif img1.ndim == 3:
        ssims = [
            ssim(img1[..., i], img2[..., i], dynamic_range)
            for i in range(img1.shape[2])
        ]
        return np.array(ssims).mean()
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_ergas(img_fake, img_real, scale=4):
    """ERGAS for 2D (H, W) or 3D (H, W, C) image; uint or float [0, 1].
    scale = spatial resolution ratio of PAN and MUL, default 4."""
    if not img_fake.shape == img_real.shape:
        raise ValueError('Input images must have the same dimensions.')
    img_fake_ = img_fake.astype(np.float64)
    img_real_ = img_real.astype(np.float64)
    if img_fake_.ndim == 2:
        mean_real = img_real_.mean()
        mse = np.mean((img_fake_ - img_real_)**2)
        return 100 / scale * np.sqrt(mse /
                                     (mean_real**2 + np.finfo(np.float64).eps))
    elif img_fake_.ndim == 3:
        means_real = img_real_.reshape(-1, img_real_.shape[2]).mean(axis=0)
        mses = ((img_fake_ - img_real_)**2).reshape(
            -1, img_fake_.shape[2]).mean(axis=0)
        return 100 / scale * np.sqrt(
            (mses / (means_real**2 + np.finfo(np.float64).eps)).mean())
    else:
        raise ValueError('Wrong input image dimensions.')


# No reference

##################
# No reference IQA
##################


def calculate_D_lambda(img_fake, img_lm, scale=4, block_size=32, p=1):
    """Spectral distortion, if not clipped, it could be very large.
    img_fake, generated HRMS
    img_lm, LRMS"""
    assert img_fake.ndim == img_lm.ndim == 3, 'Images must be 3D!'
    H_f, W_f, C_f = img_fake.shape
    H_r, W_r, C_r = img_lm.shape
    assert C_f == C_r, 'Fake and lm should have the same number of bands!'
    # D_lambda
    Q_fake = []
    Q_lm = []
    for i in range(C_f):
        for j in range(i + 1, C_f):
            # for fake
            band1 = img_fake[scale:-scale, scale:-scale, i]
            band2 = img_fake[scale:-scale, scale:-scale, j]
            Q_fake.append(qindex(band1, band2, block_size=block_size))
            # for real
            band1 = img_lm[scale:-scale, scale:-scale, i]
            band2 = img_lm[scale:-scale, scale:-scale, j]
            Q_lm.append(qindex(band1, band2, block_size=block_size))
    Q_fake = np.array(Q_fake)
    Q_lm = np.array(Q_lm)
    D_lambda_index = (np.abs(Q_fake - Q_lm)**p).mean()
    return np.clip(D_lambda_index, 0, 1)**(1 / p)


def calculate_D_s(img_fake,
                  img_lm,
                  pan_hp,
                  satellite='QuickBird',
                  scale=4,
                  block_size=32,
                  q=1):
    """Spatial distortion, if not clipped, it could be very large.
    img_fake, generated HRMS
    img_lm, LRMS
    pan_hp, HRPan"""
    # fake and lm
    if img_fake.ndim != 3 or img_lm.ndim != 3:
        raise ValueError('Real or fake MS images must be 3D!')
    H_f, W_f, C_f = img_fake.shape
    H_r, W_r, C_r = img_lm.shape
    if H_f // H_r != scale or W_f // W_r != scale:
        raise ValueError('Spatial resolution should be compatible with scale')
    if C_f != C_r:
        raise ValueError('Fake and lm should have the same number of bands!')
    # fake and pan
    if pan_hp.ndim == 2:
        pan = np.expand_dims(pan_hp, axis=-1)
    elif pan_hp.ndim == 3:
        pan = pan_hp
    else:
        raise ValueError('Panchromatic image must be 2D or 3D!')
    H_p, W_p, C_p = pan.shape
    if C_p != 1:
        raise ValueError('size of 3rd dim of Panchromatic image must be 1')
    if H_f != H_p or W_f != W_p:
        raise ValueError(
            "Pan's and fake's spatial resolution should be the same")
    # get LRPan, 2D
    pan_lr = img_resize(pan, satellite=satellite, downscale=scale)
    # D_s
    Q_hr = []
    Q_lr = []
    for i in range(C_f):
        # for HR fake
        band1 = img_fake[scale:-scale, scale:-scale, i]
        # the input PAN is 3D with size=1 along 3rd dim
        band2 = pan[scale:-scale, scale:-scale, 0]
        Q_hr.append(qindex(band1, band2, block_size=block_size))
        band1 = img_lm[scale:-scale, scale:-scale, i]
        band2 = pan_lr[scale:-scale, scale:-scale]  # this is 2D
        Q_lr.append(qindex(band1, band2, block_size=block_size))
    Q_hr = np.array(Q_hr)
    Q_lr = np.array(Q_lr)
    D_s_index = (np.abs(Q_hr - Q_lr)**q).mean()
    return np.clip(D_s_index, 0, 1)**(1 / q)


def calculate_qnr(img_fake,
                  img_lm,
                  pan,
                  satellite='QuickBird',
                  scale=4,
                  block_size=32,
                  p=1,
                  q=1,
                  alpha=1,
                  beta=1):
    """QNR - No reference IQA"""
    D_lambda_idx = calculate_D_lambda(img_fake, img_lm, scale, block_size, p)
    D_s_idx = calculate_D_s(img_fake, img_lm, pan, satellite, scale,
                            block_size, q)
    QNR_idx = (1 - D_lambda_idx)**alpha * (1 - D_s_idx)**beta
    return QNR_idx, D_lambda_idx, D_s_idx


if __name__ == '__main__':
    img = np.random.random((32, 32, 4))
    img_ = img_resize(img, satellite='QuickBird', downscale=4)
    print(img_.shape)
