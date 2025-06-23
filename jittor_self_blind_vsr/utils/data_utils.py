import random
import jittor as jt
import jittor.nn as nn
import numpy as np
import math
from jittor import Var
import skimage.color as sc
import cv2


def get_patch(*args, patch_size=17, scale=1):
    """
    Get patch from an image
    """
    ih, iw, _ = args[0].shape

    ip = patch_size
    tp = scale * ip

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret


def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = sc.rgb2ycbcr(img)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]


def np2Tensor(*args, rgb_range=255, n_colors=1):
    def _np2Tensor(img):
        img = img.astype('float64')
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # NHWC -> NCHW
        tensor = jt.array(np_transpose).float()  # numpy -> tensor
        tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)

        return tensor

    return [_np2Tensor(a) for a in args]


def data_augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = np.rot90(img)

        return img

    return [_augment(a) for a in args]


def matlab_imresize(img, scalar_scale=None, output_shape=None, method='bicubic'):
    '''same as matlab2017 imresize
    img: shape=[h, w, c]
    scalar_scale: the resize scale
        if None, using output_shape
    output_shape: the resize shape, (h, w)
        if scalar_scale=None, using this param
    method: the interpolation method
        optional: 'bicubic', 'bilinear'
        default: 'bicubic'
    '''
    if output_shape is not None:
        output_shape = tuple(output_shape)
    elif scalar_scale is not None:
        if isinstance(scalar_scale, (int, float)):
            output_shape = tuple(int(dim * scalar_scale) for dim in img.shape[:2])
        else:
            output_shape = tuple(int(dim * scale) for dim, scale in zip(img.shape[:2], scalar_scale))
    else:
        raise ValueError("Either scalar_scale or output_shape must be provided")

    # Convert method name to OpenCV interpolation flag
    if method == 'bicubic':
        interpolation = cv2.INTER_CUBIC
    elif method == 'bilinear':
        interpolation = cv2.INTER_LINEAR
    elif method == 'nearest':
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_CUBIC

    # Resize using OpenCV
    resized = cv2.resize(img, (output_shape[1], output_shape[0]), interpolation=interpolation)
    
    # Ensure the output has the same number of dimensions as the input
    if img.ndim == 3 and resized.ndim == 2:
        resized = np.expand_dims(resized, axis=2)
    
    return resized


def postprocess(*images, rgb_range, ycbcr_flag):
    def _postprocess(img, rgb_coefficient, ycbcr_flag):
        if ycbcr_flag:
            out = img.mul(rgb_coefficient).clamp(16, 235)
        else:
            out = img.mul(rgb_coefficient).clamp(0, 255).round()

        return out

    rgb_coefficient = 255 / rgb_range
    return [_postprocess(img, rgb_coefficient, ycbcr_flag) for img in images]


def calc_psnr(img1, img2, rgb_range=1., shave=4, is_rgb=False):
    if isinstance(img1, jt.Var):
        img1 = img1[:, :, shave:-shave, shave:-shave]
        img1 = img1.detach().numpy()
    if isinstance(img2, jt.Var):
        img2 = img2[:, :, shave:-shave, shave:-shave]
        img2 = img2.detach().numpy()
    mse = np.mean((img1 / rgb_range - img2 / rgb_range) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def calc_grad_sobel(img):
    if not isinstance(img, jt.Var):
        raise Exception("Now just support jt.Var. See the Type(img)={}".format(type(img)))
    if not img.ndimension() == 4:
        raise Exception("Tensor ndimension must equal to 4. See the img.ndimension={}".format(img.ndimension()))

    img = jt.mean(img, dim=1, keepdim=True)

    sobel_filter_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).reshape((1, 1, 3, 3))
    sobel_filter_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).reshape((1, 1, 3, 3))
    sobel_filter_X = jt.array(sobel_filter_X).float()
    sobel_filter_Y = jt.array(sobel_filter_Y).float()
    grad_X = nn.conv2d(img, sobel_filter_X, bias=None, stride=1, padding=1)
    grad_Y = nn.conv2d(img, sobel_filter_Y, bias=None, stride=1, padding=1)
    grad = jt.sqrt(grad_X.pow(2) + grad_Y.pow(2))

    return grad_X, grad_Y, grad


def calc_meanFilter(img, kernel_size=11, n_channel=1):
    mean_filter_X = np.ones(shape=(1, 1, kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    mean_filter_X = jt.array(mean_filter_X).float()
    new_img = jt.zeros_like(img)
    for i in range(n_channel):
        new_img[:, i:i + 1, :, :] = nn.conv2d(img[:, i:i + 1, :, :], mean_filter_X, bias=None,
                                              stride=1, padding=kernel_size // 2)
    return new_img


def warp_by_flow(x, flo):
    B, C, H, W = flo.size()

    # mesh grid
    xx = jt.arange(0, W).view(1, -1).repeat(H, 1)
    yy = jt.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = jt.concat((xx, yy), 1).float()
    vgrid = Var(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.grid_sample(x, vgrid, padding_mode='border')

    return output


#################################################################################
####                        Tensor RGB PSNR SSIM                             ####
#################################################################################


def PSNR_Tensor_RGB(img1, img2, rgb_range=1., shave=4):
    if isinstance(img1, jt.Var):
        img1 = img1[:, :, shave:-shave, shave:-shave]
        img1 = img1.detach().numpy()
    if isinstance(img2, jt.Var):
        img2 = img2[:, :, shave:-shave, shave:-shave]
        img2 = img2.detach().numpy()
    mse = np.mean((img1 / rgb_range - img2 / rgb_range) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def PSNR_EDVR(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def SSIM_EDVR(img1, img2):
    def ssim(img1, img2):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

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

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                    (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.') 