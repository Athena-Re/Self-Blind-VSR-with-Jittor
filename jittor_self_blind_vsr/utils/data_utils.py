import random
import numpy as np
import skimage.color as sc
import jittor as jt
import cv2
import math


def get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False):
    ih, iw = args[0].shape[:2]

    if not input_large:
        p = scale if multi else 1
        tp = p * patch_size
        ip = tp // scale
    else:
        tp = patch_size
        ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy

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
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]


def np2Tensor(*args, rgb_range=255, n_colors=3):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = jt.array(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]


def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(a) for a in args]


def data_augment(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))
    else:
        mode = random.randint(0, 7)
        return data_augment(img, mode)


def calc_psnr(sr, hr, rgb_range=255, is_rgb=True):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range

    if is_rgb:
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        diff = diff.mul(convert).sum(dim=1)

    mse = diff.pow(2).mean()
    return -10 * math.log10(mse)


def matlab_imresize(img, scalar_scale=None, method='bicubic', output_shape=None, antialiasing=True):
    """Imitate the behavior of MATLAB's imresize function"""
    if output_shape is not None:
        output_shape = tuple(output_shape)
    elif scalar_scale is not None:
        if isinstance(scalar_scale, (int, float)):
            scalar_scale = (scalar_scale, scalar_scale)
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


def postprocess(*images, rgb_range=255, ycbcr_flag=False, device='cpu'):
    def _postprocess(img):
        if isinstance(img, jt.Var):
            pixel_range = rgb_range / 255
            img = img.mul(pixel_range).clamp(0, rgb_range).round()
            img = img[0].detach().cpu().numpy()
            img = np.transpose(img, (1, 2, 0)).astype(np.uint8)
        return img

    return [_postprocess(img) for img in images] 