import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import math
import random
import torch.nn.functional as F
from typing import Optional, Callable
import math


def AdaIN(tensor1, tensor2):
    means1 = tensor1.mean(dim=[0, 2, 3], keepdim=True)  # shape: (1, channels, 1, 1)
    stds1 = tensor1.std(dim=[0, 2, 3], keepdim=True)    # shape: (1, channels, 1, 1)

    means2 = tensor2.mean(dim=[0, 2, 3], keepdim=True)  # shape: (1, channels, 1, 1)
    stds2 = tensor2.std(dim=[0, 2, 3], keepdim=True)    # shape: (1, channels, 1, 1)
    
    # 标准化
    tensor1 = ((tensor1 - means1) / stds1) * stds2 + means2
    return tensor1


def AdaIN_onnly_mean(tensor1, tensor2):
    means1 = tensor1.mean(dim=[0, 2, 3], keepdim=True)  # shape: (1, channels, 1, 1)
    means2 = tensor2.mean(dim=[0, 2, 3], keepdim=True)  # shape: (1, channels, 1, 1)
    tensor1 = tensor1 - means1 + means2
    return tensor1
    
    
def fuse_attention_maps(attn_list):
    # attn_list: list of attention tensors of shape [1, 77, 32, 32]

    attn_cat = torch.cat(attn_list)           # 70, 77, 32, 32
    attn_fuse = torch.mean(attn_cat, dim=[0])  # 77, 32, 32

    return attn_fuse


def get_words_idx_in_a_text(text):
    if not text or not text.strip():
        return 0

    text = text.strip()

    import re
    tokens = re.findall(r'[\w\']+|[^\w\s]', text)

    token_length = len(tokens)
    return np.array(range(token_length)) + 1
    

def attn_normalize(attn_map1, attn_map2):
    attn_map1 = AdaIN_onnly_mean(attn_map1, attn_map2)
    attn_weight1 = attn_map1 / (attn_map1 + attn_map2)
    attn_weight2 = 1 - attn_weight1
    return attn_weight1, attn_weight2


def dct(x, norm=None):
    '''
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    '''
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))
    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2
    V = 2 * V.view(*x_shape)
    return V


def idct(X, norm=None):
    '''
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    '''
    x_shape = X.shape
    N = x_shape[-1]
    X_v = X.contiguous().view(-1, x_shape[-1]) / 2
    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2
    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)
    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r
    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]
    return x.view(*x_shape)


def low_pass_DCT_filtering(latents, lp_percentile):
    b, c, h, w = latents.shape
    threshold_w = int(w * lp_percentile / 100)
    threshold_h = int(h * lp_percentile / 100)
    dct_w = dct(latents, norm='ortho')
    mask = torch.range(1, w).cuda()
    mask = torch.where(mask <= threshold_w, torch.ones_like(mask), torch.zeros_like(mask))
    mask = torch.reshape(mask, shape=(1, w)).repeat((h, 1))
    latents = idct(dct_w * mask, norm='ortho')

    latents_trans = latents.permute((0, 1, 3, 2))
    dct_h = dct(latents_trans, norm='ortho')
    mask = torch.range(1, h).cuda()
    mask = torch.where(mask <= threshold_h, torch.ones_like(mask), torch.zeros_like(mask))
    mask = torch.reshape(mask, shape=(1, h)).repeat((w, 1)).cuda()
    latents = idct(dct_h * mask, norm='ortho')
    latents = latents.permute(0, 1, 3, 2)
    return latents


def brightness_contrast_adjust(image, brightness=0, contrast=0):
    """
    parameters:
    - image: input numpy image
    - brightness: (-100, 100)
    - contrast: (-100, 100)
    return:
    - adjusted image
    """
    
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = image.copy().astype(np.float32)
        if img_float.max() > 1.0:
            img_float = img_float / 255.0
    
    brightness_val = brightness / 100.0
    contrast_val = contrast / 100.0
    
    # 公式: result = (img - 0.5) * (1 + contrast) + 0.5 + brightness
    result = (img_float - 0.5) * (1 + contrast_val) + 0.5 + brightness_val
    
    result = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)



