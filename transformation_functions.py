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


def vertical_flip_transformation():
    
    def horizontal_flip(tensor):
        # tensor: b, c, h, w
        return torch.flip(tensor, dims=[2])
    
    def horizontal_flip_inverse(tensor):
        # tensor: b, c, h, w
        return torch.flip(tensor, dims=[2])
        
    return horizontal_flip, horizontal_flip_inverse



def rotate_90_transformation():

    def rotate_90_clockwise(tensor):
        return torch.rot90(tensor, k=-1, dims=(-2, -1))

    def rotate_90_counterclockwise(tensor):
        return torch.rot90(tensor, k=1, dims=(-2, -1))

    return rotate_90_clockwise, rotate_90_counterclockwise


def rotate_180_transformation():

    def rotate_180_clockwise(tensor):
        return torch.rot90(tensor, k=-2, dims=(-2, -1))

    def rotate_180_counterclockwise(tensor):
        return torch.rot90(tensor, k=2, dims=(-2, -1))

    return rotate_180_clockwise, rotate_180_counterclockwise


def diagonal_flip():

    def forward(tensor):
         return tensor.transpose(-2, -1)

    def backward(tensor):
        return tensor.transpose(-2, -1)

    return forward, backward


def jigsaw_transformation_circle():

    def forward(tensor):
        # tensor: b, c, h, w
        b, c, h, w = tensor.shape
        h_half = h // 2
        w_half = w // 2
        block_1 = tensor[:, :, 0:h_half, 0:w_half]
        block_2 = tensor[:, :, 0:h_half, w_half:w]
        block_3 = tensor[:, :, h_half:h, w_half:w]
        block_4 = tensor[:, :, h_half:h, 0:w_half]
        tensor[:, :, 0:h_half, 0:w_half] = block_4
        tensor[:, :, 0:h_half, w_half:w] = block_1
        tensor[:, :, h_half:h, w_half:w] = block_2
        tensor[:, :, h_half:h, 0:w_half] = block_3
        return tensor

    def backward(tensor):
        # tensor: b, c, h, w
        b, c, h, w = tensor.shape
        h_half = h // 2
        w_half = w // 2
        block_1 = tensor[:, :, 0:h_half, 0:w_half]
        block_2 = tensor[:, :, 0:h_half, w_half:w]
        block_3 = tensor[:, :, h_half:h, w_half:w]
        block_4 = tensor[:, :, h_half:h, 0:w_half]
        tensor[:, :, 0:h_half, 0:w_half] = block_2
        tensor[:, :, 0:h_half, w_half:w] = block_3
        tensor[:, :, h_half:h, w_half:w] = block_4
        tensor[:, :, h_half:h, 0:w_half] = block_1
        return tensor

    return forward, backward
