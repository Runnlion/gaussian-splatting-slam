#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    """
    L1损失函数：计算网络输出和真实值之间的绝对误差均值。
    L1 Loss Function: Computes the mean absolute error between network output and ground truth.
    
    Args:
        network_output (Tensor): 网络的输出 (Network output)
        gt (Tensor): 真实值 (Ground truth)

    Returns:
        Tensor: L1损失 (L1 loss)
    """
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    """
    L2损失函数：计算网络输出和真实值之间的平方误差均值。
    L2 Loss Function: Computes the mean squared error between network output and ground truth.
    
    Args:
        network_output (Tensor): 网络的输出 (Network output)
        gt (Tensor): 真实值 (Ground truth)

    Returns:
        Tensor: L2损失 (L2 loss)
    """
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    """
    生成高斯分布窗口：用于计算图像的相似性。
    Generates a Gaussian distribution window for image similarity calculations.
    
    Args:
        window_size (int): 窗口大小 (Size of the window)
        sigma (float): 高斯分布的标准差 (Standard deviation of the Gaussian distribution)

    Returns:
        Tensor: 高斯窗口 (Gaussian window)
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """
    创建2D高斯窗口：扩展为特定通道数。
    Creates a 2D Gaussian window expanded to the specified number of channels.
    
    Args:
        window_size (int): 窗口大小 (Size of the window)
        channel (int): 图像通道数量 (Number of image channels)

    Returns:
        Variable: 包含高斯分布的2D窗口 (2D window with Gaussian distribution)
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    """
    计算结构相似性指数（SSIM）：用于评估两张图像的相似性。
    Computes Structural Similarity Index (SSIM) to evaluate the similarity between two images.
    
    Args:
        img1 (Tensor): 第一张输入图像 (First input image)
        img2 (Tensor): 第二张输入图像 (Second input image)
        window_size (int): SSIM窗口大小 (SSIM window size)
        size_average (bool): 是否对SSIM值求平均 (Whether to average the SSIM values)

    Returns:
        Tensor: SSIM得分 (SSIM score)
    """
    # Image size = [Channel, H, W]
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """
    计算SSIM的内部函数：包括局部均值和方差的计算。
    Internal function for SSIM calculation, including local mean and variance.

    Args:
        img1 (Tensor): 第一张输入图像 (First input image)
        img2 (Tensor): 第二张输入图像 (Second input image)
        window (Variable): 高斯窗口 (Gaussian window)
        window_size (int): 窗口大小 (Window size)
        channel (int): 图像通道数 (Number of channels)
        size_average (bool): 是否求平均 (Whether to average the results)

    Returns:
        Tensor: SSIM得分 (SSIM score)
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()
