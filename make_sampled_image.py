# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 04:46:24 2016

@author: oyu
"""

from chainer import Variable, cuda
import numpy as np
from env import xp
from numba import jit


@jit
def generate_xm_rgb(lm, sm, img, num_lm, g_size, img_size=256):
    xm = np.empty((num_lm, 3, g_size * g_size)).astype(np.float32)
    img_buf = img.reshape((num_lm, 3, img_size * img_size))
    zm = np.power(10, sm - 1)
    for k in range(num_lm):
        for i in range(3):
            xr = np.linspace((lm[k][0] - zm[k] / 2), (lm[k][0] + zm[k] / 2), g_size)
            xr *= img_size
            xr = np.clip(xr, 0, img_size-1).astype(np.int32)
            yr = np.linspace((lm[k][1] - zm[k] / 2), (lm[k][1] + zm[k] / 2), g_size)
            yr *= img_size
            yr = np.clip(yr, 0, img_size - 1).astype(np.int32)
            xr = img_size * np.repeat(xr, g_size) + np.tile(yr, g_size)
            xm[k][i] = img_buf[k][i][xr]
    return xm.reshape(num_lm, 3, g_size, g_size).astype(np.float32)


@jit
def generate_xm_rgb_sa(lm, sm, img, num_lm, g_size, img_size=256):
    xm = np.empty((num_lm, 3, g_size * g_size)).astype(np.float32)
    img_buf = img.reshape((num_lm, 3, img_size * img_size))
    zm = np.power(10, sm - 1)/2
    for k in range(num_lm):
        for i in range(3):
            xr = np.linspace((lm[k][0] - zm[k] / 2), (lm[k][0] + zm[k] / 2), g_size)
            xr *= img_size
            xr = np.clip(xr, 0, img_size-1).astype(np.int32)
            yr = np.linspace((lm[k][1] - zm[k] / 2), (lm[k][1] + zm[k] / 2), g_size)
            yr *= img_size
            yr = np.clip(yr, 0, img_size - 1).astype(np.int32)
            xr = img_size * np.repeat(xr, g_size) + np.tile(yr, g_size)
            xm[k][i] = img_buf[k][i][xr]
    return xm.reshape(num_lm, 3, g_size, g_size).astype(np.float32)


# 切り取り画像の作成
def generate_xm_rgb_gpu(lm, sm, x, num_lm, g_size, img_size=256):
    xm = generate_xm_rgb(cuda.to_cpu(lm), cuda.to_cpu(sm), cuda.to_cpu(x), num_lm, g_size=g_size, img_size=img_size)
    return cuda.to_gpu(xm, device=0)


def generate_xm_rgb_sa_gpu(lm, sm, x, num_lm, g_size, img_size=256):
    xm = generate_xm_rgb_sa(cuda.to_cpu(lm), cuda.to_cpu(sm), cuda.to_cpu(x), num_lm, g_size=g_size, img_size=img_size)
    return cuda.to_gpu(xm, device=0)

