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


def generate_xm_cons(lm, sm, img, num_lm, g_size, img_size=256):
    xm = np.empty((num_lm, g_size * g_size)).astype(np.float32)
    img_buf = img.reshape((num_lm, img_size * img_size))
    zm = sm
    for k in range(num_lm):
        xr = np.linspace((lm[k][0] - zm[k] / 2), (lm[k][0] + zm[k] / 2), g_size)
        xr *= img_size
        xr = np.clip(xr, 0, img_size-1).astype(np.int32)
        yr = np.linspace((lm[k][1] - zm[k] / 2), (lm[k][1] + zm[k] / 2), g_size)
        yr *= img_size
        yr = np.clip(yr, 0, img_size - 1).astype(np.int32)
        xr = img_size * np.repeat(xr, g_size) + np.tile(yr, g_size)
        xm[k] = img_buf[k][xr]
    return xm.reshape(num_lm, 1, g_size, g_size).astype(np.float32)


# 切り取り画像の作成
def generate_xm_rgb_gpu(lm, sm, x, num_lm, g_size, img_size=256):
    xm = generate_xm_rgb(cuda.to_cpu(lm), cuda.to_cpu(sm), cuda.to_cpu(x), num_lm, g_size=g_size, img_size=img_size)
    return cuda.to_gpu(xm, device=0)


def generate_xm_in_gpu(lm, sm, img, num_lm, g_size, img_size=256):
    xm = xp.empty((num_lm, g_size * g_size)).astype(xp.float32)
    img_buf = img.reshape((num_lm, img_size * img_size))
    zm = xp.power(10, sm - 1)
    for k in range(num_lm):
        xr = xp.linspace((lm[k][0] - zm[k] / 2), (lm[k][0] + zm[k] / 2), g_size)
        xr *= img_size
        xr = xp.clip(xr, 0, img_size-1).astype(np.int32)
        yr = xp.linspace((lm[k][1] - zm[k] / 2), (lm[k][1] + zm[k] / 2), g_size)
        yr *= img_size
        yr = xp.clip(yr, 0, img_size - 1).astype(np.int32)
        xr = img_size * np.repeat(xr, g_size) + xp.tile(yr, g_size)
        xm[k] = img_buf[k][xr]
    return xm.reshape(num_lm, 1, g_size, g_size).astype(xp.float32)


def generate_xm_const_gpu(lm, sm, x, num_lm, g_size, img_size=256):
    xm = generate_xm_cons(cuda.to_cpu(lm), cuda.to_cpu(sm), cuda.to_cpu(x), num_lm, g_size=g_size, img_size=img_size)
    return cuda.to_gpu(xm, device=0)


def generate_xm_const_size_gpu(lm, sm, x, num_lm, g_size, img_size=256):
    s = g_size / img_size + np.zeros((num_lm, 1))
    s = np.log10(s) + 1
    xm = generate_xm_cons(cuda.to_cpu(lm), s, cuda.to_cpu(x), num_lm, g_size=g_size, img_size=img_size)
    return cuda.to_gpu(xm, device=0)


def generate_xm_const_size(lm, sm, x, num_lm, g_size=20, img_size=256):
    s = g_size / img_size + np.zeros((num_lm, 1))
    # make g_size image
    s = np.log10(s) + 1
    xm = generate_xm_cons(lm, s, x, num_lm, g_size=g_size, img_size=img_size)
    return xm
