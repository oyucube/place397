# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 04:46:24 2016

@author: oyu
"""

from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, cuda
import numpy as np
import make_sampled_image
from env import xp


class SAF(chainer.Chain):
    def __init__(self, n_units=256, n_out=0, img_size=112, var=0.18, n_step=2, gpu_id=-1):
        super(SAF, self).__init__(
            # the size of the inputs to each layer will be inferred
            # glimpse network
            # 切り取られた画像を処理する部分　位置情報 (glimpse loc)と画像特徴量の積を出力
            glimpse_cnn_1=L.Convolution2D(3, 32, 4),  # in 20 out 16
            glimpse_cnn_2=L.Convolution2D(32, 64, 4),  # in 16 out 12
            glimpse_cnn_3=L.Convolution2D(64, 128, 4),  # in 12 out 8
            glimpse_full=L.Linear(4 * 4 * 128, n_units),
            glimpse_loc=L.Linear(3, n_units),

            # baseline network 強化学習の期待値を学習し、バイアスbとする
            baseline=L.Linear(n_units, 1),

            l_norm_c1=L.BatchNormalization(32),
            l_norm_c2=L.BatchNormalization(64),
            l_norm_c3=L.BatchNormalization(128),

            # 記憶を用いるLSTM部分
            rnn_1=L.LSTM(n_units, n_units),
            rnn_2=L.LSTM(n_units, n_units),

            # 注意領域を選択するネットワーク
            attention_loc=L.Linear(n_units, 2),
            attention_scale=L.Linear(n_units, 1),

            # 入力画像を処理するネットワーク
            context_cnn_1=L.Convolution2D(3, 32, 3),  # 64 to 62
            context_cnn_2=L.Convolution2D(32, 64, 4),  # 31 to 28
            context_cnn_3=L.Convolution2D(64, 64, 3),  # 14 to 12
            context_full=L.Linear(12 * 12 * 64, n_units),

            l_norm_cc1=L.BatchNormalization(32),
            l_norm_cc2=L.BatchNormalization(64),
            l_norm_cc3=L.BatchNormalization(64),

            class_full=L.Linear(n_units, n_out)
        )

        #
        # img parameter
        #
        if gpu_id == 0:
            self.use_gpu = True
        else:
            self.use_gpu = False
        self.img_size = img_size
        self.gsize = 20
        self.train = True
        self.var = 0.015
        self.vars = 0.015
        self.n_unit = n_units
        self.num_class = n_out
        # r determine the rate of position
        self.r = 0.5
        self.n_step = n_step

    def reset(self):
        self.rnn_1.reset_state()
        self.rnn_2.reset_state()

    def __call__(self, x, target, mode):
        self.reset()
        n_step = self.n_step
        num_lm = x.data.shape[0]
        if mode == 1:
            r_buf = 0
            l, s, b = self.first_forward(x, num_lm)
            for i in range(n_step):
                if i + 1 == n_step:
                    xm, lm, sm = self.make_img(x, l, s, num_lm, random=1)
                    l1, s1, y, b1 = self.recurrent_forward(xm, lm, sm)

                    loss, size_p = self.cul_loss(y, target, l, s, lm, sm)
                    r_buf += size_p
                    r = xp.where(
                        xp.argmax(y.data, axis=1) == xp.argmax(target.data, axis=1), 1, 0).reshape((num_lm, 1)).astype(
                        xp.float32)

                    loss += F.sum((r - b) * (r - b))
                    k = self.r * (r - b.data)
                    loss += F.sum(k * r_buf)

                    return loss / num_lm
                else:
                    xm, lm, sm = self.make_img(x, l, s, num_lm, random=1)
                    l1, s1, y, b1 = self.recurrent_forward(xm, lm, sm)
                    loss, size_p = self.cul_loss(y, target, l, s, lm, sm)
                    r_buf += size_p
                l = l1
                s = s1
                b = b1

        elif mode == 0:
            l, s, b1 = self.first_forward(x, num_lm, test=True)
            for i in range(n_step):
                if i + 1 == n_step:
                    xm, lm, sm = self.make_img(x, l, s, num_lm, random=0)
                    l1, s1, y, b = self.recurrent_forward(xm, lm, sm, test=True)
                    accuracy = y.data * target.data
                    return xp.sum(accuracy)
                else:
                    xm, lm, sm = self.make_img(x, l, s, num_lm, random=0)
                    l1, s1, y, b = self.recurrent_forward(xm, lm, sm, test=True)
                l = l1
                s = s1

    def use_model(self, x, t):
        self.reset()
        num_lm = x.data.shape[0]
        n_step = self.n_step
        s_list = xp.empty((n_step, num_lm, 1))
        l_list = xp.empty((n_step, num_lm, 2))
        l, s, b1 = self.first_forward(x, num_lm)
        for i in range(n_step):
            if i + 1 == n_step:
                xm, lm, sm = self.make_img(x, l, s, num_lm, random=0)
                l1, s1, y, b = self.recurrent_forward(xm, lm, sm)
                s_list[i] = s1.data
                l_list[i] = l1.data
                accuracy = y.data * t.data
                s_list = xp.power(10, s_list - 1)
                return xp.sum(accuracy, axis=1), l_list, s_list
            else:
                xm, lm, sm = self.make_img(x, l, s, num_lm, random=0)
                l1, s1, y, b = self.recurrent_forward(xm, lm, sm)
            l = l1
            s = s1
            s_list[i] = s.data
            l_list[i] = l.data
        return

    def first_forward(self, x, num_lm, test=False):
        x.volatile = test
        self.rnn_1(Variable(xp.zeros((num_lm, self.n_unit)).astype(xp.float32), volatile=test))
        h2 = F.relu(self.l_norm_cc1(self.context_cnn_1(F.average_pooling_2d(x, 4, stride=4))))
        h3 = F.relu(self.l_norm_cc2(self.context_cnn_2(F.max_pooling_2d(h2, 2, stride=2))))
        h4 = F.relu(self.l_norm_cc3(self.context_cnn_3(F.max_pooling_2d(h3, 2, stride=2))))
        h4r = F.relu(self.context_full(h4))
        h5 = F.relu(self.rnn_2(h4r))

        l = F.sigmoid(self.attention_loc(h5))
        s = F.sigmoid(self.attention_scale(h5))
        b = F.sigmoid(self.baseline(Variable(h5.data, volatile=test)))
        return l, s, b

    def recurrent_forward(self, xm, lm, sm, test=False):
        ls = xp.concatenate([lm.data, sm.data], axis=1)
        hgl = F.relu(self.glimpse_loc(Variable(ls, volatile=test)))

        hg1 = F.relu(self.l_norm_c1(self.glimpse_cnn_1(Variable(xm, volatile=test))))
        hg2 = F.relu(self.l_norm_c2(self.glimpse_cnn_2(hg1)))
        hg3 = F.relu(self.l_norm_c3(self.glimpse_cnn_3(F.max_pooling_2d(hg2, 2, stride=2))))
        hgf = F.relu(self.glimpse_full(hg3))

        hr1 = F.relu(self.rnn_1(hgl * hgf))
        # ベクトルの積
        hr2 = F.relu(self.rnn_2(hr1))
        l = F.sigmoid(self.attention_loc(hr2))
        s = F.sigmoid(self.attention_scale(hr2))
        y = F.softmax(self.class_full(hr1))
        b = F.sigmoid(self.baseline(Variable(hr2.data)))
        return l, s, y, b

    # loss 関数を計算

    def cul_loss(self, y, target, l, s, lm, sm):

        zm = xp.power(10, sm.data - 1)

        l1, l2 = F.split_axis(l, indices_or_sections=2, axis=1)
        m1, m2 = F.split_axis(lm, indices_or_sections=2, axis=1)
        ln_p = ((l1 - m1) * (l1 - m1) + (l2 - m2) * (l2 - m2)) / self.var / zm / zm / 2
        # size
        size_p = (sm - s) * (sm - s) / self.vars + ln_p

        accuracy = y * target

        loss = -F.sum(accuracy)
        return loss, size_p

    def make_img(self, x, l, s, num_lm, random=0):
        if random == 0:
            lm = Variable(xp.clip(l.data, 0, 1))
            sm = Variable(xp.clip(s.data, 0, 1))
        else:
            eps = xp.random.normal(0, 1, size=l.data.shape).astype(xp.float32)
            epss = xp.random.normal(0, 1, size=s.data.shape).astype(xp.float32)
            sm = xp.clip((s.data + xp.sqrt(self.var) * epss), 0, 1).astype(xp.float32)
            lm = xp.clip(l.data + xp.power(10, sm - 1) * eps * xp.sqrt(self.vars), 0, 1)
            sm = Variable(sm)
            lm = Variable(lm.astype(xp.float32))
        if self.use_gpu:
            xm = make_sampled_image.generate_xm_rgb_gpu(lm.data, sm.data, x.data, num_lm, g_size=self.gsize)
        else:
            xm = make_sampled_image.generate_xm_rgb(lm.data, sm.data, x.data, num_lm, g_size=self.gsize)
        return xm, lm, sm
