# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 04:46:33 2016

@author: oyu
"""
import os
# os.environ["CHAINER_TYPE_CHECK"] = "0" #ここでオフに  オンにしたかったら1にするかコメントアウト
import numpy as np
# 乱数のシード固定
#
# i = np.random()
# np.random.seed()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import chainer
from chainer import cuda, serializers
import sys
from tqdm import tqdm
import datetime
import importlib
import image_dataset
import socket


def get_batch(ds, index, repeat):
    nt = ds.num_target
    # print(index)
    batch_size = index.shape[0]
    bbx = np.empty((batch_size, 3, 256, 256))
    bbt = np.zeros((batch_size, nt))
    for bi in range(batch_size):
        bbx[bi] = ds[index[bi]][0]
        bbt[bi][ds[index[bi]][1]] = 1
    bbx = bbx.reshape(batch_size, 3, 256, 256).astype(np.float32)
    bbt = bbt.astype(np.float32)
    bbx = chainer.Variable(xp.asarray(xp.tile(bbx, (repeat, 1, 1, 1))), volatile="off")
    bbt = chainer.Variable(xp.asarray(xp.tile(bbt, (repeat, 1))), volatile="off")
    return bbx, bbt

#  引数分解
parser = argparse.ArgumentParser()

parser.add_argument("-b", "--batch_size", type=int, default=50,
                    help="batch size")
parser.add_argument("-e", "--epoch", type=int, default=50,
                    help="iterate training given epoch times")
parser.add_argument("-m", "--num_l", type=int, default=40,
                    help="a number of sample ")
parser.add_argument("-s", "--step", type=int, default=2,
                    help="look step")
parser.add_argument("-v", "--var", type=float, default=0.02,
                    help="sample variation")
parser.add_argument("-g", "--gpu", type=int, default=-1,
                    help="use gpu")
# train id
parser.add_argument("-i", "--id", type=str, default="5",
                    help="data id")
parser.add_argument("-a", "--am", type=str, default="model",
                    help="attention model")
# load model id
parser.add_argument("-l", "--l", type=str, default="",
                    help="load model name")

# model save id
parser.add_argument("-o", "--filename", type=str, default="v1",
                    help="prefix of output file names")
args = parser.parse_args()

file_id = args.filename
model_id = args.id
num_lm = args.num_l
n_epoch = args.epoch
train_id = args.id
label_file = args.id
num_step = args.step
train_b = args.batch_size
train_var = args.var
gpu_id = args.gpu

# naruto ならGPUモード
if socket.gethostname() == "naruto":
    gpu_id = 0
    log_dir = "/home/y-murata/storage/place397/"
    train_dataset = image_dataset.ImageDataset("/home/y-murata/data_256", label_file)
    val_dataset = image_dataset.ValidationDataset("/home/y-murata/val_256", label_file)
else:
    log_dir = ""
    train_dataset = image_dataset.ImageDataset(r"C:\Users\waka-lab\Documents\place365\data_256", label_file)
    val_dataset = image_dataset.ValidationDataset(r"C:\Users\waka-lab\Documents\place365\val_256", label_file)

xp = cuda.cupy if gpu_id >= 0 else np

data_max = train_dataset.len
test_max = val_dataset.len
img_size = 256
n_target = train_dataset.num_target
num_class = n_target
target_c = ""
test_b = test_max

# モデルの作成
model_file_name = args.am
sss = importlib.import_module(model_file_name)
model = sss.SAF(n_out=n_target, img_size=img_size, var=train_var, n_step=num_step, gpu_id=gpu_id)
# model load
if len(args.l) != 0:
    print("load model model/my{}{}.model".format(args.l, model_id))
    serializers.load_npz('model/my' + args.l + model_id + '.model', model)

# オプティマイザの設定
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

# gpuの設定
if gpu_id >= 0:
    chainer.cuda.get_device_from_id(gpu_id).use()
    model.to_gpu()

# ログの設定　精度、エラー率
acc1_array = np.full_like(np.zeros(n_epoch), np.nan)
train_acc = np.full_like(np.zeros(n_epoch), np.nan)
loss_array = np.full_like(np.zeros(n_epoch), np.nan)
max_acc = 0
date_id = datetime.datetime.now().strftime("%m%d%H%M")
log_dir = log_dir + "log/" + train_id + file_id + date_id
os.mkdir(log_dir)
out_file_name = log_dir + "/log"

log_filename = out_file_name + '.txt'
f = open(log_filename, 'w')
f.write("{} class recognition\nclass:{} use {} data set".format(num_class, target_c, model_id))
f.write("model:{}".format(model_file_name))
f.write("parameter")
f.write("step:{}\nnum_sample:{} \nbatch_size{}\nvar:{}".format(num_step, num_lm, train_b, train_var))
f.write("log dir:{}".format(out_file_name))
f.write("going to train {} epoch".format(n_epoch))
f.close()  # ファイルを閉じる

print("{} class recognition\nclass:{} use {} data set".format(num_class, target_c, model_id))
print("model:{}".format(model_file_name))
print("parameter")
print("step:{} num_sample:{} batch_size:{} var:{}".format(num_step, num_lm, train_b, train_var))
print("log dir:{}".format(log_dir))
print("going to train {} epoch".format(n_epoch))

#
# 訓練開始
#
for epoch in range(n_epoch):
    sys.stdout.write("(epoch: {})\n".format(epoch + 1))
    sys.stdout.flush()
    #   学習    
    perm = np.random.permutation(data_max)
    for i in tqdm(range(0, data_max, train_b), ncols=60):
        x, t = get_batch(train_dataset, perm[i:i+train_b], num_lm)
        # 順伝播
        model.cleargrads()
        loss_func = model(x, t, mode=1)
        del x
        del t
        loss_array[epoch] = loss_func.data
        loss_func.backward()
        loss_func.unchain_backward()  # truncate
        optimizer.update()

    # 評価
    # 順伝播
    acc = 0
    t_acc = 0
    di = 0
    perm = np.random.permutation(test_max)
    for i in range(0, 100, test_b):
        di += 1
        x, t = get_batch(train_dataset, perm[i:i+train_b], num_lm)
        # 順伝播
        x, t = get_batch(val_dataset, perm[0:test_b], 1)
        acc += model(x, t, mode=0)

        x, t = get_batch(train_dataset, perm[0:test_b], 1)
        t_acc += model(x, t, mode=0)
        del x
        del t

    # 記録
    acc1_array[epoch] = acc / test_b
    train_acc[epoch] = t_acc / test_b
    print("test_acc:{:1.4f} train_acc:{:1.4f}".format(acc1_array[epoch], train_acc[epoch]))
    best = ""
    if acc > max_acc:
        max_acc = acc
        best = "best"
    # 分類精度の保存
    with open(log_filename, mode='a') as fh:
        fh.write("test_acc:{:1.4f} train_acc:{:1.4f}".format(acc1_array[epoch], train_acc[epoch]))

    np.save(log_dir + "/test_acc.npy", acc1_array)
    np.save(log_dir + "/train_acc.npy", train_acc)
    # モデルの保存
    if gpu_id >= 0:
        model.to_cpu()
        serializers.save_npz(log_dir + "/" + best + file_id + train_id + '.model', model)
        model.to_gpu()
    else:
        serializers.save_npz(log_dir + "/" + best + file_id + train_id + '.model', model)
    # グラフの作成と保存
    plt.figure()
    plt.ylim([0, 1])
    plt.plot(acc1_array, color="green")
    plt.plot(train_acc, color="blue")
    plt.savefig(log_dir + "/acc.png")
    plt.figure()
    plt.plot(loss_array)
    plt.savefig(log_dir + "/loss.png")

with open(log_filename, mode='a') as fh:
    fh.write("last acc:{}  max_acc:{}\n".format(acc1_array[n_epoch - 1], max_acc))
