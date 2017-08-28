import numpy as np
from PIL import Image
import csv
import chainer
import os
from chainer import datasets


class ImageDataset(chainer.dataset.DatasetMixin):
    def __init__(self, path):
        self.path = path
        pairs = []
        n = 0
        with open('data/data_label_list.csv', newline='') as f:
            tsv = csv.reader(f, delimiter=',')
            for row in tsv:
                c_path = "/" + row[0][0:1] + "/" + row[0]
                print(c_path)
                files = os.listdir(path + c_path)
                for file in files:
                    file_path = path + c_path + "/" + file
                    if 'jpg' in file_path:
                        pairs.append([file_path, row[1]])
                        if n < int(row[1]):
                            n = int(row[1])
        self._pairs = pairs
        self.len = len(self._pairs)
        self.num_target = n + 1

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        filename = self._pairs[i][0]
        image = Image.open(filename)
        image = image.convert("RGB")
        image_array = np.asarray(image)
        image_array = image_array.astype('float32')
        image_array = image_array / 255
        image_array = image_array.transpose(2, 0, 1)  # order of rgb / h / w
        label = np.int32(self._pairs[i][1])
        return image_array, label


class ValidationDataset(chainer.dataset.DatasetMixin):
    def __init__(self, path):
        self.path = path
        pairs = []
        label_list = []
        n = 0
        with open('data/data_label_list.csv', newline='') as f:
            tsv = csv.reader(f, delimiter=',')
            for row in tsv:
                label_list.append([row[1], row[2]])

        with open('data/places365_val.txt', newline='') as f2:
            vals = csv.reader(f2, delimiter=' ')
            for val in vals:
                for label in label_list:
                    if int(label[1]) == int(val[1]):
                        file_path = path + "/" + val[0]
                        pairs.append([file_path, label[0]])
                        if n < int(label[0]):
                            n = int(label[0])

        self._pairs = pairs
        self.len = len(self._pairs)
        self.num_target = n + 1

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        filename = self._pairs[i][0]
        image = Image.open(filename)
        image = image.convert("RGB")
        image_array = np.asarray(image)
        image_array = image_array.astype('float32')
        image_array = image_array / 255
        image_array = image_array.transpose(2, 0, 1)  # order of rgb / h / w
        label = np.int32(self._pairs[i][1])
        return image_array, label

