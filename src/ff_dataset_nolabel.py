import numpy as np
import torch
from src import utils


class FF_Dataset_Nolabel(torch.utils.data.Dataset):
    '''不再贴 label 产生正负样本, 而是通过数据增强和数据污染产生正负样本.'''
    def __init__(self, opt, partition):
        dataset_dict = {'mnist': utils.get_MNIST_partition, 'cifar10': utils.get_CIFAR10_partition}
        self.opt = opt
        self.pos_dataset = dataset_dict[self.opt.input.dataset](opt, partition)
        self.neg_dataset = utils.negative_dataset_transform(self.pos_dataset)   # neg_dataset 中只有 images, 没有 labels.


    def __getitem__(self, index):
        '''返回 inputs dict: 包含 pos/neg/neutral sample 和 labels dict: 包含 class_label'''
        pos_sample, class_label = self.pos_dataset[index]
        neg_sample = self.neg_dataset[index]
        neutral_sample, _ = self.pos_dataset[index]

        inputs = {
            "pos_images": pos_sample,
            "neg_images": neg_sample,
            "neutral_sample": neutral_sample,
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.pos_dataset)