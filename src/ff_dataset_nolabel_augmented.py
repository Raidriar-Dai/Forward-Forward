import numpy as np
import torch
from src import utils


class FF_Dataset_Nolabel_Augmented(torch.utils.data.Dataset):
    '''
    把(原数据集 + 1*数据增强)作为正样本(共10万张), 2*数据污染作为负样本(共10万张).
    只有 cifar10 数据集, 不适用于 mnist.
    只有当 partition == train, 才生成 augmented_cifar10; 若 partition == test, 则只有 original_cifar10.
    '''
    def __init__(self, opt, partition):
        self.opt = opt
        original_cifar10 = utils.get_CIFAR10_partition_normalized(opt, partition)
        if partition == "train":
            augmented_cifar10 = utils.get_CIFAR10_train_augmented(opt) 
            self.pos_dataset = torch.utils.data.ConcatDataset([original_cifar10, augmented_cifar10])
            self.neg_dataset = utils.negative_dataset_transform(original_cifar10, alpha=2)   
            # neg_dataset 数据类型是 list; 其中只有 images, 没有 labels; 大小默认为 original_cifar10 的两倍.
        elif partition == "test":
            self.pos_dataset = original_cifar10
            self.neg_dataset = utils.negative_dataset_transform(original_cifar10)
            # 实际上, neg_dataset 也可以换成 original_cifar10, 因为 test 时用不到 neg_dataset.

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