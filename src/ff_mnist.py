import numpy as np
import torch

from src import utils


class FF_MNIST(torch.utils.data.Dataset):
    '''在朴素的 MNIST 图像数据集基础上, 新增了获取 pos/neg/neutral sample 的方法.'''
    def __init__(self, opt, partition, num_classes=10):
        self.opt = opt
        self.mnist = utils.get_MNIST_partition(opt, partition)
        self.num_classes = num_classes
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes

    def __getitem__(self, index):
        '''返回 inputs dict: 包含 pos/neg/neutral sample 和 labels dict: 包含 class_label'''
        pos_sample, neg_sample, neutral_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample,
            "neg_images": neg_sample,
            "neutral_sample": neutral_sample,
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.mnist)

    def _get_pos_sample(self, sample, class_label):
        '''把单个 sample 的第一行前10列替换为 one-hot vector.'''
        # sample 是 ToTensor() 的返回值, 形状是 C x H x W, 故 pos_sample 的第 0 维是 Channels.
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label), num_classes=self.num_classes
        )
        pos_sample = sample.clone()
        pos_sample[:, 0, : self.num_classes] = one_hot_label
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        '''Create randomly sampled one-hot label.'''
        classes = list(range(self.num_classes))
        classes.remove(class_label)  # Remove true label from possible choices.
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes
        )
        neg_sample = sample.clone()
        neg_sample[:, 0, : self.num_classes] = one_hot_label
        return neg_sample

    def _get_neutral_sample(self, z):
        z[:, 0, : self.num_classes] = self.uniform_label
        return z

    def _generate_sample(self, index):
        '''返回完整的 MNIST sample: 包含 pos/neg/neutral sample 以及 class_label'''
        sample, class_label = self.mnist[index]
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        return pos_sample, neg_sample, neutral_sample, class_label
