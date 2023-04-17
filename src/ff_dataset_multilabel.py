import numpy as np
import torch

from src import utils

class FF_Dataset_Multilabel(torch.utils.data.Dataset):
    '''pos/neg data 通过在不同 channel 的不同位置贴多个 label 来实现.'''
    def __init__(self, opt, partition):
        dataset_dict = {'mnist': utils.get_MNIST_partition, 'cifar10': utils.get_CIFAR10_partition}
        self.opt = opt
        self.dataset = dataset_dict[self.opt.input.dataset](opt, partition)
        self.num_classes = self.opt.input.num_classes
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
        return len(self.dataset)

    def _get_pos_sample(self, sample, class_label):
        '''把单个 sample 的第一行前10列替换为 one-hot vector.'''
        # sample 是 ToTensor() 的返回值, 形状是 C x H x W, 故 pos_sample 的第 0 维是 Channels.
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label), num_classes=self.num_classes
        )
        pos_sample = sample.clone()
        pos_sample[0, 0, :10] = one_hot_label
        pos_sample[0, 1, :10] = one_hot_label
        pos_sample[1, 14, 11:21] = one_hot_label
        pos_sample[1, 15, 11:21] = one_hot_label
        pos_sample[2, 30, 22:] = one_hot_label
        pos_sample[2, 31, 22:] = one_hot_label
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
        neg_sample[0, 0, :10] = one_hot_label
        neg_sample[0, 1, :10] = one_hot_label
        neg_sample[1, 14, 11:21] = one_hot_label
        neg_sample[1, 15, 11:21] = one_hot_label
        neg_sample[2, 30, 22:] = one_hot_label
        neg_sample[2, 31, 22:] = one_hot_label
        return neg_sample

    def _get_neutral_sample(self, z):
        z[0, 0, :10] = self.uniform_label
        z[0, 1, :10] = self.uniform_label
        z[1, 14, 11:21] = self.uniform_label
        z[1, 15, 11:21] = self.uniform_label
        z[2, 30, 22:] = self.uniform_label
        z[2, 31, 22:] = self.uniform_label
        return z

    def _generate_sample(self, index):
        '''返回完整的 MNIST sample: 包含 pos/neg/neutral sample 以及 class_label'''
        sample, class_label = self.dataset[index]
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        return pos_sample, neg_sample, neutral_sample, class_label
