import os
import random
from datetime import timedelta

import numpy as np
import torch
import torchvision

import wandb
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from src import ff_dataset, ff_model


def parse_args(opt):
    '''为 np/torch/random 都设置同一种子, 并打印本次实验的 config 信息, 最后原样返回 opt.'''
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    print(OmegaConf.to_yaml(opt))   # 输出配置文件的所有参数信息.
    return opt


def show_model_parameters(opt):
    '''(自定义)打印出 model.classification_loss.parameters(), 查看其具体有哪些参数.'''
    model = ff_model.FF_model(opt)
    if "cuda" in opt.device:
        model = model.cuda() 
    for x in model.classification_loss.parameters():
        print(x)


def get_model_and_optimizer(opt):
    model = ff_model.FF_model(opt)
    if "cuda" in opt.device:
        model = model.cuda()
    # print(model, "\n")  # 输出 FF_model 的组件信息.

    # "one-pass softmax" 的训练参数:
    if opt.training.test_mode == "one_pass_softmax":
        main_model_params = [
            p
            for p in model.parameters()
            if all(p is not x for x in model.linear_classifier.parameters())
            # 疑问: 经试验, model.classification_loss.parameters() 是一个空的 generator?
        ]
        # Torch.optim 的 "per-parameter options" 初始化方法: 用多个字典定义多个独立的 parameter group.
        # 纠错: main model 中的参数是否应该是 model.model.parameters(),
        # 而 downstream classification model 中的参数是否应该是 model.linear_classifier.parameters()?
        optimizer = torch.optim.SGD(
            [
                {
                    # main model 中需要更新的参数.
                    "params": main_model_params,
                    "lr": opt.training.learning_rate,
                    "weight_decay": opt.training.weight_decay,
                    "momentum": opt.training.momentum,
                },
                {
                    # downstream classification model 中需要更新的参数.
                    "params": model.linear_classifier.parameters(),
                    "lr": opt.training.downstream_learning_rate,
                    "weight_decay": opt.training.downstream_weight_decay,
                    "momentum": opt.training.momentum,
                },
            ]
        )

    # "compute goodness for each label" 的训练参数:
    elif opt.training.test_mode == "compute_each_label":
        optimizer = torch.optim.SGD(
            [
                {
                    # 只有 ff_layer 含有需要更新的参数, 不再有 cls_layer.
                    "params": model.parameters(),
                    "lr": opt.training.learning_rate,
                    "weight_decay": opt.training.weight_decay,
                    "momentum": opt.training.momentum,
                }
            ]
        )

    return model, optimizer


def get_data(opt, partition):
    '''由 FF_Dataset 封装返回一个 dataloader'''
    dataset = ff_dataset.FF_Dataset(opt, partition)

    # Improve reproducibility in dataloader.
    g = torch.Generator()
    g.manual_seed(opt.seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.input.batch_size,
        drop_last=True,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=4,
        persistent_workers=True,
    )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# def get_MNIST_partition(opt, partition):
#     '''获取 dataset 的主要成分(用于 dataset 的构造)'''
#     # 这里的 dataset 只被 transform 成为 Tensor, 没有经过 normalization 或 flatten. 
#     if partition in ["train", "val", "train_val"]:
#         mnist = torchvision.datasets.MNIST(
#             os.path.join(get_original_cwd(), opt.input.path),
#             train=True,
#             download=True,
#             transform=torchvision.transforms.ToTensor(),
#         )
#     elif partition in ["test"]:
#         mnist = torchvision.datasets.MNIST(
#             os.path.join(get_original_cwd(), opt.input.path),
#             train=False,
#             download=True,
#             transform=torchvision.transforms.ToTensor(),
#         )
#     else:
#         raise NotImplementedError

#     # 分 train 与 val 两种情况, 进一步分割 training set.
#     if partition == "train":
#         mnist = torch.utils.data.Subset(mnist, range(50000))
#     elif partition == "val":
#         mnist = torchvision.datasets.MNIST(
#             os.path.join(get_original_cwd(), opt.input.path),
#             train=True,
#             download=True,
#             transform=torchvision.transforms.ToTensor(),
#         )   # 这里为什么要再写一遍完全相同的数据集? 这样会打乱得到的 mnist 中图像的顺序吗?
#         # 可能的原因: 有些情况下, 验证集 和 训练集, 所需要的 transform 操作是不一样的.
#         mnist = torch.utils.data.Subset(mnist, range(50000, 60000))

#     return mnist


def get_MNIST_partition(opt, partition):
    '''用 random_split 来获取 MNIST 的 training set 与 validation set.''' 
    if partition in ["test"]:
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.input.path),
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    elif partition in ["train", "val", "train_val"]:
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        mnist_train, mnist_val = torch.utils.data.random_split(
            mnist, [len(mnist) - opt.input.val_size, opt.input.val_size]
            )
        if partition == "train":
            mnist = mnist_train
        elif partition == "val":
            mnist = mnist_val
    else:
        raise NotImplementedError

    return mnist


def get_CIFAR10_partition(opt, partition):
    '''获取 dataset 的主要成分(用于 dataset 的构造)'''
    if partition in ["test"]:
        cifar10 = torchvision.datasets.CIFAR10(
            os.path.join(get_original_cwd(), opt.input.path),
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    elif partition in ["train", "val", "train_val"]:
        cifar10 = torchvision.datasets.CIFAR10(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        cifar10_train, cifar10_val = torch.utils.data.random_split(
            cifar10, [len(cifar10) - opt.input.val_size, opt.input.val_size]
            )
        if partition == "train":
            cifar10 = cifar10_train
        elif partition == "val":
            cifar10 = cifar10_val
    else:
        raise NotImplementedError

    return cifar10


def dict_to_cuda(dict):
    '''把 dict 中存储的 value 放到 cuda 上'''
    for key, value in dict.items():
        dict[key] = value.cuda(non_blocking=True)
    return dict


def preprocess_inputs(opt, inputs, labels):
    '''把 inputs 和 labels 两个 dict 都放到 cuda 上'''
    if "cuda" in opt.device:
        inputs = dict_to_cuda(inputs)
        labels = dict_to_cuda(labels)
    return inputs, labels


def get_linear_cooldown_afterHalf_lr(opt, epoch, lr):
    '''当 epoch 过半之后, lr 线性减小, 当 epoch 跑满时减为 0.'''
    if epoch >= (opt.training.epochs // 2):
        return lr * 2 * (opt.training.epochs - epoch) / opt.training.epochs
    else:
        return lr


def get_linear_cooldown_lowerBound_lr(opt, epoch, lr):
    '''当 epoch 过半之后, lr 线性减小, 直至达到 opt.training.lower_bound 就不再减小, 
    一直维持到 epoch 跑满.'''
    if epoch >= (opt.training.epochs // 2):
        return max(lr * 2 * (opt.training.epochs - epoch) / opt.training.epochs, 
                   opt.training.lower_bound)
    else:
        return lr


def get_linear_cooldown_fromBegin_lr(opt, epoch, lr):
    '''从 epoch=0 起, lr 就线性减小, 当 epoch 跑满时减为 0.'''
    return lr * (opt.training.epochs - epoch) / opt.training.epochs


def get_linear_cooldown_smallerSlope_lr(opt, epoch, lr):
    '''当 epoch 过半之后, lr 线性减小, 当 epoch 跑满时减为 opt.training.slope_end.'''
    half_epochs = opt.training.epochs // 2
    if epoch >= half_epochs:
        step = (lr - opt.training.slope_end) / (half_epochs - 1)
        return lr - step * (epoch - half_epochs)
    else:
        return lr


def update_learning_rate(opt, optimizer, epoch):
    '''在每个新的 epoch 都要 cooldown 当前 optimizer 的 lr.'''
    lr_schedule_dict = {"after_half": get_linear_cooldown_afterHalf_lr,
                        "lower_bound": get_linear_cooldown_lowerBound_lr,
                        "from_begin": get_linear_cooldown_fromBegin_lr,
                        "smaller_slope": get_linear_cooldown_smallerSlope_lr}
    lr_schedule = opt.training.lr_schedule

    if opt.training.test_mode == "one_pass_softmax":
        optimizer.param_groups[0]["lr"] = lr_schedule_dict[lr_schedule[0]](
            opt, epoch, opt.training.learning_rate
        )
        optimizer.param_groups[1]["lr"] = lr_schedule_dict[lr_schedule[1]](
            opt, epoch, opt.training.downstream_learning_rate
        )

    elif opt.training.test_mode == "compute_each_label":
        optimizer.param_groups[0]["lr"] = lr_schedule_dict[lr_schedule[0]](
            opt, epoch, opt.training.learning_rate
        )

    return optimizer


def get_accuracy(opt, output, target):
    """Computes the accuracy."""
    with torch.no_grad():
        prediction = torch.argmax(output, dim=1)
        return (prediction == target).sum() / opt.input.batch_size


def print_results(opt, partition, iteration_time, scalar_outputs, epoch=None):
    if epoch is not None:
        print(f"Epoch {epoch} \t", end="")

    print(
        f"{partition} \t \t"
        f"Time: {timedelta(seconds=iteration_time)} \t",
        end="",
    )
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            print(f"{key}: {value:.4f} \t", end="") # 先输出当前 epoch 的训练/测试结果.

        # 再把当前 epoch 的训练/测试结果上传到 wandb.
        # "one_pass_softmax" 测试模式下, 存在 cls_loss 与 cls_acc 两个 metric.
        if opt.training.test_mode == "one_pass_softmax":
            if partition == "train":
                wandb.log({ "Loss": scalar_outputs["Loss"],
                            "Peer Normalization": scalar_outputs["Peer Normalization"],
                            "loss_layer_0": scalar_outputs["loss_layer_0"],
                            "loss_layer_1": scalar_outputs["loss_layer_1"],
                            "loss_layer_2": scalar_outputs["loss_layer_2"],
                            "ff_acc_layer_0": scalar_outputs["ff_accuracy_layer_0"],
                            "ff_acc_layer_1": scalar_outputs["ff_accuracy_layer_1"],
                            "ff_acc_layer_2": scalar_outputs["ff_accuracy_layer_2"],
                            "cls_loss": scalar_outputs["classification_loss"],
                            "cls_acc": scalar_outputs["classification_accuracy"] })
            elif partition == "val":
                wandb.log({ "Val Loss": scalar_outputs["Loss"],
                            "Val cls_loss": scalar_outputs["classification_loss"],
                            "Val cls_acc": scalar_outputs["classification_accuracy"] })
            elif partition == "test":
                wandb.log({ "Test Loss": scalar_outputs["Loss"],
                            "Test cls_loss": scalar_outputs["classification_loss"],
                            "Test cls_acc": scalar_outputs["classification_accuracy"] })

        # "compute_each_label" 测试模式下, 只存在 cls_acc 这个 metric.
        elif opt.training.test_mode == "compute_each_label":
            if partition == "train":
                wandb.log({ "Loss": scalar_outputs["Loss"],
                            "Peer Normalization": scalar_outputs["Peer Normalization"],
                            "loss_layer_0": scalar_outputs["loss_layer_0"],
                            "loss_layer_1": scalar_outputs["loss_layer_1"],
                            "loss_layer_2": scalar_outputs["loss_layer_2"],
                            "ff_acc_layer_0": scalar_outputs["ff_accuracy_layer_0"],
                            "ff_acc_layer_1": scalar_outputs["ff_accuracy_layer_1"],
                            "ff_acc_layer_2": scalar_outputs["ff_accuracy_layer_2"],
                            "cls_acc": scalar_outputs["classification_accuracy"] })
            elif partition == "val":
                wandb.log({ "Val cls_acc": scalar_outputs["classification_accuracy"] })
            elif partition == "test":
                wandb.log({ "Test cls_acc": scalar_outputs["classification_accuracy"] })

    print()


def log_results(result_dict, scalar_outputs, num_steps):
    for key, value in scalar_outputs.items():
        if isinstance(value, float):
            result_dict[key] += value / num_steps
        else:
            result_dict[key] += value.item() / num_steps
    return result_dict


def overlay_label_on_z(num_classes, z, label):
    '''z 为 二维 tensor, label 为 0-9 之间的数字;
    把 z 的每行前 10 列元素都替换成 label 所指示的那个标签.'''
    z_labeled = z.clone()

    # 必须把 index tensor 放到 gpu 上, 否则它与 z_labeled 就不在同一设备上.
    index = torch.arange(num_classes).cuda()
    z_labeled.index_fill_(1, index, 0)
    z_labeled[:, label] = 1

    return z_labeled