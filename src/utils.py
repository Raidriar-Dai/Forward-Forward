import os
import random
from datetime import timedelta
from collections import ChainMap
from scipy.signal import convolve2d
import numpy as np
import torch 
from torch import tensor
import torchvision
import torchvision.transforms as transforms

import wandb
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from src import ff_dataset, ff_model_convmixer, bp_model_mlp


def parse_args(opt):
    '''为 np/torch/random 都设置同一种子, 并打印本次实验的 config 信息, 最后原样返回 opt.'''
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    print(OmegaConf.to_yaml(opt))   # 输出配置文件的所有参数信息.
    return opt


# def show_model_parameters(opt):
#     '''(自定义)打印出 model.classification_loss.parameters(), 查看其具体有哪些参数.'''
#     model = ff_model.FF_model(opt)
#     if "cuda" in opt.device:
#         model = model.cuda() 
#     for x in model.classification_loss.parameters():
#         print(x)


def get_model_and_optimizer(opt):
 
    if opt.model.training_type == 'ff':
        # TODO 1: 将来若要把 conv_net 等其他 ff 网络结构整合进来,
        # 就在这里建一个 dict, 存放 <file.model>, 用来生成下面的 model object.
        model = ff_model_convmixer.FF_model_convmixer(opt)
        if "cuda" in opt.device:
            model = model.cuda()
        # print(model, "\n")  # 输出 FF_model 的组件信息.

        # "one-pass softmax" 的训练参数:
        if opt.training.test_mode == "one_pass_softmax":
            main_model_params = [
                p
                for p in model.parameters()
                if all(p is not x for x in model.linear_classifier.parameters())
            ]
            # Torch.optim 的 "per-parameter options" 初始化方法: 用多个字典定义多个独立的 parameter group.
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

    elif opt.model.training_type == 'bp':
        model = bp_model_mlp.BP_model_mlp(opt)
        if "cuda" in opt.device:
            model = model.cuda()
        # print(model, "\n")  # 输出 FF_model 的组件信息.

        optimizer = torch.optim.SGD(
            [
                {
                    "params": model.parameters(),
                    "lr": opt.training.learning_rate,
                    "weight_decay": opt.training.weight_decay,
                    "momentum": opt.training.momentum,
                }
            ]
        )
    
    else:
        raise NotImplementedError

    return model, optimizer


def get_data(opt, partition):
    '''返回一个 dataloader, 其中的 dataset 是普通的/用 FF_Dataset 包装过的 MNIST/CIFAR10.'''
    if opt.model.training_type == 'ff':
        dataset = ff_dataset.FF_Dataset(opt, partition)
    elif opt.model.training_type == 'bp':
        dataset_dict = {'mnist': get_MNIST_partition, 'cifar10': get_CIFAR10_partition}
        dataset = dataset_dict[opt.input.dataset](opt, partition)
    else:
        raise NotImplementedError

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
#     '''获取 MNIST 的 training set(前 50000 张) 与 validation set(后 10000 张)'''
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
        # 若 to_validate == False, 则不要 validation, 直接返回 training set 即可.
        if opt.training.to_validate:
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
    '''用 random_split 来获取 CIFAR10 的 unnormalized 的 training set 与 validation set.'''
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
        # 若 to_validate == False, 则不要 validation, 直接返回 training set 即可. 
        if opt.training.to_validate:
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


def get_CIFAR10_partition_normalized(opt, partition):
    '''返回经过 transform.Normalize 的 CIFAR10 dataset'''
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)
    
    train_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])
    
    if partition in ["test"]:
        cifar10 = torchvision.datasets.CIFAR10(
            os.path.join(get_original_cwd(), opt.input.path),
            train=False,
            download=True,
            transform=train_test_transform,
        )
    elif partition in ["train", "val", "train_val"]:
        cifar10 = torchvision.datasets.CIFAR10(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=train_test_transform,
        )
        # 若 to_validate == False, 则不要 validation, 直接返回 training set 即可. 
        if opt.training.to_validate:
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


def get_CIFAR10_train_augmented(opt):
    '''返回经过数据增强的 CIFAR10 train set, 只作为 train 时的正样本.'''
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees = 0, translate = (0.2, 0.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    cifar10 = torchvision.datasets.CIFAR10(
        os.path.join(get_original_cwd(), opt.input.path),
        train=True,
        download=True,
        transform=train_transform,
    )

    return cifar10


def get_CIFAR10_train_unnormalized_augmented(opt):
    '''返回经过数据增强(不包括 transform.Normalize)的 CIFAR10 train set, 只作为 train 时的正样本.'''
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees = 0, translate = (0.2, 0.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])

    cifar10 = torchvision.datasets.CIFAR10(
        os.path.join(get_original_cwd(), opt.input.path),
        train=True,
        download=True,
        transform=train_transform,
    )

    return cifar10


def dict_to_cuda(dict):
    '''把 dict 中存储的 value 放到 cuda 上'''
    for key, value in dict.items():
        dict[key] = value.cuda(non_blocking=True)
    return dict


def preprocess_inputs(opt, inputs, labels):
    '''把 inputs 和 labels 都放到 cuda 上'''
    if "cuda" in opt.device:
        if isinstance(inputs, torch.Tensor) and isinstance(labels, torch.Tensor):
            inputs = inputs.cuda()
            labels = labels.cuda()
        elif isinstance(inputs, dict) and isinstance(labels, dict):
            inputs = dict_to_cuda(inputs)
            labels = dict_to_cuda(labels)
    return inputs, labels


# def get_piecewise_linear_cooldown_lr(opt, epoch, lr, boundary_points):
#     '''通用线性 lr scheduler: 分段线性函数.'''
#     start_point, end_point = (0, lr), (opt.training.epochs - 1, boundary_points[-1])
#     boundary_points = [start_point] + boundary_points
#     boundary_points[-1] = end_point

#     for i in range(len(boundary_points) - 1):
#         # 这里两边的不等号都用了 <=, 是为了处理边界情况.
#         if epoch >= boundary_points[i][0] and epoch <= boundary_points[i+1][0]:
#             start_epoch, start_lr = boundary_points[i][0], boundary_points[i][1]
#             end_epoch, end_lr = boundary_points[i+1][0], boundary_points[i+1][1]
#             k = (end_lr - start_lr) / (end_epoch - start_epoch)
#             return k * (epoch - start_epoch) + start_lr
#         else:
#             continue

#     raise Exception("Failed to return a learning rate value.")


# # 为了 sweep 的方便, 仍然保留这个简易的 lr_scheduler.
# def get_linear_cooldown_smallerSlope_lr(opt, epoch, lr):
#     '''当 epoch 过半之后, lr 线性减小, 当 epoch 跑满时减为 opt.training.final_lr.'''
#     half_epochs = opt.training.epochs // 2
#     if epoch >= half_epochs:
#         step = (lr - opt.training.final_lr) / (half_epochs - 1)
#         return lr - step * (epoch - half_epochs)
#     else:
#         return lr


def get_piecewise_linear_warmup_cooldown_lr(opt, epoch, downstream=False):
    '''带有 warmup 和 cooldown 的分段线性函数'''
    if downstream:
        x_pts, y_pts = opt.training.downstream_lr_boundary_points_x, opt.training.downstream_lr_boundary_points_y
        x_pts = [x_pt * opt.training.epochs for x_pt in x_pts]
        y_pts = [y_pt * opt.training.downstream_learning_rate for y_pt in y_pts]
    else:
        x_pts, y_pts = opt.training.lr_boundary_points_x, opt.training.lr_boundary_points_y
        x_pts = [x_pt * opt.training.epochs for x_pt in x_pts]
        y_pts = [y_pt * opt.training.learning_rate for y_pt in y_pts]
    
    return (lambda t: np.interp([t], x_pts, y_pts)[0])(epoch)


def update_learning_rate(opt, optimizer, epoch):
    '''在每个新的 epoch 都要 warmup / cooldown 当前 optimizer 的 lr.'''
    if opt.model.training_type == 'ff':
        if opt.training.test_mode == "one_pass_softmax":
            optimizer.param_groups[0]["lr"] = get_piecewise_linear_warmup_cooldown_lr(
                opt, epoch, downstream=False
            )
            optimizer.param_groups[1]["lr"] = get_piecewise_linear_warmup_cooldown_lr(
                opt, epoch, downstream=True
            )

        elif opt.training.test_mode == "compute_each_label":
            optimizer.param_groups[0]["lr"] = get_piecewise_linear_warmup_cooldown_lr(
                opt, epoch, downstream=False
            )

    elif opt.model.training_type == 'bp':
        optimizer.param_groups[0]["lr"] = get_piecewise_linear_warmup_cooldown_lr(
            opt, epoch, downstream=False
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
        if opt.model.training_type == "ff":
            num_layers = opt.model.depth * 2 + 1
            wandb_dict1 = {"Loss": scalar_outputs["Loss"],
                           "Peer_Norm": scalar_outputs["Peer_Norm"],
                           "cls_acc": scalar_outputs["cls_acc"]}
            wandb_dict2 = {f"loss_layer_{i}": scalar_outputs[f"loss_layer_{i}"] for i in range(num_layers)}
            wandb_dict3 = {f"acc_layer_{i}": scalar_outputs[f"acc_layer_{i}"] for i in range(num_layers)}
            wandb_train_dict = dict(ChainMap(wandb_dict1, wandb_dict2, wandb_dict3))
            
            if opt.training.test_mode == "one_pass_softmax":
                # "one_pass_softmax" 测试模式下, 存在 cls_loss 与 cls_acc 两个 metric.
                wandb_train_dict["cls_loss"] = scalar_outputs["cls_loss"]            
                if partition == "train":
                    wandb.log(wandb_train_dict)
                elif partition == "val":
                    wandb.log({ "Val Loss": scalar_outputs["Loss"],
                                "Val cls_loss": scalar_outputs["cls_loss"],
                                "Val cls_acc": scalar_outputs["cls_acc"] })
                elif partition == "test":
                    wandb.log({ "Test Loss": scalar_outputs["Loss"],
                                "Test cls_loss": scalar_outputs["cls_loss"],
                                "Test cls_acc": scalar_outputs["cls_acc"] })

            elif opt.training.test_mode == "compute_each_label":
                # "compute_each_label" 测试模式下, 只存在 cls_acc 这个 metric.
                if partition == "train":
                    wandb.log(wandb_train_dict)
                elif partition == "val":
                    wandb.log({ "Val cls_acc": scalar_outputs["cls_acc"] })
                elif partition == "test":
                    wandb.log({ "Test cls_acc": scalar_outputs["cls_acc"] })

        elif opt.model.training_type == "bp":
            if partition == "train":
                wandb.log({ "Loss": scalar_outputs["Loss"],
                            "cls_acc": scalar_outputs["cls_acc"] })
            elif partition == "val":
                wandb.log({ "Val Loss": scalar_outputs["Loss"],
                            "Val cls_acc": scalar_outputs["cls_acc"] })
            elif partition == "test":
                wandb.log({ "Test Loss": scalar_outputs["Loss"],
                            "Test cls_acc": scalar_outputs["cls_acc"] })

        else:
            raise NotImplementedError

    print()


def log_results(result_dict, scalar_outputs, num_steps):
    for key, value in scalar_outputs.items():
        if isinstance(value, float):
            result_dict[key] += value / num_steps
        else:
            result_dict[key] += value.item() / num_steps
    return result_dict


def overlay_label_on_z(num_classes, z, label):
    '''注意: 在卷积网络中, z 为四维 tensor: B*C*H*W;
    把 z 的每行前 10 列元素都替换成 label 所指示的那个标签.'''
    z_labeled = z.clone().reshape(z.shape[0], -1)   # 若 z 本身就是二维, 该操作无影响.

    # 下面 z_labeled 已经变成二维了.
    # 必须把 index tensor 放到 gpu 上, 否则它与 z_labeled 就不在同一设备上.
    index = torch.arange(num_classes).cuda()
    z_labeled.index_fill_(1, index, 0)
    z_labeled[:, label] = 1

    return z_labeled.reshape(z.shape)   # modif dqr: 返回的仍是一个四维 tensor.


def create_mask(shape, iterations: int = 10):
    """
    Create a binary mask as described in (Hinton, 2022): start with a random binary image and then repeatedly blur
    the image with a filter, horizontally and vertically.
    Parameters
    ----------
    shape : tuple
        The shape of the output mask (height, width).
    iterations : int
        The number of times to blur the image.
    Returns
    -------
    numpy.ndarray
        A binary mask with the specified shape, containing fairly large regions of ones and zeros.
    """

    blur_filter_1 = np.array(((0, 0, 0), (0.25, 0.5, 0.25), (0, 0, 0)))
    blur_filter_2 = blur_filter_1.T

    # Create a random binary image
    image = np.random.randint(0, 2, size=shape)

    # Blur the image with the specified filter
    for i in range(iterations):
        image = np.abs(convolve2d(image, blur_filter_1, mode='same') / blur_filter_1.sum())
        image = np.abs(convolve2d(image, blur_filter_2, mode='same') / blur_filter_2.sum())

    # Binarize the blurred image, i.e. threshold it at 0.5
    mask = np.round(image).astype(np.uint8)

    return tensor(mask)


def create_negative_image(image_1, image_2):
    """
    适用于 C*H*W 形状的 image 输入的, 多 channel negative image 生成器.
    Create a negative image by combining two images with a binary mask.
    Parameters:
    image_1 (Tensor): The first image to be combined.
    image_2 (Tensor): The second image to be combined.
    Returns:
    Tensor: The negative image created by combining the two input images.
    Raises:
    AssertionError: If the shapes of `image_1` and `image_2` are not the same.
    Examples:
    >>> image_1 = np.random.randint(0, 2, size=(5, 5))
    >>> image_2 = np.random.randint(0, 2, size=(5, 5))
    >>> create_negative_image(image_1, image_2)
    array([[0 0 0 0 1]
           [1 1 0 1 1]
           [0 0 0 1 1]
           [0 1 1 1 0]
           [1 1 0 0 1]])
    """

    assert image_1.shape == image_2.shape, "Incompatible images and mask shapes."

    mask = create_mask((image_1.shape[1], image_1.shape[2]))

    image_1 = torch.mul(image_1, mask)
    image_2 = torch.mul(image_2, 1 - mask)

    return torch.add(image_1, image_2)


def create_negative_batch(images):
    neg_imgs = []
    batch_size = images.shape[0]
    for _ in range(batch_size):
        idx1, idx2 = np.random.randint(batch_size, size=2)
        neg_imgs.append(create_negative_image(images[idx1].squeeze(), images[idx2].squeeze()))
    return torch.unsqueeze(torch.stack(neg_imgs), dim=1)


def negative_dataset_transform(dataset, alpha=1):
    '''生成 negative dataset, 但可能重复使用用同一组 pair 的图片(不过 mask 一定是不同的)
    alpha: 倍数, 决定生成的负样本数量是提供的 dataset 的多少倍.'''
    random_pairs = np.random.randint(len(dataset), size=(alpha * len(dataset), 2))
    random_pairs = [(row[0], row[1]) for row in random_pairs]
    transformed_dataset = [
        create_negative_image(dataset[pair[0]][0].squeeze(), dataset[pair[1]][0].squeeze())
        for pair in random_pairs]
    return transformed_dataset