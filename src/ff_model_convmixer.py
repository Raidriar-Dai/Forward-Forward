import math
import torch
import torch.nn as nn
# from torchvision.ops import Permute

import wandb

from src import utils


class Residual(nn.Module):
    '''输入某种 layer <fn>, 输出 <fn> 的具有 residual connection 的形式.'''
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class Permute(nn.Module):
    """This module returns a view of the tensor input with its dimensions permuted.

    Args:
        dims (List[int]): The desired ordering of dimensions
    """

    def __init__(self, dims: list[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)


class FF_model_convmixer(torch.nn.Module):
    '''基于 convmixer 结构实现的 ffa 模型'''
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        act_fn_dict = {
            'relu': nn.ReLU,
            'relu_full_grad': ReLu_full_grad_module,
            'gelu': nn.GELU
        }

        # 专门列出 convmixer 的 model hyperparameters.
        self.dim = self.opt.model.dim
        self.depth = self.opt.model.depth
        self.kernel_size = self.opt.model.kernel_size
        self.patch_size = self.opt.model.patch_size
        self.num_classes = self.opt.input.num_classes
        self.avgpool_size = self.opt.model.avgpool_size

        self.act_fn = act_fn_dict[self.opt.model.act_fn]
        self.norm_layer = self.opt.model.norm_layer
        self.have_residual = self.opt.model.have_residual

        # 以 sequential 作为每一层 layer 的单元结构.
        self.patch_embedding_block = [
            nn.Sequential(
                nn.Conv2d(3, self.dim, kernel_size=self.patch_size, stride=self.patch_size),
                self.act_fn(),
                nn.BatchNorm2d(self.dim) if self.norm_layer == 'batchnorm' else
                nn.Sequential(
                    Permute((0,2,3,1)),
                    nn.LayerNorm(self.dim),
                    Permute((0,3,1,2))
                ) if self.norm_layer == 'layernorm' else
                nn.Identity()
            )
        ]

        # 给每一个 conv_mixer_block 都套上 Residual, 获得残差输出.
        if self.have_residual:
            self.conv_mixer_blocks = sum([
                [
                    # Depthwise convolution
                    Residual(nn.Sequential(
                        nn.Conv2d(self.dim, self.dim, self.kernel_size, groups=self.dim, padding="same"),
                        self.act_fn(),
                        nn.BatchNorm2d(self.dim) if self.norm_layer == 'batchnorm' else
                        nn.Sequential(
                            Permute((0,2,3,1)),
                            nn.LayerNorm(self.dim),
                            Permute((0,3,1,2))
                        ) if self.norm_layer == 'layernorm' else
                        nn.Identity()
                    )),
                    # Pointwise convolution(这一层 Residual 是 optional 的)
                    nn.Sequential(
                        nn.Conv2d(self.dim, self.dim, kernel_size=1),
                        self.act_fn(),
                        nn.BatchNorm2d(self.dim) if self.norm_layer == 'batchnorm' else
                        nn.Sequential(
                            Permute((0,2,3,1)),
                            nn.LayerNorm(self.dim),
                            Permute((0,3,1,2))
                        ) if self.norm_layer == 'layernorm' else
                        nn.Identity()
                    )
                ] for i in range(self.depth)
            ], start=[])

        else:
            self.conv_mixer_blocks = sum([
                [
                    # Depthwise convolution(默认没有 Residual Connection)
                    nn.Sequential(
                        nn.Conv2d(self.dim, self.dim, self.kernel_size, groups=self.dim, padding="same"),
                        self.act_fn(),
                        nn.BatchNorm2d(self.dim) if self.norm_layer == 'batchnorm' else
                        nn.Sequential(
                            Permute((0,2,3,1)),
                            nn.LayerNorm(self.dim),
                            Permute((0,3,1,2))
                        ) if self.norm_layer == 'layernorm' else
                        nn.Identity()
                    ),
                    # Pointwise convolution
                    nn.Sequential(
                        nn.Conv2d(self.dim, self.dim, kernel_size=1),
                        self.act_fn(),
                        nn.BatchNorm2d(self.dim) if self.norm_layer == 'batchnorm' else
                        nn.Sequential(
                            Permute((0,2,3,1)),
                            nn.LayerNorm(self.dim),
                            Permute((0,3,1,2))
                        ) if self.norm_layer == 'layernorm' else
                        nn.Identity()
                    )
                ] for i in range(self.depth)
            ], start=[])

        # self.patch_embedding_block = [
        #     nn.Sequential(
        #         nn.Conv2d(3, self.dim, kernel_size=self.patch_size, stride=self.patch_size),
        #         self.act_fn(),
        #     )
        # ]

        # self.conv_mixer_blocks = sum([
        #     [
        #         # Depthwise convolution(默认没有 Residual Connection)
        #         nn.Sequential(
        #             nn.Conv2d(self.dim, self.dim, self.kernel_size, groups=self.dim, padding="same"),
        #             self.act_fn(),
        #         ),
        #         # Pointwise convolution
        #         nn.Sequential(
        #             nn.Conv2d(self.dim, self.dim, kernel_size=1),
        #             self.act_fn(),
        #         )
        #     ] for i in range(self.depth)
        # ], start=[])

        self.model = nn.ModuleList(
            sum([
                self.patch_embedding_block,
                self.conv_mixer_blocks
            ], start=[])
        )

        # 定义 ff_loss, 后面也有另一自定义版本;
        self.ff_loss = nn.BCEWithLogitsLoss()

        # 只对于 one-pass softmax 实现 ffa.
        if self.opt.training.test_mode == "one_pass_softmax":
            channels_for_classification_loss = 2 * self.dim * (self.avgpool_size ** 2) * self.depth    # 默认为 2*256*(1*1)*12 = 3072*2
            self.linear_classifier = nn.Sequential(
                nn.Linear(channels_for_classification_loss, 10, bias=True)
            )
            self.classification_loss = nn.CrossEntropyLoss()

        # 用自定义函数来初始化权重;
        self._init_weights()

    # 自定义 ff_loss, 相比默认的 BCE Loss, 这个有上下界, 更稳定;
    # def ff_loss(self, logit, label):
    #         logit_sig = torch.sigmoid(logit)
    #         return torch.norm(logit_sig - label)

    def _init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                # 1. 自定义正态初始化
                # torch.nn.init.normal_(
                #     m.weight, mean=0, std=1 / math.sqrt(m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3])
                # )
                # torch.nn.init.zeros_(m.bias)

                # 2. Kaiming 初始化
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu'
                )
                nn.init.zeros_(m.bias)

        # 线性分类头的初始化采用默认方法, 故注释掉下面的全零初始化.
        # if self.opt.training.test_mode == "one_pass_softmax":
        #     for m in self.linear_classifier.modules():
        #         if isinstance(m, nn.Linear):
        #             # 1. 全零初始化
        #             nn.init.zeros_(m.weight)

        #             # 2. 自定义正态初始化
        #             # nn.init.normal_(
        #             #     m.weight, mean=0, std=1 / math.sqrt(m.weight.shape[1])
        #             # )

    def _layer_norm(self, z, eps=1e-8):
        '''输入 z 为 B*C*H*W 四维的. 把 4层channels 分为一组做 layer_norm, 
        这与后面 4层channels 为一组计算 ff_loss 相吻合.'''
        s = z.shape
        t = torch.reshape(z,(s[0], self.opt.model.num_groups_each_layer, -1))

        # 现在, t 的形状为 200 * 64 * 1024
        t = t / (torch.sqrt(torch.mean(t ** 2, dim=-1, keepdim=True)) + eps)
        return torch.reshape(t,s)


    def _layer_norm_all_units(self, z, eps=1e-8):
        '''输入 z 为 B*C*H*W 四维的. 对每一层所有 units 一起做 layer_norm.'''
        s = z.shape
        t = torch.reshape(z,(s[0], -1))

        # 现在, t 的形状为 200*3072(仅针对第一层: patch embedding)
        t = t / (torch.sqrt(torch.mean(t ** 2, dim=-1, keepdim=True)) + eps)
        return torch.reshape(t,s)


    def _calc_ff_loss(self, x, labels):
        '''输入 x 为 B*C*H*W 四维的. 用 squared_sum 与 squared_mean 两种不同方式生成 logits.'''
        if self.opt.model.goodness_type == "sum":
            # squared_sum: 每 1024 个 units(相当于 4 个 channels 为一组) 算一个平方和, 
            # 再对 64 组来取平均.
            z = torch.reshape(x, (x.shape[0], self.opt.model.num_groups_each_layer, -1))
            sum_of_squares = torch.mean(torch.sum(z ** 2, dim=-1), dim=-1)
            logits = sum_of_squares - z.shape[-1]    # 这里的 z.shape[-1] 一般为 1024, 即为 theta.

        elif self.opt.model.goodness_type == "mean":
            # squared_mean: 简单地, 对每一层 256*16*16 个 units 一起计算平均值.
            z = torch.reshape(x, (x.shape[0], -1))
            mean_of_squares = torch.mean(z ** 2, dim=-1)
            logits = mean_of_squares - self.opt.model.theta

        else:
            raise NotImplementedError

        ff_loss = self.ff_loss(logits, labels.float())

        with torch.no_grad():
            ff_accuracy = (
                torch.sum((torch.sigmoid(logits) > 0.5) == labels)  # equivalent to `logits > 0`
                / z.shape[0]
            ).item()
        return ff_loss, ff_accuracy


    def _calc_ff_loss_all_units(self, x, labels):
        '''输入 x 为 B*C*H*W 四维的. 用该层所有 units 的平方和来计算 ff_loss. 不再分组.'''
        z = torch.reshape(x,(x.shape[0],-1))

        if self.opt.model.goodness_type == "sum":
            sum_of_squares = torch.sum(z ** 2, dim=-1)
            logits = sum_of_squares - z.shape[1]

        elif self.opt.model.goodness_type == "mean":
            mean_of_squares = torch.mean(z ** 2, dim=-1)
            logits = mean_of_squares - self.opt.model.theta

        else:
            raise NotImplementedError

        ff_loss = self.ff_loss(logits, labels.float())

        with torch.no_grad():
            ff_accuracy = (
                torch.sum((torch.sigmoid(logits) > 0.5) == labels)  # equivalent to `logits > 0`
                / z.shape[0]
            ).item()
        return ff_loss, ff_accuracy


    def forward(self, inputs, labels):
        '''1 * patch_embedding layer + 12 * (Depthwise Conv + Pointwise Conv)
        最后的线性分类头: (256*12*2, 10); 没有 compute_each_label 模式. 没有 PeerNorm.'''
        scalar_outputs = {
            "Loss": torch.zeros(1, device=self.opt.device),
            "Peer_Norm": torch.zeros(1, device=self.opt.device)
        }

        z = torch.cat([inputs["pos_images"], inputs["neg_images"]], dim=0) 
        posneg_labels = torch.zeros(z.shape[0], device=self.opt.device)
        posneg_labels[: self.opt.input.batch_size] = 1
        
        # 在卷积网络 forward 过程中, z 始终是一个 B*C*H*W 的四维 tensor.
        # Flatten(Reshape) 只会发生在 utils 函数内部.
        # z 在被送入第一层之前形状为 200*3*32*32, 用的是 all units 的 layernorm.
        z = self._layer_norm_all_units(z)
        
        for idx, layer in enumerate(self.model):
            # 这里的每层 layer 都是 nn.Sequential; 其中包含了卷积 + 激活函数 + batchnorm;
            z = layer(z)
            
            ff_loss, ff_accuracy = self._calc_ff_loss(z, posneg_labels)
            
            scalar_outputs[f"loss_layer_{idx}"] = ff_loss
            scalar_outputs[f"acc_layer_{idx}"] = ff_accuracy
            scalar_outputs["Loss"] += ff_loss
            
            z = z.detach()
            z = self._layer_norm(z) # 从第 2 层开始, 就要对 channels 分组进行 layernorm.

        if self.opt.training.test_mode == "one_pass_softmax":
            scalar_outputs = self.forward_downstream_classification_model(
                inputs, labels, scalar_outputs=scalar_outputs
            )
        else:
            raise NotImplementedError
        
        return scalar_outputs


    def forward_downstream_classification_model(
        self, inputs, labels, scalar_outputs=None
    ):
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
            }

        z = inputs["neutral_sample"]
        z = self._layer_norm_all_units(z)

        input_classification_model = []

        with torch.no_grad():
            for idx, layer in enumerate(self.model):
                z = layer(z)
                z = self._layer_norm(z)

                if idx >= 1:
                    # layer_features 大小应为 100*256*1*1, 其中 100 为 batch_size
                    layer_features = nn.functional.adaptive_avg_pool2d(
                        z, (self.avgpool_size, self.avgpool_size)
                        )
                    input_classification_model.append(torch.reshape(layer_features, (layer_features.shape[0], -1)))

        input_classification_model = torch.concat(input_classification_model, dim=-1)
        output = self.linear_classifier(input_classification_model.detach())
        output = output - torch.max(output, dim=-1, keepdim=True)[0]
        classification_loss = self.classification_loss(output, labels["class_labels"])
        classification_accuracy = utils.get_accuracy(
            self.opt, output.data, labels["class_labels"]
        )

        scalar_outputs["Loss"] += classification_loss
        scalar_outputs["cls_loss"] = classification_loss
        scalar_outputs["cls_acc"] = classification_accuracy
        return scalar_outputs


class ReLU_full_grad(torch.autograd.Function):
    """ ReLU activation function that passes through the gradient irrespective of its input value. """
    # forward 和正常 ReLU 定义一样, 
    # 但 backward 就直接把回传的 grad_output 原封不动输出, 不再判别 forward 时候的 output 是否 <=0.

    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class ReLu_full_grad_module(torch.nn.Module):
    '''用 ReLU_full_grad 函数实现的 module, 用于 self.models.'''
    def __init__(self):
        super().__init__()
        self.act_fn = ReLU_full_grad()

    def forward(self, x):
        return self.act_fn.apply(x)


class Residual(nn.Module):
    '''输入某种 layer <fn>, 输出 <fn> 的具有 residual connection 的形式.'''
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x