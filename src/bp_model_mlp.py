import math

import torch
import torch.nn as nn

import wandb

from src import utils


class BP_model_mlp(torch.nn.Module):
    """The Multi-layer perceptron trained with Back Propagation (BP)."""

    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.input_dim = opt.model.input_dim
        self.num_channels = [self.opt.model.hidden_dim] * self.opt.model.num_layers
        self.output_dim = opt.input.num_classes
        # self.num_channels 只包含中间层 hidden_dims, 不包括 input_dim 与 output_dim.
        self.act_fn = ReLU_full_grad()
        
        # 初始化网络结构: self.model 最终为 [3072, 3072, 3072, 3072, 10]
        self.model = nn.ModuleList([nn.Linear(self.input_dim, self.num_channels[0])])
        for i in range(1, len(self.num_channels)):
            self.model.append(nn.Linear(self.num_channels[i - 1], self.num_channels[i]))
        # 最后一层线性分类器:[3072, 10], 并且参数保留 bias.
        self.model.append(nn.Linear(self.num_channels[-1], self.output_dim))
        
        # 定义线性分类器所用的 loss 函数.
        self.bp_cls_loss = nn.CrossEntropyLoss()
        
        # 自定义权重初始化方法.
        self._init_weights()

    def _init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                # nn.init.normal_(
                #     m.weight, mean=0, std=1 / math.sqrt(m.weight.shape[1])
                # )
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu'
                )
                nn.init.zeros_(m.bias)

    def forward(self, inputs, labels):
        '''输入 inputs 与 labels 两个 tensor, 经过 3 层 MLP 和 1 层线性分类器, 
        输出分类结果, 并与 labels 比较, 计算 bp_cls_loss/acc, 最后返回输出 dict. '''
        scalar_outputs = {
            "Loss": torch.zeros(1, device=self.opt.device),
        }

        # 输入的 inputs 是4维 tensor, 需要先 flatten 为 2维 tensor.
        z = inputs.reshape(inputs.shape[0], -1)

        for idx, layer in enumerate(self.model):
            z = layer(z)
            z = self.act_fn.apply(z)

        # z 已经变成一个 logits, 其中每个 sample 的长度均为 10.
        output = z - torch.max(z, dim=-1, keepdim=True)[0]
        bp_cls_loss = self.bp_cls_loss(output, labels)
        bp_cls_acc = utils.get_accuracy(
            self.opt, output.data, labels
        )

        scalar_outputs["Loss"] = bp_cls_loss
        scalar_outputs["classification_accuracy"] = bp_cls_acc

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

