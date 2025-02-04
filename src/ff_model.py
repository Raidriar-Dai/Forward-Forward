import math

import torch
import torch.nn as nn

from src import utils


class FF_model(torch.nn.Module):
    """The model trained with Forward-Forward (FF)."""

    def __init__(self, opt):
        super(FF_model, self).__init__()

        self.opt = opt
        self.num_channels = [self.opt.model.hidden_dim] * self.opt.model.num_layers
        self.act_fn = ReLU_full_grad()

        # Initialize the model.
        self.model = nn.ModuleList([nn.Linear(784, self.num_channels[0])])
        for i in range(1, len(self.num_channels)):
            self.model.append(nn.Linear(self.num_channels[i - 1], self.num_channels[i]))

        # Initialize forward-forward loss.
        self.ff_loss = nn.BCEWithLogitsLoss()

        # Initialize peer normalization loss.
        self.running_means = [
            torch.zeros(self.num_channels[i], device=self.opt.device) + 0.5
            for i in range(self.opt.model.num_layers)
        ]

        # Initialize downstream classification loss.
        # 这里应该是实现原论文中 "one-pass" softmax 的 test 方法.
        channels_for_classification_loss = sum(
            self.num_channels[-i] for i in range(self.opt.model.num_layers - 1)
        )   # 纠错: 是否该改成 self.num_channels[-i-1] ?
        # 下游的线性分类器 并不被包括在 self.model 中, 而是单独列为 self.linear_classifier.
        self.linear_classifier = nn.Sequential(
            nn.Linear(channels_for_classification_loss, 10, bias=False)
        )
        self.classification_loss = nn.CrossEntropyLoss()

        # Initialize weights.
        self._init_weights()

    def _init_weights(self):
        # 纠错: 是否该改为 model.children() ?
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                # 用了正态分布来初始化 weight_matrix, 而没有用 nn.Linear 默认的均匀分布.
                # weight_matrix 的形状为: (out_features,in_features), 这里的 std 用的是 out_features 来初始化.
                torch.nn.init.normal_(
                    m.weight, mean=0, std=1 / math.sqrt(m.weight.shape[0])
                )
                torch.nn.init.zeros_(m.bias)

        for m in self.linear_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)

    def _layer_norm(self, z, eps=1e-8):
        # dim=-1 指最后一个 dim, 应该总归指一个 sample 中的多个分量所在的维度.
        # 纠错: 为什么这里是 mean 而不是 sum ?
        return z / (torch.sqrt(torch.mean(z ** 2, dim=-1, keepdim=True)) + eps)

    def _calc_peer_normalization_loss(self, idx, z):
        # Only calculate mean activity over positive samples.
        mean_activity = torch.mean(z[:self.opt.input.batch_size], dim=0)

        self.running_means[idx] = self.running_means[
            idx
        ].detach() * self.opt.model.momentum + mean_activity * (
            1 - self.opt.model.momentum
        )

        peer_loss = (torch.mean(self.running_means[idx]) - self.running_means[idx]) ** 2
        return torch.mean(peer_loss)

    def _calc_ff_loss(self, z, labels):
        sum_of_squares = torch.sum(z ** 2, dim=-1)

        # 这里的 z.shape[1] 即为原论文 2.0 节所写公式中的 theta.
        logits = sum_of_squares - z.shape[1]
        ff_loss = self.ff_loss(logits, labels.float())

        with torch.no_grad():
            ff_accuracy = (
                torch.sum((torch.sigmoid(logits) > 0.5) == labels)  # equivalent to `logits > 0`
                / z.shape[0]
            ).item()
        return ff_loss, ff_accuracy

    def forward(self, inputs, labels):
        '''输入一个 batch 的 samples, 经过3层线性层后, 再经过线性分类器, 最后返回输出 dict.'''
        scalar_outputs = {
            "Loss": torch.zeros(1, device=self.opt.device),
            "Peer Normalization": torch.zeros(1, device=self.opt.device),
        }

        # Concatenate positive and negative samples and create corresponding labels.
        # 沿着 batch 维度把 pos_tensors 与 neg_tensors 拼在一起构成 z, 并创建对应的 labels.
        # 其中 pos data 的 label 值为 1, neg data 的 label 值为 0.
        z = torch.cat([inputs["pos_images"], inputs["neg_images"]], dim=0) 
        posneg_labels = torch.zeros(z.shape[0], device=self.opt.device)
        posneg_labels[: self.opt.input.batch_size] = 1

        z = z.reshape(z.shape[0], -1)

        # 从下面开始 z 就是被 flatten 成为一个二维 tensor, 分为 batch 和 length 两个维度.
        # z 在被送入第一层之前, 就已经被 normalize 了.
        z = self._layer_norm(z)

        for idx, layer in enumerate(self.model):
            z = layer(z)
            z = self.act_fn.apply(z)

            # 可选项: 是否把 peer loss 这个正则因子加入最终的 Loss 中.
            if self.opt.model.peer_normalization > 0:
                peer_loss = self._calc_peer_normalization_loss(idx, z)
                scalar_outputs["Peer Normalization"] += peer_loss
                scalar_outputs["Loss"] += self.opt.model.peer_normalization * peer_loss

            ff_loss, ff_accuracy = self._calc_ff_loss(z, posneg_labels)
            scalar_outputs[f"loss_layer_{idx}"] = ff_loss
            scalar_outputs[f"ff_accuracy_layer_{idx}"] = ff_accuracy
            scalar_outputs["Loss"] += ff_loss
            z = z.detach()  # 特别注意: 把 z 送入下一层 layer 的 forward 之前, 必须 detach 掉 z 在上一层 layer 中的计算图.

            z = self._layer_norm(z)

        scalar_outputs = self.forward_downstream_classification_model(
            inputs, labels, scalar_outputs=scalar_outputs
        )

        return scalar_outputs

    def forward_downstream_classification_model(
        self, inputs, labels, scalar_outputs=None,
    ):
        '''输入一个 batch 的 samples, 通过 ff_layers 后把 activity vector 送入线性分类器,
        计算分类误差 和 分类精度, 最终返回更新后的 scalar_outputs.'''
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
            }

        z = inputs["neutral_sample"]
        z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z)

        input_classification_model = []

        with torch.no_grad():
            for idx, layer in enumerate(self.model):
                z = layer(z)
                z = self.act_fn.apply(z)
                z = self._layer_norm(z)

                # 收集从第2层 hidden layer 开始的 activity vector.
                # 注意: 选取的都是归一化后的 activity vector.
                if idx >= 1:
                    input_classification_model.append(z)

        input_classification_model = torch.concat(input_classification_model, dim=-1)

        # 合并后的 activity vector 在输入线性分类器之前, 先要与它在 ff_layers 中的计算图解绑.
        # 这可能是为了, 最终对 classification_loss 求导时, 求到 input_classification_model 就停止,
        # 以免继续往前回溯计算图.
        output = self.linear_classifier(input_classification_model.detach())
        output = output - torch.max(output, dim=-1, keepdim=True)[0] # torch.max 的返回值是一个 
        # namedtuple (values, indices), 因此需要再取一次 [0]; 这就相当于 output 每一行减去该行最大值.
        classification_loss = self.classification_loss(output, labels["class_labels"])
        classification_accuracy = utils.get_accuracy(
            self.opt, output.data, labels["class_labels"]
        )

        scalar_outputs["Loss"] += classification_loss
        scalar_outputs["classification_loss"] = classification_loss
        scalar_outputs["classification_accuracy"] = classification_accuracy
        return scalar_outputs


class ReLU_full_grad(torch.autograd.Function):
    """ ReLU activation function that passes through the gradient irrespective of its input value. """
    # 是否意味着 x < 0 时同样有 ReLU(x) = x, 而不会有 "死亡 ReLU" 产生?

    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()
