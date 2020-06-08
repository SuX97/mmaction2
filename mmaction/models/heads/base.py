from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from ...core import top_k_accuracy
from ..builder import build_loss


class AvgConsensus(nn.Module):
    """Average consensus module.

    Args:
        dim (int): Decide which dim consensus function to apply.
            Default: 1.
    """

    def __init__(self, dim=1):
        super(AvgConsensus, self).__init__()
        self.dim = dim

    def forward(self, input):
        return input.mean(dim=self.dim, keepdim=True)


class BaseHead(nn.Module, metaclass=ABCMeta):
    """Base class for head.

    All Head should subclass it.
    All subclass should overwrite:
        Methods:`init_weights`, initializing weights in some modules.
        Methods:`forward`, supporting to forward both for training and testing.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        multi_class (bool): Determines whether it is a multi-class
            recognition task. Default: False.
        label_smooth_eps (float): Epsilon used in label smooth.
            Reference: arxiv.org/abs/1906.02629. Default: 0.
        loss_factor (float): Factor scalar multiplied on the loss.
            Default: 1.0.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss', loss_factor=1.0),
                 multi_class=False,
                 label_smooth_eps=0.0):
        super(BaseHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss_cls = build_loss(loss_cls)
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps

    @abstractmethod
    def init_weights(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    def loss(self, cls_score, labels):
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)

        if not self.multi_class:
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(), (1, 5))
            losses['top1_acc'] = torch.tensor(top_k_acc[0])
            losses['top5_acc'] = torch.tensor(top_k_acc[1])

        elif self.label_smooth_eps != 0:
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        losses['loss_cls'] = self.loss_cls(cls_score, labels)
        return losses