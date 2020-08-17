import torch
import torch.nn as nn
from mmcv.cnn import NORM_LAYERS


@NORM_LAYERS.register_module()
class SubBatchBN3d(nn.Module):
    """SubBatchBN3d that splits a norm batchnorm3d into a 'split_bn' and a
    normal 'bn'. This is used when the batchsize is changed in runtime.

    Args:
        num_features (int): The number of channels of the input feature.
        **cfg (dict): Other kwargs required for a batchnorm layer.
    """

    def __init__(self, num_features, **cfg):
        super(SubBatchBN3d, self).__init__()
        self.num_features = num_features
        self.cfg_ = cfg.copy()
        if 'num_splits' in cfg:
            self.num_splits = self.cfg_.pop('num_splits')
        else:
            self.num_splits = 1
        self.bn = nn.BatchNorm3d(num_features, **self.cfg_)
        self.num_features_split = self.num_features * self.num_splits
        self.split_bn = nn.BatchNorm3d(self.num_features_split, **self.cfg_)
        self.init_weights(cfg)

    def init_weights(self, cfg):
        """Initialize the weight of the module.

        Only keep one set of weight and bias for affine after normalization.
        """
        if cfg.get('affine', True):
            self.affine = True
            cfg['affine'] = False
            self.weight = torch.nn.Parameter(torch.ones(self.num_features))
            self.bias = torch.nn.Parameter(torch.zeros(self.num_features))
        else:
            self.affine = False

    def _get_aggregated_mean_std(self, means, stds, n):
        """Calculate the aggregated mean and stds.

        Args:
            means (torch.Tensor): mean values.
            stds (torch.Tensor): standard deviations.
            n (int): number of sets of means and stds.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (mean, std). mean is the
                aggregated mean of all split batchnorms. std is the
                aggregated std of all split batchnorms.
        """
        mean = means.view(n, -1).sum(0) / n
        std = (
            stds.view(n, -1).sum(0) / n +
            ((means.view(n, -1) - mean)**2).view(n, -1).sum(0) / n)
        return mean.detach(), std.detach()

    def aggregate_stats(self):
        """Synchronize running_mean, and running_var.

        Call this before eval.
        """
        if self.split_bn.track_running_stats:
            (self.bn.running_mean.data,
             self.bn.running_var.data) = self._get_aggregated_mean_std(
                 self.split_bn.running_mean, self.split_bn.running_var,
                 self.num_splits)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            tuple[torch.Tensor]: The normalized feature map.
        """
        if self.training:
            n, c, t, h, w = x.shape
            x = x.view(n // self.num_splits, c * self.num_splits, t, h, w)
            x = self.split_bn(x)
            x = x.view(n, c, t, h, w)
        else:
            x = self.bn(x)
        if self.affine:
            x = x * self.weight.view((-1, 1, 1, 1))
            x = x + self.bias.view((-1, 1, 1, 1))
        return x
