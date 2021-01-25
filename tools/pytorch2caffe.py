import argparse

import mmcv
import spring.nart.tools.pytorch as pytorch
import torch
from mmcv.runner import load_checkpoint

from mmaction.models import build_model


def _convert_batchnorm(module):
    """Convert the syncBNs into normal BN3ds."""
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm3d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def pytorch2caffe(model,
                  input_shape,
                  output_file='testfile',
                  input_names=['data'],
                  output_names=['outs']):
    print('Start!')
    with pytorch.convert_mode():
        pytorch.convert_v2(
            model, [tuple(input_shape)],
            output_file,
            input_names=input_names,
            output_names=output_names)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMAction2 models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--is-localizer',
        action='store_true',
        help='whether it is a localizer')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1, 3, 8, 224, 224],
        help='input video size')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # import modules from string list.

    if not args.is_localizer:
        cfg.model.backbone.pretrained = None

    # build the model
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    model = _convert_batchnorm(model)

    # onnx.export does not support kwargs
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    elif hasattr(model, '_forward') and args.is_localizer:
        model.forward = model._forward
    else:
        raise NotImplementedError(
            'Please implement the forward method for exporting.')
    print('loading...')
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    print('loaded')
    # conver model to onnx file
    pytorch2caffe(model, args.shape)
