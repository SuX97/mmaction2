import os
import random
from collections import OrderedDict

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, Runner, build_optimizer

from ..core import (DistEvalHook, DistOptimizerHook, EvalHook,
                    Fp16OptimizerHook, MultiGridHook)
from ..datasets import build_dataloader, build_dataset
from ..utils import get_root_logger


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_losses(losses):
    """Parse losses dict for different loss variants.

    Args:
        losses (dict): Loss dict.

    Returns:
        loss (float): Sum of the total loss.
        log_vars (dict): Loss dict for different variants.
    """
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def batch_processor(model, data, train_mode):
    """Batch processor function for runner.

    Args:
        model (nn.Module): The model to be trained.
        data (dict): Input data for training model.
        train_mode (bool): Store True when training model.

    Returns:
        dict: Output for training model.
    """
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss,
        log_vars=log_vars,
        num_samples=len(next(iter(data.values()))))

    return outputs


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (Dataset): Train dataset.
        cfg (dict): The config dict for training.
        distributed (bool): Whether to use distributed training.
            Default: False.
        validate (bool): Whether to do evaluation. Default: False.
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """
    logger = get_root_logger(log_level=cfg.log_level)

    # start training
    if distributed:
        _dist_train(
            model,
            dataset,
            cfg,
            validate=validate,
            logger=logger,
            timestamp=timestamp,
            meta=meta)
    else:
        _non_dist_train(
            model,
            dataset,
            cfg,
            validate=validate,
            logger=logger,
            timestamp=timestamp,
            meta=meta)


def _dist_train(model,
                dataset,
                cfg,
                validate=False,
                logger=None,
                timestamp=None,
                meta=None):
    """Distributed training function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (Dataset): Train dataset.
        cfg (dict): The config dict for training.
        validate (bool): Whether to do evaluation. Default: False.
        logger (logging.Logger | None): Logger for training. Default: None.
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None.
    """
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', {}),
        workers_per_gpu=cfg.data.get('workers_per_gpu', {}),
        dist=True,
        seed=cfg.seed)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('train_dataloader', {}))
    data_loaders = [
        build_dataloader(ds, **dataloader_setting) for ds in dataset
    ]
    # put model on gpus
    find_unused_parameters = cfg.get('find_unused_parameters', False)
    # Sets the `find_unused_parameters` parameter in
    # torch.nn.parallel.DistributedDataParallel
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(
        model,
        batch_processor,
        optimizer,
        cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config,
                                             **fp16_cfg)
    else:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())

    if validate:
        eval_cfg = cfg.get('evaluation', {})
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', {}),
            workers_per_gpu=cfg.data.get('workers_per_gpu', {}),
            num_gpus=cfg.gpus,
            dist=True,
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('val_dataloader', {}))
        val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
        runner.register_hook(DistEvalHook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model,
                    dataset,
                    cfg,
                    validate=False,
                    logger=None,
                    timestamp=None,
                    meta=None):
    """Non-Distributed training function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (Dataset): Train dataset.
        cfg (dict): The config dict for training.
        validate (bool): Whether to do evaluation. Default: False.
        logger (logging.Logger | None): Logger for training. Default: None.
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None.
    """
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', {}),
        workers_per_gpu=cfg.data.get('workers_per_gpu', {}),
        num_gpus=cfg.gpus,
        dist=False,
        seed=cfg.seed)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('train_dataloader', {}))
    data_loaders = [
        build_dataloader(ds, **dataloader_setting) for ds in dataset
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(
        model,
        batch_processor,
        optimizer,
        cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=False)
    else:
        optimizer_config = cfg.optimizer_config

    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    # multigrid setting
    multi_grid_cfg = cfg.get('multi_grid', None)
    if multi_grid_cfg is not None:
        multi_grid_scheduler = MultiGridHook(**cfg.multi_grid_cfg, **cfg.data)
        runner.register_hook(multi_grid_scheduler)

    # validation setting
    if validate:
        eval_cfg = cfg.get('evaluation', {})
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', {}),
            workers_per_gpu=cfg.data.get('workers_per_gpu', {}),
            num_gpus=cfg.gpus,
            dist=False,
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('val_dataloader', {}))
        val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
        runner.register_hook(EvalHook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
