import copy
import os.path as osp

import numpy as np
import torch
from mmcv.utils import print_log

from ..core import mean_average_precision, mean_class_accuracy#, top_k_accuracy
from ..core.evaluation.accuracy import *
from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module()
class RawframeDataset(BaseDataset):
    """Rawframe dataset for action recognition.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, total frames of the video and
    the label of a video, which are split with a whitespace.
    Example of a annotation file:

    .. code-block:: txt

        some/directory-1 163 1
        some/directory-2 122 1
        some/directory-3 258 2
        some/directory-4 234 2
        some/directory-5 295 3
        some/directory-6 121 3

    Example of a multi-class annotation file:


    .. code-block:: txt

        some/directory-1 163 1 3 5
        some/directory-2 122 1 2
        some/directory-3 258 2
        some/directory-4 234 2 4 6 8
        some/directory-5 295 3
        some/directory-6 121 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Default: False.
        num_classes (int): Number of classes in the dataset. Default: None.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl='{:06}.jpg',
                 multi_class=True,
                 num_classes=400):
        super().__init__(ann_file, pipeline, data_prefix, test_mode,
                         multi_class, num_classes)
        self.filename_tmpl = filename_tmpl

    def load_annotations(self):
        video_infos = []
        label_correlations = np.loadtxt('./annos/normed_coherence.txt')
        with open(self.ann_file, 'r') as fin:
            # cnt = 1
            for line in fin:
                line_split = line.strip().split(' ')
                if self.multi_class:
                    assert self.num_classes is not None
                    # (frame_dir, total_frames,
                    #  label) = (line_split[0], line_split[1], line_split[2:])
                    frame_dir, total_frames, label = ' '.join(line_split[:-2]), line_split[-2], line_split[-1].split(',')
                    label = list(map(int, label))
                    # onehot = torch.zeros(self.num_classes, dtype=torch.float)#,dtype=torch.long
                    # flag = [True if l > 45 else False for l in label]\
                    # cnt += 1
                    # soft_labels = False
                    soft_labels = True
                    if not self.test_mode and soft_labels:
                        correlation = np.average(label_correlations[label, :], axis=0)
                        # print(correlation.shape)
                        onehot = torch.tensor(correlation, dtype=torch.float)
                    else:
                        onehot = torch.zeros(self.num_classes, dtype=torch.float)
                    onehot[label] = 1
                    # print(label)
                    # print('*'*30)
                    # print(label_correlations[label, :])
                    # print('*'*30)
                    # print(correlation)
                    # print('*'*30)
                    # exit()
                else:
                    # frame_dir, total_frames, label = ' '.join(line_split[:-2]), line_split[-2], line_split[-1]#.split(',')
                    # label = int(label[0])
                    frame_dir, total_frames, label = ' '.join(line_split[:-2]), line_split[-2], line_split[-1].split(',')
                    label = int(label)
                if self.data_prefix is not None:
                    frame_dir = osp.join(self.data_prefix, frame_dir)
                video_infos.append(
                    dict(
                        frame_dir=frame_dir,
                        total_frames=int(total_frames),
                        label=onehot if self.multi_class else label))
        return video_infos

    def prepare_train_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metrics='personal_PR',
                 topk=(1, 5),
                 logger=None):
        """Evaluation in rawframe dataset.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            logger (obj): Training logger. Defaults: None.
            topk (int | tuple[int]): K value for top_k_accuracy metric.
                Defaults: (1, 5).
            logger (logging.Logger | None): Logger for recording.
                Default: None.

        return:
            eval_results (dict): Evaluation results dict.
        """
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        if not isinstance(topk, (int, tuple)):
            raise TypeError(
                f'topk must be int or tuple of int, but got {type(topk)}')

        if isinstance(topk, int):
            topk = (topk, )

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        # allowed_metrics = [
        #      'personal_PR'#'top_k_accuracy',  'mean_class_accuracy', 'mean_average_precision',
        # ]
        allowed_metrics = [
             'precision_recall'#'top_k_accuracy',  'mean_class_accuracy', 'mean_average_precision',
        ]
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = {}
        gt_labels = [ann['label'] for ann in self.video_infos]

        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            # if metric == 'top_k_accuracy':
            #     top_k_acc = top_k_accuracy(results, gt_labels, topk)
            #     log_msg = []
            #     for k, acc in zip(topk, top_k_acc):
            #         eval_results[f'top{k}_acc'] = acc
            #         log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
            #     log_msg = ''.join(log_msg)
            #     print_log(log_msg, logger=logger)
            #     continue

            if metric == 'mean_class_accuracy':
                mean_acc = mean_class_accuracy(results, gt_labels)
                eval_results['mean_class_accuracy'] = mean_acc
                log_msg = f'\nmean_acc\t{mean_acc:.4f}'
                print_log(log_msg, logger=logger)
                continue

            if metric == 'precision_recall':
                # print(len(results))
                # print(len(gt_labels))
                # exit()
                precision, recall = precision_recall(results, gt_labels)
                # print('*'*30)
                # print(P)
                # print('*'*30)
                # exit()
                eval_results['precision'] = precision
                eval_results['recall'] = recall
                log_msg = f'precision\t{precision}'
                print_log(log_msg, logger=logger)
                log_msg = f'recall\t{recall}'
                print_log(log_msg, logger=logger)
                # num_classes = 46-14
                # sum_precision = np.sum(np.nan_to_num(precision))
                # log_msg = f'mean precision\t{}'
                # print_log(log_msg, logger=logger)
                # sum_recall = np.sum(np.nan_to_num(recall))
                # log_msg = f'mean recall\t{sum_recall//num_classes}'
                # print_log(log_msg, logger=logger)
                continue

            if metric == 'mean_average_precision':
                gt_labels = [label.cpu().numpy() for label in gt_labels]
                mAP = mean_average_precision(results, gt_labels)
                eval_results['mean_average_precision'] = mAP
                log_msg = f'\nmean_average_precision\t{mAP:.4f}'
                print_log(log_msg, logger=logger)
                continue

        return eval_results


