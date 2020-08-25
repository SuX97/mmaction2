import os.path as osp

import torch
from mmcv.utils import print_log

from ..core import mean_class_accuracy, top_k_accuracy, precision_recall
from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module()
class VideoDataset(BaseDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt

        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2
        some/path/003.mp4 2
        some/path/004.mp4 3
        some/path/005.mp4 3

    """

    def load_annotations(self):
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split(' ')
                if self.multi_class:
                    assert self.num_classes is not None
                    # filename, label = line_split[0], line_split[1:]
                    # label = list(map(int, label))
                    # onehot = torch.zeros(self.num_classes)
                    # onehot[label] = 1.0
                    filename, label = ' '.join(line_split[:-1]), line_split[-1].split(',')
                    label = list(map(int, label))
                    onehot = torch.zeros(self.num_classes,dtype=torch.float)#,dtype=torch.long
                    onehot[label] = 1.0
                else:
                    filename, label = ' '.join(line_split[:-1]), line_split[-1].split(',')#, line_split[-1]
                    label = int(label)
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                video_infos.append(
                    dict(
                        filename=filename,
                        label=onehot if self.multi_class else label))
        return video_infos


    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 topk=(1, 5),
                 logger=None):
        """Evaluation in rawframe dataset.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            logger (obj): Training logger. Defaults: None.
            topk (tuple[int]): K value for top_k_accuracy metric.
                Defaults: (1, 5).
            logger (logging.Logger | None): Logger for recording.
                Default: None.

        Return:
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

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['top_k_accuracy', 'mean_class_accuracy', 'precision_recall']
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

            if metric == 'top_k_accuracy':
                top_k_acc = top_k_accuracy(results, gt_labels, topk)
                log_msg = []
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'top{k}_acc'] = acc
                    log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric == 'precision_recall':
                precision, recall = precision_recall(results, gt_labels)
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

            if metric == 'mean_class_accuracy':
                mean_acc = mean_class_accuracy(results, gt_labels)
                eval_results['mean_class_accuracy'] = mean_acc
                log_msg = f'\nmean_acc\t{mean_acc:.4f}'
                print_log(log_msg, logger=logger)
                continue

        return eval_results
