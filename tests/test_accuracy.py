import random

import numpy as np
import pytest

from mmaction.core.evaluation import (confusion_matrix, mean_average_precision,
                                      mean_class_accuracy, top_k_accuracy)


def gt_confusion_matrix(gt_labels, pred_labels):
    """Calculate the ground truth confusion matrix"""
    max_index = max(max(gt_labels), max(pred_labels))
    confusion_mat = np.zeros((max_index + 1, max_index + 1), dtype=np.int64)
    for gt, pred in zip(gt_labels, pred_labels):
        confusion_mat[gt][pred] += 1
    del_index = []
    for i in range(max_index):
        if sum(confusion_mat[i]) == 0 and sum(confusion_mat[:, i]) == 0:
            del_index.append(i)
    confusion_mat = np.delete(confusion_mat, del_index, axis=0)
    confusion_mat = np.delete(confusion_mat, del_index, axis=1)
    return confusion_mat


def test_confusion_matrix():
    # custom confusion_matrix
    gt_labels = [np.int64(random.randint(0, 9)) for _ in range(100)]
    pred_labels = np.random.randint(10, size=100, dtype=np.int64)
    confusion_mat = confusion_matrix(pred_labels, gt_labels)
    gt_confusion_mat = gt_confusion_matrix(gt_labels, pred_labels)
    assert np.array_equal(confusion_mat, gt_confusion_mat)

    with pytest.raises(TypeError):
        # y_pred must be list or np.ndarray
        confusion_matrix(0.5, [1])

    with pytest.raises(TypeError):
        # y_real must be list or np.ndarray
        confusion_matrix([1], 0.5)

    with pytest.raises(TypeError):
        # y_pred dtype must be np.int64
        confusion_matrix([0.5], [1])

    with pytest.raises(TypeError):
        # y_real dtype must be np.int64
        confusion_matrix([1], [0.5])


def test_topk():
    scores = [
        np.array([-0.2203, -0.7538, 1.8789, 0.4451, -0.2526]),
        np.array([-0.0413, 0.6366, 1.1155, 0.3484, 0.0395]),
        np.array([0.0365, 0.5158, 1.1067, -0.9276, -0.2124]),
        np.array([0.6232, 0.9912, -0.8562, 0.0148, 1.6413])
    ]

    # top1 acc
    k = (1, )
    top1_labels_0 = [3, 1, 1, 1]
    top1_labels_25 = [2, 0, 4, 3]
    top1_labels_50 = [2, 2, 3, 1]
    top1_labels_75 = [2, 2, 2, 3]
    top1_labels_100 = [2, 2, 2, 4]
    res = top_k_accuracy(scores, top1_labels_0, k)
    assert res == [0]
    res = top_k_accuracy(scores, top1_labels_25, k)
    assert res == [0.25]
    res = top_k_accuracy(scores, top1_labels_50, k)
    assert res == [0.5]
    res = top_k_accuracy(scores, top1_labels_75, k)
    assert res == [0.75]
    res = top_k_accuracy(scores, top1_labels_100, k)
    assert res == [1.0]

    # top1 acc, top2 acc
    k = (1, 2)
    top2_labels_0_100 = [3, 1, 1, 1]
    top2_labels_25_75 = [3, 1, 2, 3]
    res = top_k_accuracy(scores, top2_labels_0_100, k)
    assert res == [0, 1.0]
    res = top_k_accuracy(scores, top2_labels_25_75, k)
    assert res == [0.25, 0.75]

    # top1 acc, top3 acc, top5 acc
    k = (1, 3, 5)
    top5_labels_0_0_100 = [1, 0, 3, 2]
    top5_labels_0_50_100 = [1, 3, 4, 0]
    top5_labels_25_75_100 = [2, 3, 0, 2]
    res = top_k_accuracy(scores, top5_labels_0_0_100, k)
    assert res == [0, 0, 1.0]
    res = top_k_accuracy(scores, top5_labels_0_50_100, k)
    assert res == [0, 0.5, 1.0]
    res = top_k_accuracy(scores, top5_labels_25_75_100, k)
    assert res == [0.25, 0.75, 1.0]


def test_mean_class_accuracy():
    scores = [
        np.array([-0.2203, -0.7538, 1.8789, 0.4451, -0.2526]),
        np.array([-0.0413, 0.6366, 1.1155, 0.3484, 0.0395]),
        np.array([0.0365, 0.5158, 1.1067, -0.9276, -0.2124]),
        np.array([0.6232, 0.9912, -0.8562, 0.0148, 1.6413])
    ]

    # test mean class accuracy in [0, 0.25, 1/3, 0.75, 1.0]
    mean_cls_acc_0 = np.int64([1, 4, 0, 2])
    mean_cls_acc_25 = np.int64([2, 0, 4, 3])
    mean_cls_acc_33 = np.int64([2, 2, 2, 3])
    mean_cls_acc_75 = np.int64([4, 2, 2, 4])
    mean_cls_acc_100 = np.int64([2, 2, 2, 4])
    assert mean_class_accuracy(scores, mean_cls_acc_0) == 0
    assert mean_class_accuracy(scores, mean_cls_acc_25) == 0.25
    assert mean_class_accuracy(scores, mean_cls_acc_33) == 1 / 3
    assert mean_class_accuracy(scores, mean_cls_acc_75) == 0.75
    assert mean_class_accuracy(scores, mean_cls_acc_100) == 1.0


def test_mean_average_precision():
    # One sample
    y_true = [np.array([0, 0, 1, 1])]
    y_scores = [np.array([0.1, 0.4, 0.35, 0.8])]
    map = mean_average_precision(y_scores, y_true)

    precision = [2.0 / 3.0, 0.5, 1., 1.]
    recall = [1., 0.5, 0.5, 0.]
    target = -np.sum(np.diff(recall) * np.array(precision)[:-1])
    assert target == map