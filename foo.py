# -*- coding: utf-8 -*-
import os.path as osp
import numpy as np
import argparse
import json


def statistic():
    # 统计proposals在给定的阈值下有多少个正负样本
    # 分训练集和测试集
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, help="train_meta.json annotation path")
    parser.add_argument("--val", type=str, help="val_meta.json annotation path")
    parser.add_argument("--pgm_proposal", type=str, help="pgm_proposal path")
    parser.add_argument("--iou", type=float, help="iou threshold")
    args = parser.parse_args()
    train_meta, val_meta, pgm_proposal, iou = args.train, args.val, args.pgm_proposal, args.iou
    with open(train_meta, 'r', encoding='utf-8') as f:
        train = json.load(f)
    train_files = [osp.join(pgm_proposal, key + '.csv') for key in train.keys()]
    train_pos, train_neg, train_num = 0, 0, len(train_files)
    for train_file in train_files:
        proposals = np.loadtxt(train_file, dtype=np.float32, delimiter=',', skiprows=1)
        match_iou = proposals[:, 5]
        train_pos += np.sum(match_iou > iou)
        train_neg += np.sum(match_iou <= iou)
    with open(val_meta, 'r', encoding='utf-8') as f:
        val = json.load(f)
    val_files = [osp.join(pgm_proposal, key + '.csv') for key in val.keys()]
    val_pos, val_neg, val_num = 0, 0, len(val_files)
    for val_file in val_files:
        proposals = np.loadtxt(val_file, dtype=np.float32, delimiter=',', skiprows=1)
        match_iou = proposals[:, 5]
        val_pos += np.sum(match_iou > iou)
        val_neg += np.sum(match_iou <= iou)

    print(f'train positive per video: {train_pos / train_num}, ratio: {train_pos / (train_pos + train_neg)}')
    print(f'train negative per video: {train_neg / train_num}, ratio: {train_neg / (train_pos + train_neg)}')
    print(f'val positive per video: {val_pos / val_num}, ratio: {val_pos / (val_pos + val_neg)}')
    print(f'val negative per video: {val_neg / val_num}, ratio: {val_neg / (val_pos + val_neg)}')


if __name__ == '__main__':
    statistic()
