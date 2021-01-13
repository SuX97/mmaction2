import os
import os.path as osp

import numpy as np
from matplotlib import pyplot as plt


def draw(direct):
    # tem_results 文件所在上级目录  xxx.csv
    tem_results = osp.join(direct, 'tem_results')
    figure_dir = osp.join(direct, 'tem_figure')
    if not osp.exists(figure_dir):
        os.makedirs(figure_dir)
    files = [osp.join(tem_results, file)
             for file in os.listdir(tem_results)][:100]
    for file in files:
        result = np.loadtxt(file, dtype=np.float32, delimiter=',', skiprows=1)
        action, start, end = result[:, 0], result[:, 1], result[:, 2]

        if 'activitynet' in direct:
            action_file = osp.join(
                figure_dir,
                osp.splitext(osp.basename(file))[0] + '_action.png')
            plt.figure()
            plt.plot(np.array(range(len(action))), action)
            plt.savefig(action_file)

            start_file = osp.join(
                figure_dir,
                osp.splitext(osp.basename(file))[0] + '_start.png')
            plt.figure()
            plt.plot(np.array(range(len(start))), start)
            plt.savefig(start_file)

            end_file = osp.join(
                figure_dir,
                osp.splitext(osp.basename(file))[0] + '_end.png')
            plt.figure()
            plt.plot(np.array(range(len(end))), end)
            plt.savefig(end_file)
        else:
            action_file = osp.join(
                figure_dir,
                osp.splitext(osp.basename(file))[0] + '_action_100.png')
            plt.figure()
            plt.plot(np.array(range(len(action[:100]))), action[:100])
            plt.savefig(action_file)

            start_file = osp.join(
                figure_dir,
                osp.splitext(osp.basename(file))[0] + '_start_100.png')
            plt.figure()
            plt.plot(np.array(range(len(start[:100]))), start[:100])
            plt.savefig(start_file)

            end_file = osp.join(
                figure_dir,
                osp.splitext(osp.basename(file))[0] + '_end_100.png')
            plt.figure()
            plt.plot(np.array(range(len(end[:100]))), end[:100])
            plt.savefig(end_file)


if __name__ == '__main__':
    # draw('work_dirs/bsn_2000x4096_8x5_trunet_feature')
    # draw('work_dirs/bsn_400x100_32x3_20e_activitynet_feature')
    draw('work_dirs/bsn_2000x4096_32x3_70e_trunet_feature')
