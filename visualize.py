import os
import os.path as osp

import numpy as np
from matplotlib import pyplot as plt


def draw(direct):
    # tem_results 文件所在上级目录  xxx.csv
    tem_results = osp.join(direct, 'tem_results')
    figure_dir = osp.join(direct, 'tem_figure')
    files = [osp.join(tem_results, file)
             for file in os.listdir(tem_results)][:100]
    for file in files:
        result = np.loadtxt(file, dtype=np.float32, delimiter=',', skiprows=1)
        action, start, end = result[:, 0], result[:, 1], result[:, 2]

        action_file = osp.join(
            figure_dir,
            osp.splitext(osp.basename(file))[0] + '_action.png')
        plt.figure()
        plt.plot(np.array(range(2000)), action)
        plt.savefig(action_file)

        start_file = osp.join(
            figure_dir,
            osp.splitext(osp.basename(file))[0] + '_start.png')
        plt.figure()
        plt.plot(np.array(range(2000)), start)
        plt.savefig(start_file)

        end_file = osp.join(figure_dir,
                            osp.splitext(osp.basename(file))[0] + '_end.png')
        plt.figure()
        plt.plot(np.array(range(2000)), end)
        plt.savefig(end_file)


if __name__ == '__main__':
    draw('work_dirs/bsn_2000x4096_8x5_trunet_feature')
    draw('work_dirs/bsn_400x100_32x3_20e_activitynet_feature')
