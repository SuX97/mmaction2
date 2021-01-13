import numpy as np
import os


def train_bsn():
    tem_cmd = "CUDA_VISIBLE_DEVICES=4,5,6 PORT=29501 " \
              "bash tools/dist_train.sh " \
              "configs/localization/bsn/bsn_tem_400x100_32x3_20e_activitynet_feature.py 3"
    os.system(tem_cmd)

    os.system('mv work_dirs/bsn_400x100_32x3_20e_activitynet_feature/latest.pth '
              'work_dirs/bsn_400x100_32x3_20e_activitynet_feature/tem_latest.pth')

    tem_generate = "CUDA_VISIBLE_DEVICES=4 PORT=29502 " \
                   "bash tools/dist_test.sh " \
                   "configs/localization/bsn/bsn_tem_400x100_32x3_20e_activitynet_feature.py " \
                   "work_dirs/bsn_400x100_32x3_20e_activitynet_feature/tem_latest.pth 1"
    os.system(tem_generate)

    pgm_cmd = "python tools/bsn_proposal_generation.py " \
              "configs/localization/bsn/bsn_pgm_400x100_32x3_20e_activitynet_feature.py " \
              "--mode train"
    os.system(pgm_cmd)

    pem_cmd = "CUDA_VISIBLE_DEVICES=4,5,6 PORT=29503 " \
              "bash tools/dist_train.sh " \
              "configs/localization/bsn/bsn_pem_400x100_32x3_20e_activitynet_feature.py 3"
    os.system(pem_cmd)


if __name__ == '__main__':
    train_bsn()
