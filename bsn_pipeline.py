import numpy as np
import os


# def train_bsn_anet():
#     tem_cmd = "CUDA_VISIBLE_DEVICES=4,5,6 PORT=29501 " \
#               "bash tools/dist_train.sh " \
#               "configs/localization/bsn/bsn_tem_400x100_32x3_20e_activitynet_feature.py 3"
#     os.system(tem_cmd)
#
#     os.system('mv work_dirs/bsn_400x100_32x3_20e_activitynet_feature/latest.pth '
#               'work_dirs/bsn_400x100_32x3_20e_activitynet_feature/tem_latest.pth')
#
#     tem_generate = "CUDA_VISIBLE_DEVICES=4 PORT=29502 " \
#                    "bash tools/dist_test.sh " \
#                    "configs/localization/bsn/bsn_tem_400x100_32x3_20e_activitynet_feature.py " \
#                    "work_dirs/bsn_400x100_32x3_20e_activitynet_feature/tem_latest.pth 1"
#     os.system(tem_generate)
#
#     pgm_cmd = "python tools/bsn_proposal_generation.py " \
#               "configs/localization/bsn/bsn_pgm_400x100_32x3_20e_activitynet_feature.py " \
#               "--mode train"
#     os.system(pgm_cmd)
#
#     pem_cmd = "CUDA_VISIBLE_DEVICES=4,5,6 PORT=29503 " \
#               "bash tools/dist_train.sh " \
#               "configs/localization/bsn/bsn_pem_400x100_32x3_20e_activitynet_feature.py 3"
#     os.system(pem_cmd)


def train_bsn():
    # tem_config: ann_file_val=ann_file_train, ann_file_test=ann_file_train
    # tem_config: data_root_val=data_root
    CUDA = ['7', '8', '9']
    shape = '2000x4096'
    batch = 32
    gpus = 3
    epoch = 70
    dataset = 'trunet'
    port = 29510

    tem_cmd = f"CUDA_VISIBLE_DEVICES={','.join(CUDA)} PORT=29501 " \
              "bash tools/dist_train.sh " \
              f"configs/localization/bsn/bsn_tem_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature.py {len(CUDA)}"
    os.system(tem_cmd)

    os.system(f'mv work_dirs/bsn_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature/latest.pth '
              f'work_dirs/bsn_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature/tem_latest.pth')

    tem_generate = f"CUDA_VISIBLE_DEVICES={CUDA[0]} PORT={port} " \
                   "bash tools/dist_test.sh " \
                   f"configs/localization/bsn/bsn_tem_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature.py " \
                   f"work_dirs/bsn_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature/tem_latest.pth 1"
    os.system(tem_generate)

    pgm_cmd = "python tools/bsn_proposal_generation.py " \
              f"configs/localization/bsn/bsn_pgm_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature.py " \
              "--mode train"
    os.system(pgm_cmd)

    pem_cmd = f"CUDA_VISIBLE_DEVICES={','.join(CUDA)} PORT={port} " \
              "bash tools/dist_train.sh " \
              f"configs/localization/bsn/bsn_pem_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature.py {len(CUDA)}"
    os.system(pem_cmd)

    os.system(f'mv work_dirs/bsn_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature/latest.pth '
              f'work_dirs/bsn_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature/pem_latest.pth')


def evaluate_bsn():
    # tem_config: ann_file_val='val_meta.json', ann_file_test='val_meta.json'
    # tem_config: data_root_val=val split directory
    CUDA = ['4', '5', '6']
    shape = '400x100'
    batch = 32
    gpus = 3
    epoch = 20
    dataset = 'activitynet'
    port = 29510

    tem_generate = f"CUDA_VISIBLE_DEVICES={CUDA[0]} PORT={port} " \
                   "bash tools/dist_test.sh " \
                   f"configs/localization/bsn/bsn_tem_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature.py " \
                   f"work_dirs/bsn_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature/tem_latest.pth 1"
    os.system(tem_generate)

    pgm_cmd = "python tools/bsn_proposal_generation.py " \
              f"configs/localization/bsn/bsn_pgm_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature.py " \
              "--mode test"
    os.system(pgm_cmd)

    pem_generate = f"CUDA_VISIBLE_DEVICES={CUDA[0]} PORT={port} " \
                   "bash tools/dist_test.sh " \
                   f"configs/localization/bsn/bsn_pem_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature.py " \
                   f"work_dirs/bsn_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature/pem_latest.pth 1"
    os.system(pem_generate)

    eval_cmd = f"python tools/analysis/report_map.py " if dataset == 'activitynet' \
               else f"python tools/analysis/report_trunet_map.py "
    eval_cmd += f"--proposal work_dirs/bsn_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature/results.json "
    if dataset == 'activitynet':
        eval_cmd += "--gt data/ActivityNet/anet_anno_val.json " \
                    f"--det-output work_dirs/bsn_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature/det_results.json "
    else:
        eval_cmd += "--gt data/TruNet/val_meta.json "


if __name__ == '__main__':
    evaluate_bsn()
