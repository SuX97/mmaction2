# -*- coding: utf-8 -*-
import os


def inference():
    tem_cmd = "GPUS=1 GPUS_PER_NODE=1 bash tools/slurm_test.sh " \
              "ha_vug tem_trunet " \
              "configs/localization/bsn/bsn_tem_2000x4096_8x24_70e_trunet_feature.py " \
              "work_dirs/bsn_2000x4096_8x24_70e_trunet_feature/tem_latest.pth " \
              "--cfg-options dist_params.port=29510"
    os.system(tem_cmd)

    pgm_cmd = "srun -p ha_vug " \
              "python tools/bsn_proposal_generation.py " \
              "configs/localization/bsn/bsn_pgm_2000x4096_trunet_feature.py " \
              "--mode test"
    os.system(pgm_cmd)

    pem_cmd = "GPUS=1 GPUS_PER_NODE=1 bash tools/slurm_test.sh " \
              "ha_vug pem_trunet " \
              "configs/localization/bsn/bsn_pem_2000x4096_8x24_70e_trunet_feature.py " \
              "work_dirs/bsn_2000x4096_8x24_70e_trunet_feature/pem_latest.pth " \
              "--eval AR@AN " \
              "--cfg-options dist_params.port=29511"
    os.system(pem_cmd)

    map_cmd = "srun -p ha_vug " \
              "python tools/analysis/report_trunet_map.py " \
              "--proposal work_dirs/bsn_2000x4096_8x24_70e_trunet_feature/results.json"
    os.system(map_cmd)


if __name__ == '__main__':
    inference()
