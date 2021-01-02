# -*- coding: utf-8 -*-
import os


def inference():
    # inference train split(already tem results and pgm results)
    '''train_pem_cmd = "GPUS=1 GPUS_PER_NODE=1 bash tools/slurm_test.sh " \
                    "ha_vug pem_train configs/localization/bsn/bsn_pem_2000x4096_8x24_70e_trunet_feature.py " \
                    "work_dirs/bsn_pem_2000x4096_8x24_70e_trunet_feature/latest.pth " \
                    "--eval AR@AN " \
                    "--cfg-options dist_params.port=29510 " \
                    "test_pipeline.pgm_proposals_dir=" \
                    "'work_dirs/bsn_pgm_2000x4096_trunet_feature/pgm_train_proposals/' " \
                    "test_pipeline.pgm_features_dir=" \
                    "'work_dirs/bsn_pgm_2000x4096_trunet_feature/pgm_train_features/' " \
                    "val_pipeline.pgm_proposals_dir=" \
                    "'work_dirs/bsn_pgm_2000x4096_trunet_feature/pgm_train_proposals/' " \
                    "val_pipeline.pgm_features_dir=" \
                    "'work_dirs/bsn_pgm_2000x4096_trunet_feature/pgm_train_features/' " \
                    "output_config.out='work_dirs/bsn_pem_2000x4096_8x24_70e_trunet_feature/train_results.json'"
    print(train_pem_cmd)
    os.system(train_pem_cmd)'''

    # inference test split
    val_tem_cmd = "GPUS=1 GPUS_PER_NODE=1 bash tools/slurm_test.sh " \
                  "ha_vug tem_val configs/localization/bsn/bsn_tem_2000x4096_8x24_70e_trunet_feature.py " \
                  "work_dirs/bsn_tem_2000x4096_8x24_70e_trunet_feature/latest.pth " \
                  "--cfg-options dist_params.port=29510 " \
                  "output_config.out='work_dirs/bsn_tem_2000x4096_8x24_70e_trunet_feature/tem_val_results/'"
    print(val_tem_cmd)
    os.system(val_tem_cmd)

    val_pgm_cmd = "srun -p ha_vug --job-name=pgm_val " \
                  "python tools/bsn_proposal_generation.py " \
                  "configs/localization/bsn/bsn_pgm_2000x4096_trunet_feature.py " \
                  "--mode test"
    print(val_pgm_cmd)
    os.system(val_pgm_cmd)

    '''val_pem_cmd = "GPUS=1 GPUS_PER_NODE=1 bash tools/slurm_test.sh " \
                  "ha_vug pem_val configs/localization/bsn/bsn_pem_2000x4096_8x24_70e_trunet_feature.py " \
                  "work_dirs/bsn_pem_2000x4096_8x24_70e_trunet_feature/latest.pth " \
                  "--eval AR@AN " \
                  "--cfg-options dist_params.port=29510 " \
                  "test_pipeline.pgm_proposals_dir=" \
                  "'work_dirs/bsn_pgm_2000x4096_trunet_feature/pgm_val_proposals/' " \
                  "test_pipeline.pgm_features_dir=" \
                  "'work_dirs/bsn_pgm_2000x4096_trunet_feature/pgm_val_features/' " \
                  "val_pipeline.pgm_proposals_dir=" \
                  "'work_dirs/bsn_pgm_2000x4096_trunet_feature/pgm_val_proposals/' " \
                  "val_pipeline.pgm_features_dir=" \
                  "'work_dirs/bsn_pgm_2000x4096_trunet_feature/pgm_val_features/' " \
                  "output_config.out='work_dirs/bsn_pem_2000x4096_8x24_70e_trunet_feature/val_results.json'"
    print(val_pem_cmd)
    os.system(val_pem_cmd)'''


if __name__ == '__main__':
    inference()
