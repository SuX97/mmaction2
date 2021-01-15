import argparse
import os


def train_bsn():
    # tem_config: ann_file_val = ann_file_train, ann_file_test = ann_file_train
    # tem_config: data_root_val=data_root
    tem_cmd = f'GPUS={GPUS} GPUS_PER_NODE={GPUS_PER_NODE} ' \
              f'bash tools/slurm_train.sh {part} {job_name} ' \
              'configs/localization/bsn/' \
              f'bsn_tem_{shape}_{batch}x{GPUS}_{epoch}e_{dataset}_feature.py'\
              f' --cfg-options dist_params.port={port}'
    os.system(tem_cmd)

    os.system('mv work_dirs/'
              f'bsn_{shape}_{batch}x{GPUS}_{epoch}e_{dataset}_feature/'
              'latest.pth '
              'work_dirs/'
              f'bsn_{shape}_{batch}x{GPUS}_{epoch}e_{dataset}_feature/'
              'tem_latest.pth')

    tem_gen = f'GPUS={GPUS} GPUS_PER_NODE={GPUS_PER_NODE} ' \
              f'bash tools/slurm_test.sh {part} {job_name} ' \
              'configs/localization/bsn/' \
              f'bsn_tem_{shape}_{batch}x{GPUS}_{epoch}e_{dataset}_feature.py'\
              f' work_dirs/ ' \
              f'bsn_{shape}_{batch}x{GPUS}_{epoch}e_{dataset}_feature/' \
              f'tem_latest.pth ' \
              f'--cfg-options dist_params.port={port}'
    os.system(tem_gen)

    pgm_cmd = 'python tools/bsn_proposal_generation.py ' \
              'configs/localization/bsn/' \
              f'bsn_pgm_{shape}_{batch}x{GPUS}_{epoch}e_{dataset}_feature.py' \
              ' --mode train'
    os.system(pgm_cmd)

    pem_cmd = f'GPUS={GPUS} GPUS_PER_NODE={GPUS_PER_NODE} ' \
              f'bash tools/slurm_train.sh {part} {job_name} ' \
              'configs/localization/bsn/' \
              f'bsn_pem_{shape}_{batch}x{GPUS}_{epoch}e_{dataset}_feature.py' \
              f' --cfg-options dist_params.port={port}'
    os.system(pem_cmd)

    os.system('mv work_dirs/'
              f'bsn_{shape}_{batch}x{GPUS}_{epoch}e_{dataset}_feature/'
              'latest.pth '
              'work_dirs/'
              f'bsn_{shape}_{batch}x{GPUS}_{epoch}e_{dataset}_feature/'
              'pem_latest.pth')


def evaluate_bsn():
    # tem_config: ann_file_val='val_meta.json', ann_file_test='val_meta.json'
    # tem_config: data_root_val=val split directory
    tem_gen = f'GPUS={GPUS} GPUS_PER_NODE={GPUS_PER_NODE} ' \
              f'bash tools/slurm_test.sh {part} {job_name} ' \
              'configs/localization/bsn/' \
              f'bsn_tem_{shape}_{batch}x{GPUS}_{epoch}e_{dataset}_feature.py' \
              f' work_dirs/ ' \
              f'bsn_{shape}_{batch}x{GPUS}_{epoch}e_{dataset}_feature/' \
              f'tem_latest.pth ' \
              f'--cfg-options dist_params.port={port}'
    os.system(tem_gen)

    pgm_cmd = 'python tools/bsn_proposal_generation.py ' \
              'configs/localization/bsn/' \
              f'bsn_pgm_{shape}_{batch}x{GPUS}_{epoch}e_{dataset}_feature.py'\
              ' --mode test'
    os.system(pgm_cmd)

    pem_gen = f'GPUS={GPUS} GPUS_PER_NODE={GPUS_PER_NODE} ' \
              f'bash tools/slurm_test.sh {part} {job_name} ' \
              'configs/localization/bsn/' \
              f'bsn_pem_{shape}_{batch}x{GPUS}_{epoch}e_{dataset}_feature.py' \
              f' work_dirs/ ' \
              f'bsn_{shape}_{batch}x{GPUS}_{epoch}e_{dataset}_feature/' \
              f'pem_latest.pth ' \
              f'--cfg-options dist_params.port={port}'
    os.system(pem_gen)

    eval_cmd = 'python tools/analysis/report_map.py ' \
        if dataset == 'activitynet' \
        else 'python tools/analysis/report_trunet_map.py '
    eval_cmd += '--proposal work_dirs/' \
                f'bsn_{shape}_{batch}x{GPUS}_{epoch}e_{dataset}_feature/' \
                'results.json '
    if dataset == 'activitynet':
        eval_cmd += '--gt data/ActivityNet/anet_anno_val.json ' \
                    '--det-output work_dirs/' \
                    f'bsn_{shape}_{batch}x{GPUS}_{epoch}e_{dataset}_feature/' \
                    'det_results.json '
    else:
        eval_cmd += '--gt data/TruNet/val_meta.json '
    os.system(eval_cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPUS', type=int)
    parser.add_argument('--GPUS_PER_NODE', type=int)
    parser.add_argument('--part', type=str)
    parser.add_argument('--job-name', type=int)
    parser.add_argument('--port', type=int)
    parser.add_argument('--shape', type=str)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    GPUS, GPUS_PER_NODE, part, job_name, port = (args.GPUS, args.GPUS_PER_NODE,
                                                 args.part, args.job_name,
                                                 args.port)
    shape, batch, epoch, dataset, train = (args.shape, args.batch, args.epoch,
                                           args.dataset, args.train)
    if train:
        train_bsn()
    else:
        evaluate_bsn()
