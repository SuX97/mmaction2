import argparse
import os


def train_bsn():
    # tem_config: ann_file_val=ann_file_train, ann_file_test=ann_file_train
    # tem_config: data_root_val=data_root
    tem_cmd = f"CUDA_VISIBLE_DEVICES={','.join(CUDA)} PORT=29501 " \
              'bash tools/dist_train.sh ' \
              'configs/localization/bsn/' \
              f'bsn_tem_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature.py'\
              f' {len(CUDA)}'

    mv_tem = 'mv work_dirs/' \
             f'bsn_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature/' \
             'latest.pth ' \
             'work_dirs/' \
             f'bsn_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature/' \
             'tem_latest.pth'

    tem_gen = f'CUDA_VISIBLE_DEVICES={CUDA[0]} PORT={port} ' \
              'bash tools/dist_test.sh ' \
              'configs/localization/bsn/' \
              f'bsn_tem_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature.py'\
              ' work_dirs/' \
              f'bsn_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature/' \
              'tem_latest.pth 1'

    pgm_cmd = 'python tools/bsn_proposal_generation.py ' \
              'configs/localization/bsn/' \
              f'bsn_pgm_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature.py'\
              ' --mode train'

    pem_cmd = f"CUDA_VISIBLE_DEVICES={','.join(CUDA)} PORT={port} " \
              'bash tools/dist_train.sh ' \
              'configs/localization/bsn/' \
              f'bsn_pem_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature.py'\
              f' {len(CUDA)}'

    mv_pem = 'mv work_dirs/' \
             f'bsn_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature/' \
             'latest.pth ' \
             'work_dirs/' \
             f'bsn_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature/' \
             'pem_latest.pth'

    flag = False
    if tem_train or flag:
        flag = True
        status = os.system(tem_cmd)
        if status != 0:
            return
        os.system(mv_tem)
    if tem or flag:
        flag = True
        status = os.system(tem_gen)
        if status != 0:
            return
    if pgm or flag:
        flag = True
        status = os.system(pgm_cmd)
        if status != 0:
            return
    if pem or flag:
        status = os.system(pem_cmd)
        if status != 0:
            return
        os.system(mv_pem)


def evaluate_bsn():
    # tem_config: ann_file_val='val_meta.json', ann_file_test='val_meta.json'
    # tem_config: data_root_val=val split directory
    tem_gen = f'CUDA_VISIBLE_DEVICES={CUDA[0]} PORT={port} ' \
              'bash tools/dist_test.sh ' \
              'configs/localization/bsn/' \
              f'bsn_tem_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature.py '\
              'work_dirs/' \
              f'bsn_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature/' \
              'tem_latest.pth 1'

    pgm_cmd = 'python tools/bsn_proposal_generation.py ' \
              'configs/localization/bsn/' \
              f'bsn_pgm_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature.py'\
              ' --mode test'

    pem_gen = f'CUDA_VISIBLE_DEVICES={CUDA[0]} PORT={port} ' \
              'bash tools/dist_test.sh ' \
              'configs/localization/bsn/' \
              f'bsn_pem_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature.py'\
              ' work_dirs/' \
              f'bsn_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature/' \
              'pem_latest.pth 1'

    eval_cmd = 'python tools/analysis/report_map.py ' \
               if dataset == 'activitynet' \
               else 'python tools/analysis/report_trunet_map.py '
    eval_cmd += '--proposal work_dirs/' \
                f'bsn_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature/' \
                'results.json '
    if dataset == 'activitynet':
        eval_cmd += '--gt data/ActivityNet/anet_anno_val.json ' \
                    '--det-output work_dirs/' \
                    f'bsn_{shape}_{batch}x{gpus}_{epoch}e_{dataset}_feature/'\
                    'det_results.json '
    else:
        eval_cmd += '--gt data/TruNet/val_meta.json '

    flag = False
    if tem or flag:
        flag = True
        status = os.system(tem_gen)
        if status != 0:
            return
    if pgm or flag:
        flag = True
        status = os.system(pgm_cmd)
        if status != 0:
            return
    if pem or flag:
        flag = True
        status = os.system(pem_gen)
        if status != 0:
            return
    if mAP or flag:
        os.system(eval_cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', nargs='+', type=str)
    parser.add_argument('--shape', type=str)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--port', type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--tem-train', action='store_true')
    parser.add_argument('--tem', action='store_true')
    parser.add_argument('--pgm', action='store_true')
    parser.add_argument('--pem', action='store_true')
    parser.add_argument('--map', action='store_true')

    args = parser.parse_args()
    CUDA, shape, batch, epoch, dataset, port = (args.cuda, args.shape,
                                                args.batch, args.epoch,
                                                args.dataset, args.port)
    tem_train, tem, pgm, pem, mAP = (args.tem_train, args.tem, args.pgm,
                                     args.pem, args.map)
    gpus = len(CUDA)
    train = args.train
    if train:
        train_bsn()
    else:
        evaluate_bsn()
