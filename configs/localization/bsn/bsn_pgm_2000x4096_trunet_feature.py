# dataset settings
dataset_type = 'TruNetDataset'
data_root = 'data/train_mean_2000/'
data_root_val = 'data/val_mean_2000/'
ann_file_train = 'data/train_meta.json'
ann_file_val = 'data/val_meta.json'
ann_file_test = 'data/val_meta.json'

work_dir = 'work_dirs/bsn_pgm_2000x4096_trunet_feature/'
tem_work_dir = 'work_dirs/bsn_tem_2000x4096_8x24_70e_trunet_feature/'
tem_results_dir = f'{tem_work_dir}/tem_val_results/'
pgm_proposals_dir = f'{work_dir}/pgm_val_proposals/'
pgm_features_dir = f'{work_dir}/pgm_val_features/'

temporal_scale = 2000
pgm_proposals_cfg = dict(
    pgm_proposals_thread=8, temporal_scale=temporal_scale, peak_threshold=0.5)
pgm_features_test_cfg = dict(
    pgm_features_thread=4,
    top_k=1000,
    num_sample_start=8,
    num_sample_end=8,
    num_sample_action=16,
    num_sample_interp=3,
    bsp_boundary_ratio=0.2)
pgm_features_train_cfg = dict(
    pgm_features_thread=4,
    top_k=500,
    num_sample_start=8,
    num_sample_end=8,
    num_sample_action=16,
    num_sample_interp=3,
    bsp_boundary_ratio=0.2)
