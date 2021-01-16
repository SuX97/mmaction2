# dataset settings
dataset_type = 'TruNetDataset'
data_root = 'data/TruNet/train_mean_2000/'
data_root_val = 'data/TruNet/val_mean_2000/'
ann_file_train = 'data/TruNet/train_meta.json'
ann_file_val = 'data/TruNet/val_meta.json'
ann_file_test = 'data/TruNet/val_meta.json'

work_dir = 'work_dirs/bsn_temporalx4096_1x1_70e_trunet_feature/'
tem_results_dir = f'{work_dir}/tem_results/'
pgm_proposals_dir = f'{work_dir}/pgm_proposals/'
pgm_features_dir = f'{work_dir}/pgm_features/'

temporal_scale = 2000
pgm_proposals_cfg = dict(
    pgm_proposals_thread=1, temporal_scale=temporal_scale, peak_threshold=0.5)
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
