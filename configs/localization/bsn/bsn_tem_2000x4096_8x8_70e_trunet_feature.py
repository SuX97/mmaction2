# model settings
model = dict(
    type='TEM',
    temporal_dim=2000,
    boundary_ratio=0.1,
    tem_feat_dim=4096,
    tem_hidden_dim=512,
    tem_match_threshold=0.5)
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='score')
# dataset settings
dataset_type = 'TruNetDataset'
data_root = 'data/train_mean_2000/'
data_root_val = 'data/val_mean_2000/'
ann_file_train = 'data/train_meta.json'
ann_file_val = 'data/val_meta.json'
ann_file_test = 'data/val_meta.json'

work_dir = 'work_dirs/bsn_tem_2000x4096_8x8_70e_trunet_feature/'
tem_results_dir = f'{work_dir}/tem_results/'

test_pipeline = [
    dict(type='LoadLocalizationFeature'),
    dict(
        type='Collect',
        keys=['raw_feature'],
        meta_name='video_meta',
        meta_keys=['video_name']),
    dict(type='ToTensor', keys=['raw_feature'])
]
train_pipeline = [
    dict(type='LoadLocalizationFeature'),
    dict(type='GenerateLocalizationLabels'),
    dict(
        type='Collect',
        keys=['raw_feature', 'gt_bbox'],
        meta_name='video_meta',
        meta_keys=['video_name']),
    dict(type='ToTensor', keys=['raw_feature', 'gt_bbox']),
    dict(type='ToDataContainer', fields=[dict(key='gt_bbox', stack=False)])
]
val_pipeline = [
    dict(type='LoadLocalizationFeature'),
    dict(type='GenerateLocalizationLabels'),
    dict(
        type='Collect',
        keys=['raw_feature', 'gt_bbox'],
        meta_name='video_meta',
        meta_keys=['video_name']),
    dict(type='ToTensor', keys=['raw_feature', 'gt_bbox']),
    dict(type='ToDataContainer', fields=[dict(key='gt_bbox', stack=False)])
]

data = dict(
    videos_per_gpu=8,
    workers_per_gpu=8,
    train_dataloader=dict(drop_last=True),
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        data_prefix=data_root_val),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        data_prefix=data_root_val),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        data_prefix=data_root))

# optimizer
optimizer = dict(
    type='SGD', lr=0.001 * 8 * 2 * 4 / 256, momentum=0.9,
    weight_decay=0.0005)  # batch_size

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=80)

total_epochs = 70
checkpoint_config = dict(interval=10, filename_tmpl='tem_epoch_{}.pth')

log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
output_config = dict(out=tem_results_dir, output_format='csv')
