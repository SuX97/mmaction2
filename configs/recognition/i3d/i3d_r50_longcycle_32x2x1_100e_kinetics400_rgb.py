# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3d',
        pretrained2d=False,
        pretrained=None,
        depth=50,
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=dict(type='BN3d'),
        norm_eval=False,
        inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
        zero_init_residual=False),
    cls_head=dict(
        type='I3DHead',
        num_classes=400,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01))
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips=None)
multi_grid = dict(
    default_s=224,
    long_cycle=True,
    short_cycle=False,
    long_cycle_factors=((0.25, 0.5**0.5), (0.5, 0.5**0.5), (0.5, 1), (1, 1)),
    short_cycle_factors=(0.5, 0.5**0.5),
    epoch_factor=1.0)
# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/kinetics400/rawframes_train'
data_root_val = 'data/kinetics400/rawframes_val'
ann_file_train = 'data/kinetics400/kinetics400_train_list_rawframes.txt'
ann_file_val = 'data/kinetics400/kinetics400_val_list_rawframes.txt'
ann_file_test = 'data/kinetics400/kinetics400_val_list_rawframes.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
mc_cfg = dict(
    server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf',
    client_cfg='/mnt/lustre/share/memcached_client/client.conf',
    sys_path='/mnt/lustre/share/pymc/py3')
train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(
        type='RawFrameDecode',
        io_backend='memcached',
        decoding_backend='turbojpeg',
        **mc_cfg),
    dict(type='Resize', scale=(-1, 256), lazy=True),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.8),
        random_crop=False,
        max_wh_scale_gap=0,
        lazy=True),
    dict(type='Resize', scale=(224, 224), keep_ratio=False, lazy=True),
    dict(type='Flip', flip_ratio=0.5, lazy=True),
    dict(type='Fuse'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode', io_backend='memcached', **mc_cfg),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode', io_backend='memcached', **mc_cfg),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        short_cycle_factors=multi_grid['short_cycle_factors']),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[40, 80])
total_epochs = 100
checkpoint_config = dict(interval=1)
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl', port=29501)
log_level = 'INFO'
work_dir = './work_dirs/i3d_r50_multigrid_32x3x1_100e_kinetics400_rgb/'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
