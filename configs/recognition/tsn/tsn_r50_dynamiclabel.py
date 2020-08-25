# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=46,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        # consensus=dict(type='MaxConsensus', dim=1),
        dropout_ratio=0.4,
        # loss_cls=dict(type='BCELossWithLogits', loss_weight=100.0),
        loss_cls=dict(type='BCELossWithLogits'),
        multi_class=True,
        # label_smooth_eps=5e-2,
        init_std=0.01))
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips=None)
# dataset settings
dataset_type = 'RawframeDataset'
data_root = ''
data_root_val = ''
# ann_file_train = './annos/mapped_train_anno_frame_new.txt'
ann_file_train = './annos/oppo_train_val_merge_set.txt'
# ann_file_val = './annos/mapped_val_anno_frame.txt'
# ann_file_train = '/mnt/lustrenew/DATAshare/vug/video/OppoAlbum_DynamicLabels/train_anno_frame.txt'
# ann_file_train = './annos/mapped_train_anno_frame_new.txt'
ann_file_val = './annos/200806_test_anno_videos.txt'
# ann_file_val = './annos/mapped_val_anno_frame_new.txt'
# ann_file_val = '/mnt/lustrenew/DATAshare/vug/video/OppoAlbum_DynamicLabels/0715OPPOtestSetMultiLabel/0806_mapped_val_anno_frame.txt'
# ann_file_train = '/mnt/lustrenew/DATAshare/vug/video/OppoAlbum_DynamicLabels/0715OPPOtestSetMultiLabel/mapped_train_anno_frame.txt'
# ann_file_val = '/mnt/lustrenew/DATAshare/vug/video/OppoAlbum_DynamicLabels/0715OPPOtestSetMultiLabel/mapped_val_anno_frame.txt'

ann_file_test = './annos/200806_test_anno_videos.txt'
# ann_file_test = '/mnt/lustrenew/DATAshare/vug/video/OppoAlbum_DynamicLabels/0715OPPOtestSetMultiLabel/0806_mapped_val_anno_frame.txt'
# ann_file_test = './annos/2_video_val_anno_frame.txt'
# ann_file_test = '/mnt/lustrenew/DATAshare/vug/video/OppoAlbum_DynamicLabels/0715OPPOtestSetMultiLabel/video_val_anno_frame.txt'
# ann_file_test = '/mnt/lustrenew/DATAshare/vug/video/OppoAlbum_DynamicLabels/0715OPPOtestSetMultiLabel/2_test_anno_frame.txt'
#ann_file_test = '/mnt/lustrenew/DATAshare/vug/video/OppoAlbum_DynamicLabels/test_anno_frame.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
mc_cfg = dict(
    server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf',
    client_cfg='/mnt/lustre/share/memcached_client/client.conf',
    sys_path='/mnt/lustre/share/pymc/py3')

train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=3),
    dict(
        type='FrameSelector',
        io_backend='memcached',
        decoding_backend='turbojpeg',
        **mc_cfg),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=3,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
'''
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=25,
        test_mode=True),
    dict(type='FrameSelector', 
        io_backend='memcached', 
        decoding_backend='turbojpeg',
        **mc_cfg),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
'''
# '''
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=25,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
# '''
'''
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=3),
    dict(
        type='FrameSelector',
        io_backend='memcached',
        decoding_backend='turbojpeg',
        **mc_cfg),
    dict(type='Resize', scale=(-1, 256), lazy=True),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        lazy=True),
    dict(type='Resize', scale=(224, 224), keep_ratio=False, lazy=True),
    dict(type='Flip', flip_ratio=0.5, lazy=True),
    dict(type='Fuse'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=3,
        test_mode=True),
    dict(type='FrameSelector', 
        io_backend='memcached', 
        decoding_backend='turbojpeg',
        **mc_cfg),
    dict(type='Resize', scale=(-1, 256), lazy=True),
    dict(type='CenterCrop', crop_size=224, lazy=True),
    dict(type='Flip', flip_ratio=0, lazy=True),
    dict(type='Fuse'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=25,
        test_mode=True),
    dict(type='FrameSelector', 
        io_backend='memcached', 
        decoding_backend='turbojpeg',
        **mc_cfg),
    dict(type='Resize', scale=(-1, 256), lazy=True),
    dict(type='TenCrop', crop_size=224, lazy=True),
    dict(type='Flip', flip_ratio=0, lazy=True),
    dict(type='Fuse'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
'''
data = dict(
    videos_per_gpu=64,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        multi_class=True,
        num_classes=46),
    val=dict(
        type='VideoDataset',
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        multi_class=True,
        num_classes=46),
    test=dict(
        type='VideoDataset',
        # type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        multi_class=True,
        num_classes=46))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[25, 40])
total_epochs = 50
checkpoint_config = dict(interval=1)
evaluation = dict(
    interval=1, metrics=['precision_recall'])#, topk=(1, 5))
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl', port=12523)
log_level = 'INFO'
work_dir = './work_dirs/tsn_r50_1x1x3_100e_kinetics400_rgb/'
load_from = None
resume_from = None
workflow = [('train', 1)]
