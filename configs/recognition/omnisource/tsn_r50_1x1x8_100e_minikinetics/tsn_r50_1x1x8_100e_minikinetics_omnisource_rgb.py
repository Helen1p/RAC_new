_base_ = [
    '../../../_base_/models/tsn_r50.py',
    '../../../_base_/schedules/sgd_100e.py',
    '../../../_base_/default_runtime.py'
]

# model settings
model = dict(cls_head=dict(num_classes=200))

omnisource = True
# dataset settings
dataset_type = 'VideoDataset'
# The flag indicates using joint training
omnisource = True

data_root = 'data/OmniSource/kinetics_200_train'
data_root_val = 'data/OmniSource/kinetics_200_val'
web_root = 'data/OmniSource/'
iv_root = 'data/OmniSource/insvideo_200'
kraw_root = 'data/OmniSource/kinetics_raw_200_train'

ann_file_train = 'data/OmniSource/annotations/kinetics_200/k200_train.txt'
ann_file_web = ('data/OmniSource/annotations/webimage_200/'
                'tsn_8seg_webimage_200_wodup.txt')
ann_file_iv = ('data/OmniSource/annotations/insvideo_200/'
               'slowonly_8x8_insvideo_200_wodup.txt')
ann_file_kraw = ('data/OmniSource/annotations/kinetics_raw_200/'
                 'slowonly_8x8_kinetics_raw_200.json')

ann_file_val = 'data/OmniSource/annotations/kinetics_200/k200_val.txt'
ann_file_test = 'data/OmniSource/annotations/kinetics_200/k200_val.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
train_web_pipeline = [
    dict(type='ImageDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
train_iv_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
train_kraw_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
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
        num_clips=8,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

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
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=12,
    omni_videos_per_gpu=[12, 64, 12, 12],
    train_ratio=[2, 1, 1, 1],
    workers_per_gpu=1,
    train=[
        dict(
            type=dataset_type,
            ann_file=ann_file_train,
            data_prefix=data_root,
            pipeline=train_pipeline),
        dict(
            type='ImageDataset',
            ann_file=ann_file_web,
            data_prefix=web_root,
            pipeline=train_web_pipeline,
            num_classes=200,
            sample_by_class=True,
            power=0.5),
        dict(
            type=dataset_type,
            ann_file=ann_file_iv,
            data_prefix=iv_root,
            pipeline=train_iv_pipeline,
            num_classes=200,
            sample_by_class=True,
            power=0.5),
        dict(
            type='RawVideoDataset',
            ann_file=ann_file_kraw,
            data_prefix=kraw_root,
            pipeline=train_kraw_pipeline,
            clipname_tmpl='part_{}.mp4',
            sampling_strategy='positive')
    ],
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    type='SGD', lr=0.00375, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus

# runtime settings
work_dir = ('./work_dirs/omnisource/'
            'tsn_r50_1x1x8_100e_minikinetics_omnisource_rgb')
