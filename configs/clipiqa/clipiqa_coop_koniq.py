exp_name = 'clipiqa_coop_koniq'

# model settings
model = dict(
    type='CLIPIQA',
    generator=dict(
        type='CLIPIQAPredictor',
        backbone_name='RN50',
        # classnames=[
        #     ['Good photo.', 'Bad photo.'],
        # ]),

        # test degradation attributes
        classnames=[
            ['Good photo.', 'Bad photo.'],
            ['Bright photo.', 'Dark photo.'],
            ['Sharp photo.', 'Blurry photo.'],
            ['Noisy photo.', 'Clean photo.'],
            ['Colorful photo.', 'Dull photo.'],
            ['High contrast photo.', 'Low contrast photo.'],
        ]),

        # test AVA attributes
        # classnames=[
        #     ['Aesthetic photo.', 'Not aesthetic photo.'],
        #     ['Happy photo.', 'Sad photo.'],
        #     ['Natural photo.', 'Synthetic photo.'],
        #     ['New photo.', 'Old photo.'],
        #     ['Scary photo.', 'Peaceful photo.'],
        #     ['Complex photo.', 'Simple photo.'],
        # ]),
    pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = dict(fix_iter=5000)
test_cfg = dict(metrics=['PSNR'], crop_border=0)

# dataset settings
train_dataset_type = 'IQAKoniqDataset'
val_dataset_type = 'IQAKoniqDataset'

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged',
        backend='pillow'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(
        type='Normalize',
        keys=['lq'],
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
        to_rgb=True),
    # dict(type='Resize', keys=['lq'], scale=1/2, keep_ratio=True),
    dict(
        type='Flip', keys=['lq'], flip_ratio=0.5,
        direction='horizontal'),
    # dict(type='Flip', keys=['lq'], flip_ratio=0.5, direction='vertical'),
    # dict(type='RandomTransposeHW', keys=['lq'], transpose_ratio=0.5),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path']),
    dict(type='ImageToTensor', keys=['lq']),
    dict(type='ToTensor', keys=['gt'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged',
        backend='pillow'),
    # dict(type='Resize_PIL', keys=['lq'], scale=512, interpolation='bicubic'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(
        type='Normalize',
        keys=['lq'],
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
        to_rgb=True),
    # dict(type='Resize', keys=['lq'], scale=1/4, keep_ratio=True),
    # dict(type='Resize', keys=['lq'], scale=(224, 224)),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path']),
    dict(type='ImageToTensor', keys=['lq']),
    dict(type='ToTensor', keys=['gt'])
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
]

data = dict(
    workers_per_gpu=6,
    train_dataloader=dict(samples_per_gpu=64, drop_last=True),  # 2 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=dict(
        type='RepeatDataset',
        times=100,
        dataset=dict(
            type=train_dataset_type,
            img_folder=r'D:\Dataset\koniq10k/512x384/',
            ann_file='D:/Dataset/koniq10k/koniq10k_distributions_sets.csv',
            pipeline=train_pipeline,
            test_mode=False)),
    # val
    val=dict(
        type=val_dataset_type,
        img_folder=r'D:\Dataset\koniq10k/512x384/',
        ann_file='D:/Dataset/koniq10k/koniq10k_distributions_sets.csv',
        pipeline=test_pipeline,
        test_mode=True),
    # test
    test=dict(
        type=val_dataset_type,
        img_folder=r'D:\Dataset\koniq10k/512x384/',
        ann_file='D:/Dataset/koniq10k/koniq10k_distributions_sets.csv',
        pipeline=test_pipeline,
        test_mode=True),
)

# optimizer
optimizers = dict(
    generator=dict(
        type='SGD',
        lr=0.002))

# learning policy
# total_iters = 500000
total_iters = 100000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[300000],
    restart_weights=[1],
    min_lr=1e-7)

# checkpoint_config = dict(interval=50000, save_optimizer=True, by_epoch=False)
checkpoint_config = dict(interval=10000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
# evaluation = dict(interval=10000, save_image=False, gpu_collect=True)
evaluation = dict(interval=5000, save_image=False)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True