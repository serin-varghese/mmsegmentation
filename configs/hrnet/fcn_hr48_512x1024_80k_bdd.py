_base_ = [
    "../_base_/models/fcn_hr18.py",
    "../_base_/datasets/bdd100k.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_160k.py",
]

crop_size = (1024, 1024)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean = [123.675, 116.28, 103.53],  # RGB
    std = [58.395, 57.12, 57.375],     # RGB
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255
)

model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384])))

auxiliary_head=dict(
    type='FCNHead',
    in_channels=192,  # from stage 3 or any other
    channels=256,
    num_convs=1,
    num_classes=19,  # or whatever your num_classes is
    loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
)

optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "pos_block": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
            "head": dict(lr_mult=10.0),
        }
    ),
)

param_scheduler = [
    dict(
        type="LinearLR", 
        start_factor=1e-6, 
        by_epoch=False, 
        begin=0, 
        end=1500
    ),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=100000,  # match max_iters
        by_epoch=False,
    )
]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=100000, val_interval=5000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=10000),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)

train_dataloader = dict(batch_size=8, num_workers=4)
val_dataloader = dict(batch_size=4, num_workers=4)
test_dataloader = val_dataloader