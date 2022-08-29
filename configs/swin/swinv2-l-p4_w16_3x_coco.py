_base_ = [
    '../_base_/models/swinv2.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
    	_delete_=True, 
        type='SwinTransformerV2',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=16,
        drop_path_rate=0.5,
        patch_norm=True,
        convert_weights=True,
    ))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

evaluation = dict(metric=['bbox'])

lr_config = dict(step=[8, 11])

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook')
    ])
log_level = 'INFO'
# do not use mmdet version fp16
fp16 = None
