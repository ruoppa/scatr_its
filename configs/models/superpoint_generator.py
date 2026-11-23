_base_ = [
    "mmdet3d::_base_/default_runtime.py",
]
custom_imports = dict(imports=["scatr_its"])

model = dict(
    type="SuperpointGenerator",
    data_preprocessor=dict(type="Det3DDataPreprocessor"),
    in_channels=None,
    num_channels=None,
    voxel_size=None,
    num_semantic_classes=None,
    min_spatial_shape=128,
    fg_semantic_class=1,
    radius=None,
    step_size_ratio=None,
    num_points=None,
    sp_merge_mode=None,
    chunk_size=20_000,
    reg_lambda_fg=None,
    reg_lambda_bg=None,
    min_size=None,
    graph_k=None,
    separate_bg_fg=None,
    spatial_weight=None,
    edge_weight_mode=None,
    cylinder_stride_mode=None,
    max_object_size=None,
    max_iterations=None,
    test_mode=None,
    force_rewrite=None,
    backbone=dict(
        type="SpConvUNet",
        num_planes=None,
        return_blocks=True,
    ),
    train_cfg=dict(),
    test_cfg=dict(
        superpoint_data_dir=None,
    ),
)

# Dataset settings
dataset_type = "ForestDataset"
split_infos_names = dict(
    training="training_infos.pkl",
    validation="validation_infos.pkl",
    test="test_infos.pkl",
)
data_prefix = dict(
    pts="points",
    pts_instance_mask="instance_mask",
    pts_semantic_mask="semantic_mask",
)

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="DEPTH",
        shift_height=False,
        use_color=False,
        load_dim=None,
        use_dim=None,
    ),
    dict(
        type="LoadAnnotations3D_",
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_superpoint_labels=False,
        with_instance_centers=False,
    ),
    dict(type="CylinderCrop", radius=None),
    dict(type="GridSample", grid_size=None),
    dict(type="PointSample_", num_points=None),
    dict(type="SkipEmptyScene_"),
    dict(type="PointInstClassMapping_", num_classes=None),
    dict(
        type="RandomFlip3D",
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.0,
    ),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-3.14, 3.14],
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False,
    ),
    dict(
        type="Pack3DDetInputs_",
        keys=[
            "points",
            "gt_labels_3d",
            "pts_semantic_mask",
            "pts_instance_mask",
            "ratio_inspoint",
            "instance_mask",
        ],
    ),
]
val_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="DEPTH",
        shift_height=False,
        use_color=False,
        load_dim=None,
        use_dim=None,
    ),
    dict(
        type="LoadAnnotations3D_",
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_superpoint_labels=False,
        with_instance_centers=False,
    ),
    dict(type="CylinderCrop", radius=None, fix_center=True),
    dict(type="GridSample", grid_size=None),
    dict(type="PointSample_", num_points=None),
    dict(type="SkipEmptyScene_"),
    dict(type="PointInstClassMapping_", num_classes=None),
    dict(
        type="Pack3DDetInputs_",
        keys=[
            "points",
            "gt_labels_3d",
            "pts_semantic_mask",
            "pts_instance_mask",
            "instance_mask",
        ],
    ),
]
test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="DEPTH",
        shift_height=False,
        use_color=False,
        load_dim=None,
        use_dim=None,
    ),
    dict(
        type="LoadAnnotations3D_",
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=False,
        with_superpoint_labels=False,
    ),
    dict(
        type="Pack3DDetInputs_",
        keys=[
            "points",
            "gt_labels_3d",
            "pts_semantic_mask",
            "pts_instance_mask",
            "instance_mask",
            "pts_superpoint_labels",
        ],
    ),
]

# Run settings
train_dataloader = dict(
    batch_size=2,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=None,
        ann_file=split_infos_names["training"],
        data_prefix=data_prefix,
        pipeline=None,
        filter_empty_gt=True,
        box_type_3d="Depth",
        backend_args=None,
    ),
)
val_dataloader = dict(
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=None,
        ann_file=split_infos_names["validation"],
        data_prefix=data_prefix,
        pipeline=None,
        box_type_3d="Depth",
        test_mode=True,
        backend_args=None,
    ),
)
test_dataloader = dict(
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=None,
        ann_file=split_infos_names["test"],
        data_prefix=data_prefix,
        pipeline=None,
        box_type_3d="Depth",
        test_mode=True,
        backend_args=None,
    ),
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2),
)
param_scheduler = dict(type="PolyLR", begin=0, end=450000, power=0.9, by_epoch=False)
custom_hooks = [dict(type="EmptyCacheHook", after_iter=True)]
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook_",
        interval=1,
        max_keep_ckpts=3,
        save_optimizer=True,
        keep_every_n_epochs=None,
    ),
    logger=dict(type="LoggerHook", interval=20),
    runtime_info=dict(type="RuntimeInfoHook_"),
)

val_evaluator = dict(
    type="SuperpointMetric",
    data_path=None,
    superpoint_path=None,
    metadata_path=None,
    iou_thresh=None,
    metric_list=None,
    instance_gt_field=None,
    semantic_gt_field=None,
    semantic_class_names=None,
    file_type=None,
    ignore_classes=None,
    semantic_class_mapping=None,
    graph_k=None,
    test_mode=False,
)
test_evaluator = val_evaluator

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=None, val_interval=None)
val_cfg = dict(type="ValLoop", val_interval=None)
test_cfg = dict(type="TestLoop")
randomness=dict(seed=33)
find_unused_parameters = True
