_base_ = [
    "mmdet3d::_base_/default_runtime.py",
]
custom_imports = dict(imports=["scatr_its"])

model = dict(
    type="SCATR_ITS",
    data_preprocessor=dict(type="Det3DDataPreprocessor"),
    in_channels=None,
    num_channels=None,
    voxel_size=None,
    num_semantic_classes=None,
    min_spatial_shape=128,
    fg_semantic_class=1,
    bg_semantic_class=0,
    bg_instance_class=0,
    query_point_num=300,
    prepare_epochs=None,
    radius=None,
    step_size_ratio=None,
    num_points=None,
    score_th=None,
    overlap_th=None,
    chunk_size=20_000,
    cylinder_stride_mode=None,
    max_object_size=None,
    sp_processing=True,
    sp_size_aware=True,
    dynamic_query_count=True,
    test_mode=None,
    backbone=dict(
        type="SpConvUNet",
        num_planes=None,
        return_blocks=True,
    ),
    decoder=dict(
        type="QueryDecoder",
        num_layers=6,
        num_semantic_queries=None,
        in_channels=None,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn="gelu",
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=True,
        instance_only=False,
    ),
    dynamic_query_module=dict(
        type="DynamicQueryModule",
        bins=None,
        in_channels=None,
        ordinal=None,
    ),
    criterion=dict(
        type="ForestUnifiedCriterion",
        num_semantic_classes=None,
        sem_criterion=dict(
            type="ForestSemanticCriterion",
            loss_weight=0.2,
            seg_loss=dict(type="mmdet.CrossEntropyLoss", use_sigmoid=True),
            superpoint_size_aware=False,
        ),
        inst_criterion=dict(
            type="ForestInstanceCriterion",
            matcher=dict(type="One2ManyMatcher"),
            loss_weight=[1.0, 1.0, 0.5],
            fix_dice_loss_weight=True,
            iter_matcher=True,
            fix_mean_loss=True,
            superpoint_size_aware=False,
        ),
        instance_only=False,
    ),
    train_cfg=dict(),
    test_cfg=dict(
        topk_insts=300,
        inst_score_thr=0.0,
        pan_score_thr=0.0,
        npoint_thr=10,
        obj_normalization=True,
        obj_normalization_thr=0.01,
        sp_score_thr=0.15,
        z_diff_thr=5,
        nms=True,
        matrix_nms_kernel="linear",
        output_data_dir=None,
        file_extension=None,
        instance_pred_field=None,
        semantic_pred_field=None,
        instance_gt_field=None,
        semantic_gt_field=None,
        score_pred_field=None,
        input_file_extension=None,
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
    pts_superpoint_labels="superpoint_labels"
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
        with_superpoint_labels=True,
        with_instance_centers=False,
    ),
    dict(
        type="NormalizePointsReflectance",
        coords_dim=3,
        method="iqr_minmax",
        skip_flag=False,
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
            "pts_superpoint_labels",
            "superpoint_sizes",
            "pts_center_offsets",
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
        with_superpoint_labels=True,
        with_instance_centers=False,
    ),
    dict(
        type="NormalizePointsReflectance",
        coords_dim=3,
        method="iqr_minmax",
        skip_flag=False,
    ),
    dict(type="CylinderCrop", radius=None, fix_center=True),
    dict(type="GridSample", grid_size=0.2),
    dict(type="PointSample_", num_points=640000),
    dict(type="SkipEmptyScene_"),
    dict(type="PointInstClassMapping_", num_classes=None),
    dict(
        type="Pack3DDetInputs_",
        keys=[
            "points",
            "gt_labels_3d",
            "pts_semantic_mask",
            "pts_instance_mask",
            "ratio_inspoint",
            "instance_mask",
            "pts_superpoint_labels",
            "superpoint_sizes",
            "pts_center_offsets",
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
        with_mask_3d=True,
        with_seg_3d=True,
        with_superpoint_labels=True,
        with_instance_centers=False,
    ),
    dict(
        type="NormalizePointsReflectance",
        coords_dim=3,
        method="iqr_minmax",
        skip_flag=False,
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
    type="ForestMetric",
    data_path=None,
    metadata_path=None,
    iou_thresh=None,
    metric_list=None,
    instance_pred_field=None,
    confidence_field=None,
    instance_gt_field=None,
    semantic_pred_field=None,
    semantic_gt_field=None,
    semantic_class_names=None,
    file_type=None,
    ignore_classes=None,
    filter_small_instances=None,
    min_instance_points=None,
    min_instance_h=None,
    test_mode=False,
)
test_evaluator = val_evaluator

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=None, val_interval=None)
val_cfg = dict(type="ValLoop", val_interval=None)
test_cfg = dict(type="TestLoop")
randomness=dict(seed=33)
find_unused_parameters = True
