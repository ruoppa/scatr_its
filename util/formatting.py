# Adapted from mmdet3d/datasets/transforms/formating.py
import numpy as np

from typing import Dict
from util.structures import InstanceData_
from mmdet3d.datasets.transforms import Pack3DDetInputs
from mmdet3d.datasets.transforms.formating import to_tensor
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import BaseInstance3DBoxes, Det3DDataSample, PointData
from mmdet3d.structures.points import BasePoints


@TRANSFORMS.register_module()
class Pack3DDetInputs_(Pack3DDetInputs):
    """Just add elastic_coords, sp_pts_mask, and gt_sp_masks."""

    INPUTS_KEYS = ["points", "img", "elastic_coords"]
    SEG_KEYS = [
        "gt_seg_map",
        "pts_instance_mask",
        "pts_semantic_mask",
        "gt_semantic_seg",
        "sp_pts_mask",
        "ratio_inspoint",
        "instance_mask",
        "pts_superpoint_labels",
        "superpoint_sizes",
        "pts_center_offsets",
    ]
    INSTANCEDATA_3D_KEYS = [
        "gt_bboxes_3d",
        "gt_labels_3d",
        "attr_labels",
        "depths",
        "centers_2d",
        "gt_sp_masks",
    ]

    # Add offsets and scales to meta keys
    def __init__(
        self,
        keys,
        meta_keys=(
            "img_path",
            "ori_shape",
            "img_shape",
            "lidar2img",
            "depth2img",
            "cam2img",
            "pad_shape",
            "scale_factor",
            "flip",
            "pcd_horizontal_flip",
            "pcd_vertical_flip",
            "box_mode_3d",
            "box_type_3d",
            "img_norm_cfg",
            "num_pts_feats",
            "pcd_trans",
            "sample_idx",
            "pcd_scale_factor",
            "pcd_rotation",
            "pcd_rotation_angle",
            "lidar_path",
            "transformation_3d_flow",
            "trans_mat",
            "affine_aug",
            "sweep_img_metas",
            "ori_cam2img",
            "cam2global",
            "crop_offset",
            "img_crop_offset",
            "resize_img_shape",
            "lidar2cam",
            "ori_lidar2img",
            "num_ref_frames",
            "num_views",
            "ego2global",
            "offsets",
            "scales",
        ),
    ):
        super().__init__(keys, meta_keys)

    def pack_single_results(self, results: Dict) -> Dict:
        """Method to pack the single input data. when the value in this dict is
        a list, it usually is in Augmentations Testing.

        Args:
            results (Dict): Result dict from the data pipeline.

        Returns:
            Dict: A dict contains

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info
              of the sample.
        """
        # Format 3D data
        if "points" in results:
            if isinstance(results["points"], BasePoints):
                results["points"] = results["points"].tensor

        if "img" in results:
            if isinstance(results["img"], list):
                # process multiple imgs in single frame
                imgs = np.stack(results["img"], axis=0)
                if imgs.flags.c_contiguous:
                    imgs = to_tensor(imgs).permute(0, 3, 1, 2).contiguous()
                else:
                    imgs = to_tensor(np.ascontiguousarray(imgs.transpose(0, 3, 1, 2)))
                results["img"] = imgs
            else:
                img = results["img"]
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                # To improve the computational speed by by 3-5 times, apply:
                # `torch.permute()` rather than `np.transpose()`.
                # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
                # for more details
                if img.flags.c_contiguous:
                    img = to_tensor(img).permute(2, 0, 1).contiguous()
                else:
                    img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
                results["img"] = img

        for key in [
            "proposals",
            "gt_bboxes",
            "gt_bboxes_ignore",
            "gt_labels",
            "gt_bboxes_labels",
            "attr_labels",
            "pts_instance_mask",
            "pts_semantic_mask",
            "sp_pts_mask",
            "gt_sp_masks",
            "elastic_coords",
            "centers_2d",
            "depths",
            "gt_labels_3d",
            "pts_superpoint_labels",
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = [to_tensor(res) for res in results[key]]
            else:
                results[key] = to_tensor(results[key])
        if "gt_bboxes_3d" in results:
            if not isinstance(results["gt_bboxes_3d"], BaseInstance3DBoxes):
                results["gt_bboxes_3d"] = to_tensor(results["gt_bboxes_3d"])

        if "gt_semantic_seg" in results:
            results["gt_semantic_seg"] = to_tensor(results["gt_semantic_seg"][None])
        if "gt_seg_map" in results:
            results["gt_seg_map"] = results["gt_seg_map"][None, ...]

        data_sample = Det3DDataSample()
        gt_instances_3d = InstanceData_()
        gt_instances = InstanceData_()
        gt_pts_seg = PointData()

        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]
        data_sample.set_metainfo(img_metas)

        inputs = {}
        for key in self.keys:
            if key in results:
                if key in self.INPUTS_KEYS:
                    inputs[key] = results[key]
                elif key in self.INSTANCEDATA_3D_KEYS:
                    gt_instances_3d[self._remove_prefix(key)] = results[key]
                elif key in self.INSTANCEDATA_2D_KEYS:
                    if key == "gt_bboxes_labels":
                        gt_instances["labels"] = results[key]
                    else:
                        gt_instances[self._remove_prefix(key)] = results[key]
                elif key in self.SEG_KEYS:
                    gt_pts_seg[self._remove_prefix(key)] = results[key]
                else:
                    raise NotImplementedError(
                        f"Please modify "
                        f"`Pack3DDetInputs` "
                        f"to insert {key} in the "
                        f"corresponding field"
                    )

        data_sample.gt_instances_3d = gt_instances_3d
        data_sample.gt_instances = gt_instances
        data_sample.gt_pts_seg = gt_pts_seg
        if "eval_ann_info" in results:
            data_sample.eval_ann_info = results["eval_ann_info"]
        else:
            data_sample.eval_ann_info = None
        packed_results = dict()
        packed_results["data_samples"] = data_sample
        packed_results["inputs"] = inputs

        return packed_results
