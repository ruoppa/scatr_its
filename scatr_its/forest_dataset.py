import logging

import numpy as np
import os.path as osp

from typing import Union
from mmengine.logging import print_log
from mmdet3d.datasets.scannet_dataset import ScanNetDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class ForestDataset(ScanNetDataset):
    METAINFO = {
        "classes": ("other", "tree"),
        "seg_valid_class_ids": (0, 1),
        "seg_all_class_ids": (0, 1),
    }

    def __init__(
        self,
        *args,
        data_prefix: dict = dict(
            pts="points",
            pts_instance_mask="instance_mask",
            pts_semantic_mask="semantic_mask",
            pts_superpoint_labels="superpoint_labels",
        ),
        **kwargs,
    ):
        super().__init__(*args, data_prefix=data_prefix, **kwargs)

    def get_scene_idxs(self, *args, **kwargs) -> np.ndarray:
        return np.arange(len(self)).astype(np.int32)

    def parse_data_info(self, info):
        # Include offests in data info. See source code of Det3DDataSet
        if self.modality["use_lidar"]:
            info["offsets"] = info["lidar_points"]["offsets"]
            info["scales"] = info["lidar_points"]["scales"]
        info["pts_superpoint_labels_path"] = osp.join(
            self.data_prefix.get("pts_superpoint_labels", ""),
            info["pts_superpoint_labels_path"],
        )
        info = super().parse_data_info(info)

        return info

    def parse_ann_info(self, info: dict) -> Union[dict, None]:
        """Process the `instances` in data info to `ann_info`.

        In `Custom3DDataset`, we simply concatenate all the field
        in `instances` to `np.ndarray`, you can do the specific
        process in subclass. You have to convert `gt_bboxes_3d`
        to different coordinates according to the task.

        (custom function to add instance centers)

        Args:
            info (dict): Info dict.

        Returns:
            dict or None: Processed `ann_info`.
        """
        # add s or gt prefix for most keys after concat
        # we only process 3d annotations here, the corresponding
        # 2d annotation process is in the `LoadAnnotations3D`
        # in `transforms`
        name_mapping = {
            "bbox_label_3d": "gt_labels_3d",
            "bbox_label": "gt_bboxes_labels",
            "bbox": "gt_bboxes",
            "bbox_3d": "gt_bboxes_3d",
            "depth": "depths",
            "center_2d": "centers_2d",
            "attr_label": "attr_labels",
            "velocity": "velocities",
            "instance_center": "instance_centers",  # <- Only change
        }
        instances = info["instances"]
        # empty gt
        if len(instances) == 0:
            return None
        else:
            keys = list(instances[0].keys())
            ann_info = dict()
            for ann_name in keys:
                temp_anns = [item[ann_name] for item in instances]
                # map the original dataset label to training label
                if "label" in ann_name and ann_name != "attr_label":
                    temp_anns = [self.label_mapping[item] for item in temp_anns]
                if ann_name in name_mapping:
                    mapped_ann_name = name_mapping[ann_name]
                else:
                    mapped_ann_name = ann_name

                if "label" in ann_name:
                    temp_anns = np.array(temp_anns).astype(np.int64)
                elif ann_name in name_mapping:
                    temp_anns = np.array(temp_anns).astype(np.float32)
                else:
                    temp_anns = np.array(temp_anns)

                ann_info[mapped_ann_name] = temp_anns
            ann_info["instances"] = info["instances"]

            for label in ann_info["gt_labels_3d"]:
                if label != -1:
                    cat_name = self.metainfo["classes"][label]
                    self.num_ins_per_cat[cat_name] += 1

        return ann_info

    def __getitem__(self, idx: int) -> dict:
        """Copy of mmengine.dataset.BaseDataset __getitem__ method, with behavior changed such that
        empty scenes are also allowed in test mode (since they can occur during validation too)
        """
        if not self._fully_initialized:
            print_log(
                "Please call `full_init()` method manually to accelerate the speed.",
                logger="current",
                level=logging.WARNING,
            )
            self.full_init()

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = self._rand_another()
                continue
            return data

        raise Exception(
            f"Cannot find a valid sample after fetching {self.max_refetch} times! "
            "Please check your data path and pipeline"
        )
