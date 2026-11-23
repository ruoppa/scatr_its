import mmengine

import numpy as np

from mmengine.fileio import get
from mmdet3d.datasets.transforms import LoadAnnotations3D
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadAnnotations3D_(LoadAnnotations3D):
    def __init__(
        self,
        *args,
        with_superpoint_labels: bool = True,
        with_instance_centers: bool = False,
        **kwargs,
    ) -> None:
        """Wrapper for LoadAnnotations3D that adds the ability to load precomputed superpoint labels

        Args:
            with_superpoint_labels (bool, optional): If True, load superpoint labels from file. Defaults to True.
        """
        super().__init__(*args, **kwargs)
        self.with_superpoint_labels = with_superpoint_labels
        self.with_instance_centers = with_instance_centers

    def _load_superpoint_labels(self, results: dict) -> dict:
        """Private function to load 3D (precomputed) superpoint labels.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D superpoint labels.
        """
        pts_superpoint_labels_path = results["pts_superpoint_labels_path"]

        try:
            mask_bytes = get(pts_superpoint_labels_path, backend_args=self.backend_args)
            pts_superpoint_labels = np.frombuffer(mask_bytes, dtype=np.int64)
        except ConnectionError:
            mmengine.check_file_exist(pts_superpoint_labels_path)
            pts_superpoint_labels = np.fromfile(
                pts_superpoint_labels_path, dtype=np.int64
            )
        _, pts_superpoint_labels = np.unique(pts_superpoint_labels, return_inverse=True)

        results["pts_superpoint_labels"] = pts_superpoint_labels
        # 'eval_ann_info' will be passed to evaluator
        if "eval_ann_info" in results:
            results["eval_ann_info"]["pts_superpoint_labels"] = pts_superpoint_labels

        return results

    def _load_instance_centers(self, results: dict) -> dict:
        """Private function to load precomputed instance centers (xyz coordinates + instance labels)

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded instance centers.
        """
        if "ann_info" in results:
            results["instance_centers"] = results["ann_info"]["instance_centers"]
        else:
            results["instance_centers"] = results["eval_ann_info"]["instance_centers"]

        return results

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
            semantic segmentation annotations.
        """
        if "eval_ann_info" in results:
            if results["eval_ann_info"] is None:
                results["eval_ann_info"] = dict()
        results = super().transform(results)
        if self.with_superpoint_labels:
            results = self._load_superpoint_labels(results)
        if self.with_instance_centers:
            results = self._load_instance_centers(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_panoptic_3d:
            results = self._load_panoptic_3d(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        indent_str = "    "
        repr_str = self.__class__.__name__ + "(\n"
        repr_str += (
            f"{indent_str}with_superpoint_labels={self.with_superpoint_labels}, "
        )
        repr_str += f"{indent_str}with_instance_centers={self.with_instance_centers}, "
        repr_str += f"{indent_str}with_bbox_3d={self.with_bbox_3d}, "
        repr_str += f"{indent_str}with_label_3d={self.with_label_3d}, "
        repr_str += f"{indent_str}with_attr_label={self.with_attr_label}, "
        repr_str += f"{indent_str}with_mask_3d={self.with_mask_3d}, "
        repr_str += f"{indent_str}with_seg_3d={self.with_seg_3d}, "
        repr_str += f"{indent_str}with_panoptic_3d={self.with_panoptic_3d}, "
        repr_str += f"{indent_str}with_bbox={self.with_bbox}, "
        repr_str += f"{indent_str}with_label={self.with_label}, "
        repr_str += f"{indent_str}with_mask={self.with_mask}, "
        repr_str += f"{indent_str}with_seg={self.with_seg}, "
        repr_str += f"{indent_str}with_bbox_depth={self.with_bbox_depth}, "
        repr_str += f"{indent_str}poly2mask={self.poly2mask})"
        repr_str += f"{indent_str}seg_offset={self.seg_offset})"

        return repr_str
