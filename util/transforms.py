import torch

import numpy as np

from torch import Tensor
from typing import Dict, Tuple, Optional
from mmcv.transforms import BaseTransform
from mmdet3d.datasets.transforms import PointSample
from mmdet3d.structures import BasePoints
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PointInstClassMapping_(BaseTransform):
    """Delete instances from non-instaces classes.

    Required Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)

    Modified Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)

    Added Keys:
    - gt_labels_3d (int)

    Args:
        num_classes (int): Number of classes.
    """

    def __init__(self, num_classes: int, structured3d: bool = False):
        self.num_classes = num_classes
        self.structured3d = structured3d

    def transform(self, input_dict: Dict) -> Dict:
        """Private function for deleting
            instances from non-instaces classes.

        Args:
            input_dict (Dict): Result dict from loading pipeline.

        Returns:
            Dict: results, 'pts_instance_mask', 'pts_semantic_mask', are updated in the result dict. 'gt_labels_3d' is added.
        """

        # because pts_instance_mask contains instances from non-instaces
        # classes
        pts_instance_mask = np.array(input_dict["pts_instance_mask"])
        pts_semantic_mask = input_dict["pts_semantic_mask"]

        if self.structured3d:
            # wall as one instance
            pts_instance_mask[pts_semantic_mask == 0] = pts_instance_mask.max() + 1
            # floor as one instance
            pts_instance_mask[pts_semantic_mask == 1] = pts_instance_mask.max() + 1

        pts_instance_mask[pts_semantic_mask == self.num_classes] = -1
        pts_semantic_mask[pts_semantic_mask == self.num_classes] = -1

        idxs = np.unique(pts_instance_mask)
        mapping = np.zeros(np.max(idxs) + 2, dtype=int)
        new_idxs = np.arange(len(idxs))
        if idxs[0] == -1:
            mapping[idxs] = new_idxs - 1
            new_idxs = new_idxs[:-1]
        else:
            mapping[idxs] = new_idxs
        pts_instance_mask = mapping[pts_instance_mask]

        input_dict["pts_instance_mask"] = pts_instance_mask
        input_dict["pts_semantic_mask"] = pts_semantic_mask

        gt_labels = np.zeros(len(new_idxs), dtype=int)
        for inst in new_idxs:
            gt_labels[inst] = pts_semantic_mask[pts_instance_mask == inst][0]

        input_dict["gt_labels_3d"] = gt_labels

        return input_dict


@TRANSFORMS.register_module()
class PointSample_(PointSample):
    def _points_random_sampling(
        self, points: BasePoints, num_samples: int
    ) -> Tuple[BasePoints, np.ndarray]:
        """Points random sampling. Sample points to a certain number.

        Args:
            points (BasePoints): 3D Points.
            num_samples (int): Number of samples to be sampled.

        Returns:
            Tuple[BasePoints, np.ndarray]:
                - points (BasePoints): 3D Points.
                - choices (np.ndarray): The generated random samples.
        """

        point_range = range(len(points))
        choices = np.random.choice(point_range, min(num_samples, len(points)))

        return points[choices], choices

    def transform(self, input_dict: Dict) -> Dict:
        """Transform function to sample points to in indoor scenes.

        Args:
            input_dict (Dict): Result dict from loading pipeline.

        Returns:
            Dict: Results after sampling, 'points', 'pts_instance_mask', 'pts_semantic_mask', sp_pts_mask' keys are updated in the result dict.
        """
        points = input_dict["points"]

        # if point number smaller than num_point, skip
        if len(points) < self.num_points:
            return input_dict

        points, choices = self._points_random_sampling(points, self.num_points)
        input_dict["points"] = points
        pts_instance_mask = input_dict.get("pts_instance_mask", None)
        pts_semantic_mask = input_dict.get("pts_semantic_mask", None)
        vote_label = input_dict.get("vote_label", None)
        instance_mask = input_dict.get("instance_mask", None)
        sp_pts_mask = input_dict.get("sp_pts_mask", None)
        pts_superpoint_labels = input_dict.get("pts_superpoint_labels", None)
        pts_center_offsets = input_dict.get("pts_center_offsets", None)

        if pts_instance_mask is not None:
            pts_instance_mask = pts_instance_mask[choices]

            idxs = np.unique(pts_instance_mask)
            mapping = np.zeros(np.max(idxs) + 2, dtype=int)
            new_idxs = np.arange(len(idxs))
            if idxs[0] == -1:
                mapping[idxs] = new_idxs - 1
            else:
                mapping[idxs] = new_idxs
            pts_instance_mask = mapping[pts_instance_mask]

            input_dict["pts_instance_mask"] = pts_instance_mask

        if pts_semantic_mask is not None:
            pts_semantic_mask = pts_semantic_mask[choices]
            input_dict["pts_semantic_mask"] = pts_semantic_mask

        if vote_label is not None:
            vote_label = vote_label[choices]
            input_dict["vote_label"] = vote_label

        if instance_mask is not None:
            instance_mask = instance_mask[choices]
            input_dict["instance_mask"] = instance_mask

        if sp_pts_mask is not None:
            sp_pts_mask = sp_pts_mask[choices]
            sp_pts_mask = np.unique(sp_pts_mask, return_inverse=True)[1]
            input_dict["sp_pts_mask"] = sp_pts_mask

        if pts_superpoint_labels is not None:
            pts_superpoint_labels = pts_superpoint_labels[choices]
            _, pts_superpoint_labels, superpoint_sizes = np.unique(
                pts_superpoint_labels, return_inverse=True, return_counts=True
            )
            input_dict["pts_superpoint_labels"] = pts_superpoint_labels
            input_dict["superpoint_sizes"] = superpoint_sizes
        
        if pts_center_offsets is not None:
            pts_center_offsets = pts_center_offsets[choices]
            input_dict["pts_center_offsets"] = pts_center_offsets

        if "eval_ann_info" in input_dict:
            input_dict["eval_ann_info"]["pts_instance_mask"] = pts_instance_mask
            input_dict["eval_ann_info"]["pts_semantic_mask"] = pts_semantic_mask
            input_dict["eval_ann_info"]["instance_mask"] = instance_mask
            input_dict["eval_ann_info"]["pts_superpoint_labels"] = pts_superpoint_labels

        return input_dict


@TRANSFORMS.register_module()
class SkipEmptyScene(BaseTransform):
    """Skip empty scene during training.

    Required Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)
    - points (:obj:`BasePoints`)
    - gt_labels_3d (int)

    Modified Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)
    - points (:obj:`BasePoints`)
    - gt_labels_3d (int)

    """

    def transform(self, input_dict: Dict) -> Dict:
        """Private function for skipping empty scene during training.

        Args:
            input_dict (Dict): Result dict from loading pipeline.

        Returns:
            Dict: results, 'pts_instance_mask', 'pts_semantic_mask', 'points', 'gt_labels_3d' are updated in the result dict.
        """

        if len(input_dict["gt_labels_3d"]) != 0:
            self.inst = input_dict["pts_instance_mask"]
            self.sem = input_dict["pts_semantic_mask"]
            self.gt_labels = input_dict["gt_labels_3d"]
            self.points = input_dict["points"]
        else:
            input_dict["pts_instance_mask"] = self.inst
            input_dict["pts_semantic_mask"] = self.sem
            input_dict["gt_labels_3d"] = self.gt_labels
            input_dict["points"] = self.points

        return input_dict


@TRANSFORMS.register_module()
class SkipEmptyScene_(BaseTransform):
    """Skip empty scene during training.

    Required Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)
    - points (:obj:`BasePoints`)
    - gt_labels_3d (int)

    Modified Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)
    - points (:obj:`BasePoints`)
    - gt_labels_3d (int)

    """

    def transform(self, input_dict: Dict) -> Dict:
        """Private function for skipping empty scene during training.

        Args:
            input_dict (Dict): Result dict from loading pipeline.

        Returns:
            Dict: results, 'pts_instance_mask', 'pts_semantic_mask', 'points', 'gt_labels_3d' are updated in the result dict.
        """

        if len(input_dict["points"]) == 0:
            return None
        pts_instance_mask = input_dict.get("pts_instance_mask", None)
        if len(np.unique(pts_instance_mask)) < 2:
            return None

        return input_dict


@TRANSFORMS.register_module()
class CylinderCrop(BaseTransform):
    _VALID_NORMALIZATIONS = ["center"]

    def __init__(
        self,
        radius: int,
        normalize: Optional[str] = None,
        fix_center: bool = False,
    ):
        self.radius = radius
        if normalize is not None:
            assert normalize in self._VALID_NORMALIZATIONS, (
                f"'normalize' should be one of '{self._VALID_NORMALIZATIONS}', got '{normalize}' instead!"
            )
            self.normalize = normalize
        else:
            self.normalize = None
        self.fix_center = fix_center

    def transform(self, input_dict: Dict) -> Dict:
        assert "points" in input_dict.keys()
        # Get the tensor of points
        points_tensor = input_dict["points"].tensor
        device = points_tensor.device
        n_points = points_tensor.shape[0]
        if self.fix_center:
            # Fix center to the mean of coordinates (useful for validation)
            center = torch.mean(points_tensor, dim=0)
        else:
            # Select a random center point
            center = points_tensor[torch.randint(0, n_points, (1,))][0]
        # Calculate indices of points within the radius
        dx = points_tensor[:, 0] - center[0]
        dy = points_tensor[:, 1] - center[1]
        cylinder_mask = (dx * dx + dy * dy) < (self.radius**2)
        choices = torch.nonzero(cylinder_mask, as_tuple=False).squeeze(1)
        choices_numpy = choices.cpu().numpy()

        # Update points tensor
        if "points" in input_dict.keys():
            input_dict["points"] = input_dict["points"][choices_numpy]
        pts_instance_mask = input_dict.get("pts_instance_mask", None)
        pts_semantic_mask = input_dict.get("pts_semantic_mask", None)
        sp_pts_mask = input_dict.get("sp_pts_mask", None)
        pts_superpoint_labels = input_dict.get("pts_superpoint_labels", None)
        gt_instance_centers = input_dict.get("instance_centers", None)

        if (pts_instance_mask is not None) and (pts_semantic_mask is not None):
            pts_instance_mask_torch = torch.from_numpy(pts_instance_mask).to(
                device=device
            )
            pts_instance_mask_torch = pts_instance_mask_torch.clone()
            pts_semantic_mask_torch = torch.from_numpy(pts_semantic_mask).to(
                device=device
            )
            instance_mask = pts_semantic_mask_torch != 0
            pts_instance_mask_torch[~instance_mask] = -1
            cylinder_pts_instance_mask_torch = pts_instance_mask_torch[choices]
            # Compute number of points in original instances and the instances within the cylinder
            orig_inst_ids, orig_point_counts = torch.unique(
                pts_instance_mask_torch, return_counts=True
            )
            cylinder_inst_ids, cylinder_point_counts = torch.unique(
                cylinder_pts_instance_mask_torch, return_counts=True
            )
            orig_point_count_lookup = {
                int(k.item()): float(v.item())
                for k, v in zip(orig_inst_ids, orig_point_counts)
            }
            # Compute percentage of instance points within the cylinder for each instance
            ratio_inspoint = {}
            for inst_id, point_count in zip(cylinder_inst_ids, cylinder_point_counts):
                inst_id_val = int(inst_id.item())
                if inst_id_val == -1:
                    continue
                orig_point_count = orig_point_count_lookup.get(inst_id_val, 1)
                if orig_point_count > 0:
                    ratio_inspoint[inst_id_val] = (
                        float(point_count.item()) / orig_point_count
                    )
                else:
                    ratio_inspoint[inst_id_val] = 0.0
            # Map instance ids to a continuous range
            valid_mask = cylinder_pts_instance_mask_torch != -1
            orig_cylinder_inst_ids, cont_cylinder_pts_instance_mask = torch.unique(
                cylinder_pts_instance_mask_torch[valid_mask], return_inverse=True
            )
            cylinder_pts_instance_mask_torch[valid_mask] = (
                cont_cylinder_pts_instance_mask
            )
            ratio_inspoint_cont = {}
            for i, orig_inst_id in enumerate(orig_cylinder_inst_ids):
                orig_inst_id_val = int(orig_inst_id.item())
                ratio_inspoint_cont[i] = ratio_inspoint[orig_inst_id_val]

            # Compute instance center offsets for all points (if possible)
            if ("points" in input_dict.keys()) and (gt_instance_centers is not None):
                cylinder_points = input_dict["points"].tensor[:, :3]
                center_offsets = torch.zeros_like(cylinder_points, device=device)
                gt_instance_centers_tensor = torch.from_numpy(gt_instance_centers)
                # Indices [0, 1, 2] of gt instance centers contain center coordinates
                instance_centers = gt_instance_centers_tensor[:, :3]
                # Index 3 contains the corresponding instance labels
                orig_center_inst_ids = gt_instance_centers_tensor[:, 3].long()
                # Find corresponding instance labels from the continuous labels within the cylinder
                sorted_orig_bbox_inst_ids, sort_ind = torch.sort(orig_center_inst_ids)
                match_ind = torch.searchsorted(
                    sorted_orig_bbox_inst_ids, orig_cylinder_inst_ids
                )
                bbox_match_ind = sort_ind[match_ind]
                # Fetch instance center corresponding to each instance within the cylinder
                cylinder_inst_centers = instance_centers[bbox_match_ind]
                # Compute offsets w.r.t. corresponding istance centers for each point
                # (offset for non-instance points is set to zero)
                center_offsets[valid_mask] = (
                    cylinder_inst_centers[cylinder_pts_instance_mask_torch[valid_mask]]
                    - cylinder_points[valid_mask]
                )
                input_dict["pts_center_offsets"] = center_offsets.cpu().numpy()

            input_dict["pts_instance_mask"] = (
                cylinder_pts_instance_mask_torch.cpu().numpy()
            )
            input_dict["ratio_inspoint"] = ratio_inspoint_cont
            input_dict["instance_mask"] = instance_mask[choices].cpu().numpy()

        # Normalize coordinate values, if necessary
        if "points" in input_dict.keys():
            if self.normalize == "center":
                normalized_points = input_dict["points"].tensor
                # Normalize xyz coordinates by centering xy values around cylinder center and s.t. min z-coordinate is 0
                center_norm = torch.zeros((3,), device=device)
                center_norm[:2] = center[:2].clone()
                center[2] = torch.min(normalized_points[:, 2])
                normalized_points[:, :3] -= center_norm
                num_dims = normalized_points.shape[1]
                input_dict["points"] = type(input_dict["points"])(
                    normalized_points, points_dim=num_dims
                )
            elif self.normalize is not None:
                raise NotImplementedError(
                    f"Normalization of type '{self.normalize}' not yet implemented!"
                )

        if pts_semantic_mask is not None:
            pts_semantic_mask = pts_semantic_mask[choices_numpy]
            input_dict["pts_semantic_mask"] = pts_semantic_mask

        if sp_pts_mask is not None:
            sp_pts_mask = sp_pts_mask[choices_numpy]
            sp_pts_mask = np.unique(sp_pts_mask, return_inverse=True)[1]
            input_dict["sp_pts_mask"] = sp_pts_mask

        if pts_superpoint_labels is not None:
            pts_superpoint_labels = pts_superpoint_labels[choices_numpy]
            _, pts_superpoint_labels, superpoint_sizes = np.unique(
                pts_superpoint_labels, return_inverse=True, return_counts=True
            )
            input_dict["pts_superpoint_labels"] = pts_superpoint_labels
            input_dict["superpoint_sizes"] = superpoint_sizes

        if "eval_ann_info" in input_dict:
            input_dict["eval_ann_info"]["pts_instance_mask"] = pts_instance_mask
            input_dict["eval_ann_info"]["pts_semantic_mask"] = pts_semantic_mask
            input_dict["eval_ann_info"]["instance_mask"] = instance_mask
            input_dict["eval_ann_info"]["pts_superpoint_labels"] = pts_superpoint_labels

        return input_dict


@TRANSFORMS.register_module()
class CylinderCrop_RemoveOutpoints(BaseTransform):
    def __init__(self, radius=8):
        self.radius = radius

    def transform(self, input_dict: Dict) -> Dict:
        assert "points" in input_dict.keys()

        # Get the tensor of points
        points_tensor = input_dict["points"].tensor.numpy()

        # Select a random center point
        center = points_tensor[np.random.randint(points_tensor.shape[0])]

        # Calculate indices of points within the radius
        choices = np.where(
            (np.sum(np.square(points_tensor[:, :2] - center[:2]), 1) < self.radius**2)
        )[0]

        # Update the points tensor
        if "points" in input_dict.keys():
            input_dict["points"] = input_dict["points"][choices]
        pts_instance_mask = input_dict.get("pts_instance_mask", None)
        pts_semantic_mask = input_dict.get("pts_semantic_mask", None)
        sp_pts_mask = input_dict.get("sp_pts_mask", None)

        # Initialize the instance mask based on semantic labels
        instance_mask = (
            pts_semantic_mask != 0
        )  # Background points have -1 in pts_instance_mask
        pts_instance_mask = pts_instance_mask.copy()
        pts_instance_mask[~instance_mask] = -1

        if pts_instance_mask is not None:
            original_pts_instance_mask = pts_instance_mask
            pts_instance_mask = pts_instance_mask[choices]

            # Initialize vote_label and ratio_inspoint
            vote_label = np.empty((len(choices), 3))
            vote_label[:] = np.nan  # Set initial values to NaN for easier debugging
            ratio_inspoint = {}

            # Find all unique instance indices after cropping
            valid_choices = []
            unique_instances = np.unique(pts_instance_mask)

            for instance in unique_instances:
                if instance != -1:  # Only process non-background instances
                    original_count = np.sum(original_pts_instance_mask == instance)
                    new_count = np.sum(pts_instance_mask == instance)

                    if original_count > 0:
                        ratio = new_count / original_count
                    else:
                        ratio = 0

                    if ratio == 1:  # Only keep fully contained instances
                        ratio_inspoint[instance] = ratio
                        valid_choices.extend(np.where(pts_instance_mask == instance)[0])

                        # Calculate the vote_label for the instance
                        ind = np.where(original_pts_instance_mask == instance)[0]
                        if len(ind) > 0:
                            pos = points_tensor[ind, :3]
                            max_pos = pos.max(0)
                            min_pos = pos.min(0)
                            center = 0.5 * (min_pos + max_pos)

                            # Find the points in the cylinder that belong to this instance
                            cylinder_ind = np.where(pts_instance_mask == instance)[0]
                            vote_label[cylinder_ind, :] = (
                                center - points_tensor[choices[cylinder_ind], :3]
                            )

            # Add background points to valid choices
            background_choices = np.where(pts_instance_mask == -1)[0]
            valid_choices = np.array(valid_choices + background_choices.tolist())

            # Filter instance_mask and create new continuous instance labels
            filtered_instance_mask = pts_instance_mask[valid_choices]
            valid_instance_ids = np.unique(filtered_instance_mask)

            # Create a mapping from original instance IDs to new continuous instance labels, keeping -1 as background
            instance_mapping = {-1: -1}
            non_background_instances = [
                inst for inst in valid_instance_ids if inst != -1
            ]
            instance_mapping.update(
                {
                    old_id: new_id
                    for new_id, old_id in enumerate(non_background_instances)
                }
            )

            # Apply the new mapping to filtered_instance_mask
            new_instance_mask = np.vectorize(instance_mapping.get)(
                filtered_instance_mask
            )

            # Update ratio_inspoint with new continuous instance labels
            new_ratio_inspoint = {
                instance_mapping[inst]: ratio_inspoint[inst]
                for inst in valid_instance_ids
                if inst in ratio_inspoint
            }

            # Update input_dict with valid choices
            input_dict["points"] = input_dict["points"][valid_choices]
            input_dict["pts_instance_mask"] = new_instance_mask
            input_dict["vote_label"] = torch.tensor(
                vote_label[valid_choices], dtype=torch.float32
            )
            input_dict["instance_mask"] = instance_mask[choices][valid_choices]
            input_dict["ratio_inspoint"] = new_ratio_inspoint

        # Update semantic mask if present
        if pts_semantic_mask is not None:
            input_dict["pts_semantic_mask"] = pts_semantic_mask[choices][valid_choices]

        # Update superpoint mask if present
        if sp_pts_mask is not None:
            sp_pts_mask = sp_pts_mask[choices][valid_choices]
            sp_pts_mask = np.unique(sp_pts_mask, return_inverse=True)[1]
            input_dict["sp_pts_mask"] = sp_pts_mask

        # Update evaluation annotation info if present
        if "eval_ann_info" in input_dict:
            input_dict["eval_ann_info"]["pts_instance_mask"] = input_dict[
                "pts_instance_mask"
            ]
            input_dict["eval_ann_info"]["pts_semantic_mask"] = input_dict[
                "pts_semantic_mask"
            ]
            input_dict["eval_ann_info"]["instance_mask"] = input_dict["instance_mask"]

        return input_dict


@TRANSFORMS.register_module()
class GridSample(BaseTransform):
    def __init__(self, grid_size=0.2, mode="train", hash_type="fnv"):
        self.grid_size = grid_size
        self.mode = mode
        self.hash = self._fnv_hash_vec if hash_type == "fnv" else self._ravel_hash_vec

    def transform(self, input_dict: Dict) -> Dict:
        assert "points" in input_dict.keys()
        points = input_dict["points"]

        scaled_points = points.tensor / self.grid_size
        grid_points = torch.floor(scaled_points).int()
        min_points = torch.min(grid_points, dim=0).values
        grid_points -= min_points
        scaled_points -= min_points
        min_points = min_points * self.grid_size

        key = self.hash(grid_points)
        idx_sort = torch.argsort(key)
        key_sort = key[idx_sort]
        unique_results = torch.unique(key_sort, return_inverse=True, return_counts=True)
        _, _, count = unique_results

        if self.mode == "train":  # train mode
            idx_select = (
                torch.cumsum(torch.cat((torch.tensor([0]), count[:-1])), dim=0)
                + torch.randint(0, count.max(), count.size()) % count
            )
            choices = idx_sort[idx_select]
        else:
            raise NotImplementedError("Only train mode is implemented in this example")

        # Subsampled data
        input_dict["points"] = points[choices]

        pts_instance_mask = input_dict.get("pts_instance_mask", None)
        pts_semantic_mask = input_dict.get("pts_semantic_mask", None)
        vote_label = input_dict.get("vote_label", None)
        instance_mask = input_dict.get("instance_mask", None)
        sp_pts_mask = input_dict.get("sp_pts_mask", None)
        pts_superpoint_labels = input_dict.get("pts_superpoint_labels", None)
        pts_center_offsets = input_dict.get("pts_center_offsets", None)

        if pts_instance_mask is not None:
            pts_instance_mask = pts_instance_mask[choices]

            idxs = np.unique(pts_instance_mask)
            mapping = np.zeros(np.max(idxs) + 2, dtype=int)
            new_idxs = np.arange(len(idxs))
            if idxs[0] == -1:
                mapping[idxs] = new_idxs - 1
            else:
                mapping[idxs] = new_idxs
            pts_instance_mask = mapping[pts_instance_mask]

            input_dict["pts_instance_mask"] = pts_instance_mask

        if pts_semantic_mask is not None:
            pts_semantic_mask = pts_semantic_mask[choices]
            input_dict["pts_semantic_mask"] = pts_semantic_mask

        if vote_label is not None:
            vote_label = vote_label[choices]
            input_dict["vote_label"] = vote_label

        if instance_mask is not None:
            instance_mask = instance_mask[choices]
            input_dict["instance_mask"] = instance_mask

        if sp_pts_mask is not None:
            sp_pts_mask = sp_pts_mask[choices]
            sp_pts_mask = np.unique(sp_pts_mask, return_inverse=True)[1]
            input_dict["sp_pts_mask"] = sp_pts_mask

        if pts_superpoint_labels is not None:
            pts_superpoint_labels = pts_superpoint_labels[choices]
            _, pts_superpoint_labels, superpoint_sizes = np.unique(
                pts_superpoint_labels, return_inverse=True, return_counts=True
            )
            input_dict["pts_superpoint_labels"] = pts_superpoint_labels
            input_dict["superpoint_sizes"] = superpoint_sizes

        if pts_center_offsets is not None:
            pts_center_offsets = pts_center_offsets[choices]
            input_dict["pts_center_offsets"] = pts_center_offsets

        if "eval_ann_info" in input_dict:
            input_dict["eval_ann_info"]["pts_instance_mask"] = pts_instance_mask
            input_dict["eval_ann_info"]["pts_semantic_mask"] = pts_semantic_mask
            input_dict["eval_ann_info"]["instance_mask"] = instance_mask
            input_dict["eval_ann_info"]["pts_superpoint_labels"] = pts_superpoint_labels

        return input_dict

    def _fnv_hash_vec(self, vec: Tensor):
        # Use smaller values to avoid overflow issues
        FNV_prime = torch.tensor(16777619, dtype=torch.int64)
        offset_basis = torch.tensor(2166136261, dtype=torch.int64)
        hash = torch.full((vec.shape[0],), offset_basis, dtype=torch.int64)
        for i in range(vec.shape[1]):
            hash = hash ^ vec[:, i].to(torch.int64)
            hash = hash * FNV_prime

        return hash

    def _ravel_hash_vec(self, vec: Tensor):
        # Implement the ravel hash function for vectors
        vec_max = torch.max(vec, dim=0).values + 1
        hash = torch.ravel_multi_index(vec.t(), vec_max)

        return hash


@TRANSFORMS.register_module()
class NormalizePointsReflectance:
    def __init__(
        self,
        coords_dim: int = 3,
        method: str = "iqr_minmax",
        skip_flag: bool = False,
        epsilon: float = 1e-8,
    ):
        self.coords_dim = coords_dim
        self.method = method
        self.skip_flag = skip_flag
        # When skip flag is set, normalization is skipped completely. This is useful when running
        # inference on "bluepoints" where the data data is already normalized after the first iteration
        self.epsilon = epsilon

    def __call__(self, input_dict: Dict) -> Dict:
        points = input_dict["points"].tensor
        num_dims = points.shape[1]
        # Check whether input data contains reflectance and whether skip flag is set
        if (num_dims > self.coords_dim) and not self.skip_flag:
            reflectance = points[:, self.coords_dim :]

            if self.method == "minmax":
                reflectance = self._normalize_minmax(reflectance)
            elif self.method == "iqr":
                reflectance = self._normalize_iqr(reflectance)
            elif self.method == "iqr_minmax":
                reflectance = self._normalize_iqr(reflectance)
                reflectance = self._normalize_minmax(reflectance)
            else:
                raise NotImplementedError(
                    f"Reflectance normalization method '{self.method}' not implemented!"
                )
            points[:, self.coords_dim :] = reflectance

            input_dict["points"] = type(input_dict["points"])(
                points, points_dim=num_dims
            )

        return input_dict

    def _normalize_minmax(self, reflectance: Tensor) -> Tensor:
        min_reflectance, _ = torch.min(reflectance, dim=0)
        max_reflectance, _ = torch.max(reflectance, dim=0)
        reflectance = (reflectance - min_reflectance) / (
            max_reflectance - min_reflectance + self.epsilon
        )

        return reflectance

    def _normalize_iqr(self, reflectance: Tensor) -> Tensor:
        reflectance_q1 = torch.quantile(reflectance, q=0.25, dim=0)
        reflectance_q3 = torch.quantile(reflectance, q=0.75, dim=0)
        median, _ = torch.median(reflectance, dim=0)
        reflectance_iqr = reflectance_q3 - reflectance_q1
        reflectance = (reflectance - median) / (reflectance_iqr + self.epsilon)

        return reflectance
