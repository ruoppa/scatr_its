import os
import laspy

import numpy as np

from plyfile import PlyElement
from plyfile import PlyData as PlyData_
from typing import Union, List, Optional, Tuple, Dict

_MAX_LAZ_SIZE = 40_000_000


# Wrapper for objects read with plyfile
class PlyData:
    def __init__(self, data: Union[PlyData_, PlyElement, np.ndarray]):
        if isinstance(data, PlyData_):
            self._data = data["vertex"].data
        elif isinstance(data, PlyElement):
            self._data = data.data
        else:
            self._data = data

    @property
    def xyz(self):
        return np.vstack(
            [
                self._data["x"],
                self._data["y"],
                self._data["z"],
            ]
        ).T

    def __getitem__(self, index: Union[int, np.ndarray, str]):
        """Return a portion of the PlyData object indicated by indices. The returned object is a copy, i.e. any operations
        done to it do not affect the original PlyData object. Alternatively, if the index is a string, return the corresponding
        field in the data

        Args:
            index (Union[int, np.ndarray, str]): indices of PlyData object to return or the name of the field which to return

        Returns:
            Any: new PlyData or field
        """
        if isinstance(index, int):
            index = [index]
        if isinstance(index, str):
            if index in self._data.dtype.names:
                return self._data[index]
        else:
            new_data = self._data[index].copy()
            return PlyData(new_data)

    def __setitem__(self, key: Union[str, List[int], np.ndarray], value):
        if isinstance(key, str):
            if key in self._data.dtype.names:
                self._data[key] = value
                return
            raise KeyError(f"'PlyData' object has no attribute '{key}'")

        try:
            self._data[key] = value
        except Exception as e:
            raise TypeError(f"Invalid index assignment: {e}")

    def __getattr__(self, __name: str):
        # If we try to access an attribute that is not a property of the class, try finding the attribute from the extra fields instead
        try:
            return self._data[__name]
        except ValueError:
            raise AttributeError(f"'PlyData' object has no attribute '{__name}'")

    def __setattr__(self, __name, value):
        # Internal attributes should be set normally
        if __name.startswith("_"):
            object.__setattr__(self, __name, value)
            return
        # Try to set PLY field
        try:
            self._data[__name] = value
        except Exception:
            raise AttributeError(f"'PlyData' object has no attribute '{__name}'")


def field_exists(pc: Union[laspy.LasData, PlyData], field_name: str) -> bool:
    """Check if a field exists in a point cloud

    Args:
        pc (Union[laspy.LasData, PlyData]): point cloud
        field_name (str): name of field to check

    Returns:
        bool: boolean indicating whether the field exists
    """
    if isinstance(pc, PlyData):
        exists = field_name in pc._data.dtype.names
    else:
        exists = (field_name in pc.point_format.dimension_names) or (
            field_name in pc.point_format.extra_dimension_names
        )

    return exists


def read_pc(
    path: str,
    file_type: str,
    instance_pred_field: Optional[str] = None,
    instance_gt_field: Optional[str] = None,
    semantic_gt_field: Optional[str] = None,
    ignore_classes: Optional[List[int]] = [],
    filter_small_instances: bool = False,
    min_instance_points: Optional[int] = None,
    min_instance_h: Optional[float] = None,
    semantic_class_mapping: Optional[Dict[int, int]] = None,
) -> Union[laspy.LasData, PlyData]:
    """Read point cloud from las, laz or ply file and perform necessary preprocessing based on configuration file

    Args:
        path (str): path to point cloud file
        file_type (str): file type of point cloud. Supported: las, laz, ply
        instance_pred_field (Optional[str]): name of the field with predicted instance labels
        instance_gt_field (Optional[str]): name of the field with ground truth instance labels
        semantic_gt_field (Optional[str]): name of the field with ground truth semantic labels
        ignore_classes (List[int]): list of semantic classes to ignore. Defaults to [].
        filter_small_instances (bool): if True, remove small predicted instance. Defaults to False.
        min_instance_points (Optional[int]): minimum number of points an instance must contain. Defaults to None.
        min_instance_h (Optional[float]): minimum height of instance (absolute difference in min/max z-coordinates). Defaults to None.

    Returns:
        Union[laspy.LasData, PlyData]: preprocessed point cloud
    """
    if file_type == "ply":
        pc = _read_ply(path)
    else:  # las or laz
        pc = laspy.read(path)
    # Preprocess the gt and prediction instance fields such that 0 = non-tree
    # The function makes the following assumptions
    #   1. Smallest value in the field = non-tree
    #   2. If smallest value in the field > 0, then there are no non-tree points
    if instance_gt_field is not None:
        instance_gt_min = np.min(pc[instance_gt_field])
        if instance_gt_min <= 0:
            pc[instance_gt_field] += np.abs(instance_gt_min)
    if instance_pred_field is not None:
        instance_pred_min = np.min(pc[instance_pred_field])
        if instance_pred_min <= 0:
            pc[instance_pred_field] += np.abs(instance_pred_min)
    # Remove points in ignored classes
    if semantic_gt_field is not None:
        if ignore_classes is not None:
            keep_mask = ~np.isin(pc[semantic_gt_field], ignore_classes)
            pc = pc[keep_mask]
        if semantic_class_mapping is not None:
            for original_class, new_class in semantic_class_mapping.items():
                pc[semantic_gt_field][pc[semantic_gt_field] == original_class] = (
                    new_class
                )
    # Map semantic classes
    # Filter out small predicted instances, if necessary
    if filter_small_instances:
        pc = _filter_predicted_instances(
            pc, instance_pred_field, min_instance_points, min_instance_h
        )

    return pc


def _filter_predicted_instances(
    pc: Union[laspy.LasData, PlyData],
    instance_pred_field: str,
    min_instance_points: int,
    min_instance_h: float,
) -> Union[laspy.LasData, PlyData]:
    """Remove small predicted instances that are unlikely to be trees. Instances are removed based on two criterion: 1. number of points, 2. height, i.e. difference between min and max z-coordinates

    Args:
        pc (Union[laspy.LasData, PlyData]): point cloud to filter
        instance_pred_field (str): name of the field with predicted instance labels
        min_instance_points (int): minimum number of points an instance must contain
        min_instance_h (float): minimum height of instance (absolute difference in min/max z-coordinates)

    Returns:
        Union[laspy.LasData, PlyData]: point cloud from which the small predicted instances have been removed
    """
    preds = pc[instance_pred_field]
    z_coords = pc.z
    # Group by instance ID and calculate number of points
    unique_ids, inverse_indices = np.unique(preds, return_inverse=True)
    counts = np.bincount(inverse_indices)
    # Compute instance heights
    min_z = np.full(unique_ids.shape, np.inf)
    max_z = np.full(unique_ids.shape, -np.inf)
    np.minimum.at(min_z, inverse_indices, z_coords)
    np.maximum.at(max_z, inverse_indices, z_coords)
    heights = max_z - min_z
    # Filter out small instances
    keep_mask = (counts >= min_instance_points) & (heights >= min_instance_h)
    keep_ids = unique_ids[keep_mask]
    points_keep_mask = np.isin(preds, keep_ids)
    pc[instance_pred_field][~points_keep_mask] = 0

    return pc


def _read_ply(path: str) -> PlyData:
    ply = PlyData_.read(path)
    ply = PlyData(ply)

    return ply


def write_pc(
    path: str,
    file_type: str,
    points: np.ndarray,
    semantic_pred: np.ndarray,
    semantic_pred_field: str,
    instance_pred: np.ndarray,
    instance_pred_field: str,
    scores: np.ndarray,
    score_pred_field: str,
    extra_features: Optional[Dict[int, str]] = None,
    offsets: Optional[np.ndarray] = None,
    scales: Optional[np.ndarray] = None,
    semantic_gt: Optional[np.ndarray] = None,
    semantic_gt_field: Optional[str] = None,
    instance_gt: Optional[np.ndarray] = None,
    instance_gt_field: Optional[np.ndarray] = None,
) -> None:
    """Write a point cloud to file

    Args:
        path (str): path for saving the point cloud
        file_type (str): file type for point cloud (supported: las, laz, ply)
        points (np.ndarray): point array [N, 3] or [N, 3 + F], where F is the number of extra features
        semantic_pred (np.ndarray): semantic prediction for each point
        semantic_pred_field (str): name of semantic prediction field in the result
        instance_pred (np.ndarray): instance prediction for each point
        instance_pred_field (str): name of instance prediction field in the result
        scores (np.ndarray): instance score prediction for each point
        score_pred_field (str): name of instance score prediction field in the result
        extra_features (Optional[Dict[int, str]], optional): List of extra features in the data (column index in points, feature name). Defaults to None.
        offsets (Optional[np.ndarray], optional): xyz offset applied to coordinates before saving. Defaults to None.
        scales (Optional[np.ndarray], optional): scales for saving data. Only used for las/laz files. Defaults to None.
        semantic_gt (Optional[np.ndarray], optional): semantic ground truth for each point. Defaults to None.
        semantic_gt_field (Optional[str], optional): name of semantic ground truth field in the result. Defaults to None.
        instance_gt (Optional[np.ndarray], optional): instance ground truth for each point. Defaults to None.
        instance_gt_field (Optional[np.ndarray], optional): name of instance ground truth field in the result. Defaults to None.

    Raises:
        NotImplementedError: raised if file type not supported
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if file_type in ["las", "laz"]:
        _save_las(
            path,
            points,
            offsets,
            semantic_pred,
            semantic_pred_field,
            instance_pred,
            instance_pred_field,
            scores,
            score_pred_field,
            extra_features,
            scales,
            semantic_gt,
            semantic_gt_field,
            instance_gt,
            instance_gt_field,
        )
    elif file_type == "ply":
        _save_ply(
            path,
            points,
            offsets,
            semantic_pred,
            semantic_pred_field,
            instance_pred,
            instance_pred_field,
            scores,
            score_pred_field,
            extra_features,
            semantic_gt,
            semantic_gt_field,
            instance_gt,
            instance_gt_field,
        )
    else:
        raise NotImplementedError(
            f"Saving point clouds in file format '{file_type}' is unsupported!"
        )


def _save_las(
    path: str,
    points: np.ndarray,
    offsets: Optional[np.ndarray],
    semantic_pred: np.ndarray,
    semantic_pred_field: str,
    instance_pred: np.ndarray,
    instance_pred_field: str,
    scores: np.ndarray,
    score_pred_field: str,
    extra_features: Optional[Dict[int, str]] = None,
    scales: Optional[np.ndarray] = None,
    semantic_gt: Optional[np.ndarray] = None,
    semantic_gt_field: Optional[str] = None,
    instance_gt: Optional[np.ndarray] = None,
    instance_gt_field: Optional[np.ndarray] = None,
) -> None:
    xyz_coords = points[:, :3].astype(np.float64)
    # Cast back to original coordinate reference frame by adding offsets (if available)
    if offsets is not None:
        xyz_coords += offsets
    header = laspy.LasHeader(point_format=7, version="1.4")
    # 1st row contains scales and second row contains las/laz offsets (if such were available in the original data)
    las_offsets = scales[1, :].copy()
    las_scales = scales[0, :].copy()
    # Compute safe scale and offset for data
    las_scales, las_offsets = _get_las_offsets_and_scales(xyz_coords, las_scales, las_offsets)
    header.scales = las_scales
    header.offsets = las_offsets
    # Create a new LasData object
    las = laspy.LasData(header)
    las.x = xyz_coords[:, 0]
    las.y = xyz_coords[:, 1]
    las.z = xyz_coords[:, 2]
    fields_to_add = []
    values_to_add = []

    # Collect extra fields and values that need to be added to the laspy object before saving
    if extra_features is not None:
        n_extra_features = len(extra_features)
        missing_fields = _get_params_if_field_not_exist(
            las, list(extra_features.values()), [np.float32] * n_extra_features
        )
        fields_to_add.extend(missing_fields)
        for i, name in extra_features.items():
            values_to_add.append((name, points[:, i]))

    missing_fields = _get_params_if_field_not_exist(
        las,
        [semantic_pred_field, instance_pred_field, score_pred_field],
        [np.uint16, np.uint16, np.float32],
    )
    fields_to_add.extend(missing_fields)
    values_to_add.extend(
        [
            (semantic_pred_field, semantic_pred.astype(np.uint16)),
            (instance_pred_field, instance_pred.astype(np.uint16)),
            (score_pred_field, scores),
        ]
    )
    if semantic_gt is not None:
        fields_to_add.extend(_get_params_if_field_not_exist(las, semantic_gt_field, np.uint16))
        values_to_add.append((semantic_gt_field, semantic_gt.astype(np.uint16)))
    if instance_gt is not None:
        fields_to_add.extend(_get_params_if_field_not_exist(las, instance_gt_field, np.uint16))
        values_to_add.append((instance_gt_field, instance_gt.astype(np.uint16)))
    # Add collected extra fields and values
    _add_las_fields_and_values(las, fields_to_add, values_to_add)

    if (len(las.x) > _MAX_LAZ_SIZE) and (path[-3:] == "laz"):
        # laz files become corrupted with large file sizes, switch to las
        path = f"{path[:-1]}s"

    las.write(path)


def _get_params_if_field_not_exist(
    las: laspy.LasData,
    field: Union[str, List[str]],
    dtype: Union[np.dtype, List[np.dtype]],
) -> List[laspy.ExtraBytesParams]:
    params_list = []
    if not isinstance(field, List):
        field = [field]
    if not isinstance(dtype, List):
        dtype = [dtype]
    for f_i, d_i in zip(field, dtype):
        if not field_exists(las, f_i):
            params_list.append(laspy.ExtraBytesParams(f_i, type=d_i))

    return params_list


def _add_las_fields_and_values(
    las: laspy.LasData,
    fields_to_add: List[laspy.ExtraBytesParams],
    values_to_add: List[Tuple[str, np.ndarray]],
) -> None:
    las.add_extra_dims(fields_to_add)
    for field_name, field_values in values_to_add:
        las[field_name] = field_values


def _get_las_offsets_and_scales(
    points: np.ndarray,
    scales: np.ndarray,
    offsets: np.ndarray,
    max_int: int = 2**31 - 1,
    min_scale: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    # Negative scales -> original data did not have scales, compute them. Otherwise, use original scales and offsets
    if np.any(scales <= 0):
        min_vals = points.min(axis=0)
        max_vals = points.max(axis=0)
        scales = (max_vals - min_vals) / max_int
        scales = np.maximum(scales, min_scale)
        offsets = min_vals

    return scales, offsets


def _save_ply(
    path: str,
    points: np.ndarray,
    offsets: Optional[np.ndarray],
    semantic_pred: np.ndarray,
    semantic_pred_field: str,
    instance_pred: np.ndarray,
    instance_pred_field: str,
    scores: np.ndarray,
    score_pred_field: str,
    extra_features: Optional[Dict[int, str]] = None,
    semantic_gt: Optional[np.ndarray] = None,
    semantic_gt_field: Optional[str] = None,
    instance_gt: Optional[np.ndarray] = None,
    instance_gt_field: Optional[np.ndarray] = None,
) -> None:
    # Cast back to original coordinate reference frame by adding offsets (if available)
    if offsets is not None:
        points[:, :3] += offsets
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
    ]
    if extra_features is not None:
        points_xyz = points[:, :3].copy()
        for i, name in extra_features.items():
            dtype.append((name, "f4"))
            points_xyz = np.hstack((points_xyz, points[:, [i]]))
        points = points_xyz
    else:
        points = points[:, :3]

    dtype.extend(
        [
            (semantic_pred_field, "u2"),
            (instance_pred_field, "u2"),
            (score_pred_field, "f4"),
        ]
    )
    if semantic_gt is not None:
        dtype += [(semantic_gt_field, "u2")]
    if instance_gt is not None:
        dtype += [(instance_gt_field, "u2")]
    # Generate data array
    n_points = points.shape[0]
    if (semantic_gt is not None) and (instance_gt is not None):
        vertex = np.array(
            [
                tuple(points[i])
                + (
                    semantic_pred[i],
                    instance_pred[i],
                    scores[i],
                    semantic_gt[i],
                    instance_gt[i],
                )
                for i in range(n_points)
            ],
            dtype=dtype,
        )
    elif semantic_gt is not None:
        vertex = np.array(
            [
                tuple(points[i])
                + (
                    semantic_pred[i],
                    instance_pred[i],
                    scores[i],
                    semantic_gt[i],
                )
                for i in range(n_points)
            ],
            dtype=dtype,
        )
    elif instance_gt is not None:
        vertex = np.array(
            [
                tuple(points[i])
                + (
                    semantic_pred[i],
                    instance_pred[i],
                    scores[i],
                    semantic_gt[i],
                )
                for i in range(n_points)
            ],
            dtype=dtype,
        )
    else:
        vertex = np.array(
            [
                tuple(points[i])
                + (
                    semantic_pred[i],
                    instance_pred[i],
                    scores[i],
                )
                for i in range(n_points)
            ],
            dtype=dtype,
        )

    elements = PlyElement.describe(vertex, "vertex")
    PlyData_([elements], text=True).write(path)
