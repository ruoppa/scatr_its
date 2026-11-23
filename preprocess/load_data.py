import os
import time
import shutil
import laspy

import numpy as np

from mmengine.config import Config
from mmengine.logging import MMLogger, print_log
from plyfile import PlyData
from typing import List, Optional, Dict, Tuple, Union


def export_scans(
    data_config: Config,
    unlabeled_mode: bool = False,
    force_rewrite: bool = False,
    superpoint_mode: bool = False,
    logger: Optional[MMLogger] = None,
) -> None:
    """Preprocess all data splits

    Args:
        data_root_path (str): Data root path.
        dataset_name (str): Name of dataset to preprocess
        preprocess_config (Config): Preprocessing configuration containt the necessary parameters for _export_one_scan().
        data_split_config (Config): Configuration containing the data splits.
        unlabeled_mode (bool, optional): If True, process in unlabeled mode (no labels required). Defaults to False.
        force_rewrite (bool, optional): If True, preprocess the file again, even if a preprocessed version already exists. Defaults to False.
        superpoint_mode (bool, optional): If True, include superpoints in preprocessing (superpoints must have been generated first). Defaults to False.
        superpoint_data_dir (Optional[str]): Name of directory containing generated superpoint labels. Defaults to None.
    """
    for split in data_config.data_split.keys():
        split_start_time = time.time()
        print_log(f"==> Preprocessing '{split}' split", logger=logger)
        raw_data_split_path = data_config.preprocess[f"{split}_dir"]
        if superpoint_mode:
            superpoint_path = data_config.output.superpoint_data_dir
        preprocessed_data_path = data_config.preprocess.preprocessed_data_dir
        if not os.path.exists(preprocessed_data_path):
            os.makedirs(preprocessed_data_path)
        n_files = len(data_config.data_split[split])
        if n_files > 0:
            skip_count = 0
            for i, file in enumerate(data_config.data_split[split], start=1):
                status = f"==> Preprocessing '{file}' [{i}/{n_files}]"
                print(f"\r{status.ljust(100)}", end="", flush=True)
                file_path = os.path.join(raw_data_split_path, file)
                output_path = os.path.join(preprocessed_data_path, file)
                if superpoint_mode:
                    superpoint_file_path = os.path.join(superpoint_path, f"{file}.npy")
                    if (
                        not os.path.isfile(f"{output_path}_superpoint_labels.npy")
                        or force_rewrite
                    ):
                        _export_superpoints(
                            superpoint_file_path,
                            output_path,
                        )
                        # No need to do the scan exporting in superpoint mode even when force_rewrite is True,
                        # since the first round of preprocessing before superpoint generation has already handled it
                        continue 
                if os.path.isfile(f"{output_path}_points.npy") and not force_rewrite:
                    status = f"==> '{file}' has already been preprocessed! Skipping. [{i}/{n_files}]"
                    print(f"\r{status.ljust(100)}", end="", flush=True)
                    skip_count += 1
                    continue
                # Preprocess file
                _export_one_scan(
                    file_path,
                    output_filename=output_path,
                    unlabeled_mode=unlabeled_mode,
                    **data_config.preprocess,
                )
            print()
            split_end_time = time.time()
            status = f"==> Finish preprocessing '{split}' split in {(split_end_time - split_start_time):.3f} s!"
            if skip_count > 0:
                status += f" (Skipped [{skip_count}/{n_files}])"
            print_log(status, logger=logger)
        else:
            print_log(f"==> '{split}' split is empty!", logger=logger)


def _extract_bbox(
    mesh_vertices: np.ndarray,
    instance_labels: np.ndarray,
    non_instance_id: int,
    return_centers: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Extract bounding boxes from given vertices

    Args:
        mesh_vertices (np.ndarray): vertices of the mesh (i.e. points of the point cloud)
        instance_labels (np.ndarray): instance labels for each point
        non_instance_id (int): id of points that are not part of any instance
        return_centers (bool, optional): if True, also return object/boundix box center positions as a separate value. Defaults to False.

    Returns:
        np.ndarray: bounding boxes for each instance
    """
    valid_mask = instance_labels != non_instance_id
    mesh_vertices = mesh_vertices[valid_mask]
    instance_labels = instance_labels[valid_mask]
    if len(instance_labels) == 0:
        if return_centers:
            return np.zeros((0, 7)), np.zeros((0, 3))
        else:
            return np.zeros((0, 7))
    # Sort points by instance label (avoids masking since instances can be processed in order)
    order = np.argsort(instance_labels)
    mesh_vertices = mesh_vertices[order]
    instance_labels = instance_labels[order]
    # Fetch first index of each instance
    unq_instance_labels, start_ind = np.unique(instance_labels, return_index=True)
    n_instances = len(unq_instance_labels)
    instance_bboxes = np.zeros((n_instances, 7))

    for i in range(n_instances):
        instance_start_ind = start_ind[i]
        instance_end_ind = (
            start_ind[i + 1] if i + 1 < n_instances else len(instance_labels)
        )
        instance_points = mesh_vertices[instance_start_ind:instance_end_ind]
        min_points = instance_points.min(axis=0)
        max_points = instance_points.max(axis=0)
        location = (min_points + max_points) / 2
        dimensions = max_points - min_points
        instance_bboxes[i, :3] = location
        instance_bboxes[i, 3:6] = dimensions
        # dim 6 for labels -> all trees are in instance class 1
        instance_bboxes[i, 6] = 1

    if return_centers:
        instance_centers = np.zeros((n_instances, 4))
        instance_centers[:, :3] = instance_bboxes[:, :3].copy()
        instance_centers[:, 3] = unq_instance_labels

        return instance_bboxes, instance_centers

    return instance_bboxes


def _export_one_scan(
    file_path: str,
    instance_field: str,
    semantic_field: str,
    file_extension: str,
    non_instance_id: int,
    feat_to_load: Optional[List[str]] = None,
    output_filename: Optional[str] = None,
    ignore_semantic_classes: Optional[List[int]] = None,
    semantic_class_mapping: Optional[Dict[int, int]] = None,
    max_n_points: Optional[int] = None,
    unlabeled_mode: bool = False,
    **kwargs,
) -> Optional[
    Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
]:
    """Preprocess a single point cloud into separate point, label and bounding box files

    Args:
        file_path (str): path to file to process
        instance_field (str): name of the instance field in the data
        semantic_field (str): name of the semantic classification field in the data
        file_extension (str): filetype extension of the raw data
        non_instance_id (int): id of points that are not part of any instance
        feat_to_load (Optional[List[str]]): names of extra features to load. Defaults to None.
        output_filename (Optional[str]): path to output folder (if not provided, files not saved). Defaults to None.
        ignore_semantic_classes (Optional[List[int]]): list of semantic classes to ignore. Defaults to None.
        semantic_class_mapping (Optional[Dict[int, int]]): mapping for semantic classes (old class -> new class). Defaults to None.
        max_n_points (Optional[int]): maximum number of points per instance. If over this threshold, data is downsampled using random sampling. Defaults to None.
        unlabeled_mode (bool, optional): If True, process in unlabeled mode (no labels required). Defaults to False.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,]]: Preprocessed data. Only returned if output file was not specified.
    """
    if file_extension == "ply":
        plydata = PlyData.read(f"{file_path}.{file_extension}")
        pc = plydata["vertex"]
        # No scale/offset data for ply files
        scales = -np.ones((2, 3), dtype=np.float64)
    else:  # las or laz
        pc = laspy.read(f"{file_path}.{file_extension}")
        # Fetch las scales and offsets
        scales = np.vstack([pc.header.scales, pc.header.offset])
    points = np.empty((len(pc["x"]), 3), dtype=np.float64)
    points[:, 0] = np.asarray(pc["x"])
    points[:, 1] = np.asarray(pc["y"])
    points[:, 2] = np.asarray(pc["z"])
    # Load additional features
    features = None
    if (feat_to_load is not None) and (len(feat_to_load) > 0):
        features = []
        for feat_name in feat_to_load:
            try:
                features.append(np.asarray(pc[feat_name]))
            except Exception:
                raise ValueError(f"No field of name '{feat_name}'")
        features = np.vstack(features).astype(np.float32).T

    # float64 for offsets for higher precision
    mean_x = np.mean(points[:, 0])
    mean_y = np.mean(points[:, 1])
    min_z = np.min(points[:, 2])
    offsets = np.array([mean_x, mean_y, min_z])
    points[:, 0] -= mean_x
    points[:, 1] -= mean_y
    points[:, 2] -= min_z

    points = points.astype(np.float32)
    # No alignment matrices necessary, use identity matrix
    axis_align_matrix = np.eye(4)

    if not unlabeled_mode:
        semantic_labels = np.asarray(pc[semantic_field]).astype(np.int64)
        instance_labels = np.asarray(pc[instance_field]).astype(np.int64)

        if ignore_semantic_classes is not None:
            keep_mask = ~np.isin(semantic_labels, ignore_semantic_classes)
            points = points[keep_mask, :]
            if features is not None:
                features = features[keep_mask, :]
            semantic_labels = semantic_labels[keep_mask]
            instance_labels = instance_labels[keep_mask]
        if semantic_class_mapping is not None:
            for original_class, new_class in semantic_class_mapping.items():
                semantic_labels[semantic_labels == original_class] = new_class
        # Set non-instance label to 0
        instance_labels[instance_labels == non_instance_id] = 0
    else:
        semantic_labels = np.zeros(points.shape[0], dtype=np.int64)
        instance_labels = np.zeros(points.shape[0], dtype=np.int64)

    if max_n_points is not None:
        n_points = points.shape[0]
        if n_points > max_n_points:
            choices = np.random.permutation(n_points)[:max_n_points]
            points = points[choices, :]
            semantic_labels = semantic_labels[choices]
            instance_labels = instance_labels[choices]
    # Extract bounding boxes (unaligned and aligned are the same in this case)
    unaligned_bboxes, instance_centers = _extract_bbox(points, instance_labels, 0, return_centers=True)
    aligned_bboxes = unaligned_bboxes
    # Concatenate xyz-coordinates with other features
    if features is not None:
        points = np.concatenate((points, features), axis=1)

    if output_filename is not None:
        np.save(f"{output_filename}_points.npy", points)
        np.save(f"{output_filename}_offsets.npy", offsets)
        # NOTE: in unlabeled mode the annotations are empty
        np.save(f"{output_filename}_semantic_labels.npy", semantic_labels)
        np.save(f"{output_filename}_instance_labels.npy", instance_labels)
        np.save(f"{output_filename}_unaligned_bboxes.npy", unaligned_bboxes)
        np.save(f"{output_filename}_aligned_bboxes.npy", aligned_bboxes)
        np.save(f"{output_filename}_axis_align_matrix.npy", axis_align_matrix)
        np.save(f"{output_filename}_instance_centers.npy", instance_centers)
        # If not las data, scales filled with -1
        np.save(f"{output_filename}_scales.npy", scales)
    else:
        return (
            points,
            semantic_labels,
            instance_labels,
            unaligned_bboxes,
            aligned_bboxes,
            axis_align_matrix,
            instance_centers,
            offsets,
            scales,
        )


def _export_superpoints(
    file_path: str,
    output_filename: str,
) -> Optional[np.ndarray]:
    shutil.copy(file_path, f"{output_filename}_superpoint_labels.npy")
