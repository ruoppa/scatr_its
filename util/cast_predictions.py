import os
import time
import laspy
import argparse

import numpy as np
import numpy.lib.recfunctions as rfn

from tqdm import tqdm
from typing import List, Union, Dict, Optional
from util import read_pc, load_config
from scipy.spatial import cKDTree
from mmengine.config import Config
from mmengine.logging import MMLogger, print_log
from plyfile import PlyData, PlyElement

_MAX_LAZ_SIZE = 40_000_000


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Model config file path.",
    )
    parser.add_argument(
        "--data_config",
        required=True,
        help="Dataset config file path",
    )
    parser.add_argument(
        "--pipeline_config",
        default="configs/pipelines",
        help="Pipeline template config file path.",
    )
    parser.add_argument(
        "--prec",
        type=int,
        default=3,
        help="Decimal precision used when casting predictions",
    )
    parser.add_argument(
        "--force_nn",
        action="store_true",
        help="Force using nearest neighbor search for casting. Can be extremely slow/infeasible for large point clouds",
    )
    parser.add_argument(
        "--unlabeled_mode",
        "-u",
        action="store_true",
        help="Set if data has no ground truth labels.",
    )
    args = parser.parse_args()
    args.prec = 10 ** (-args.prec)

    return args


def cast_all(
    config: Union[str, None],
    data_config: Union[str, Config],
    pipeline_config: Union[str, None] = None,
    unlabeled_mode: bool = False,
    force_nn: bool = False,
    prec: int = 1e-3,
    logger: Optional[MMLogger] = None,
) -> None:
    if not isinstance(data_config, Config):
        config, data_config = load_config(
            data_config_path=data_config,
            model_config_path=config,
            pipeline_template_path=pipeline_config,
            test_mode=True,
            superpoint_mode=False,
        )
    print_log("==> Casting predictions to original input point clouds", logger=logger)
    for filename in tqdm(data_config.data_split.test, desc="==> Processing files"):
        pred_pc_path = os.path.join(
            data_config.output.output_data_dir,
            f"{filename}.{data_config.output.file_extension}",
        )
        orig_pc_path = os.path.join(
            data_config.preprocess.test_dir,
            f"{filename}.{data_config.preprocess.file_extension}",
        )
        cast_predictions_to_original_pc(
            pred_pc_path,
            orig_pc_path,
            data_config.output.instance_pred_field,
            data_config.output.score_pred_field,
            data_config.output.semantic_pred_field,
            data_config.preprocess.semantic_field,
            data_config.output.file_extension,
            data_config.preprocess.file_extension,
            unlabeled_mode,
            data_config.preprocess.ignore_semantic_classes,
            force_nn,
            prec,
        )


def cast_predictions_to_original_pc(
    pred_pc_path: str,
    orig_pc_path: str,
    instance_pred_field: str,
    score_field: str,
    semantic_pred_field: str,
    semantic_gt_field: str,
    pred_file_type: str,
    orig_file_type: str,
    unlabeled_mode: bool = False,
    ignore_classes: List[int] = [],
    force_nn: bool = False,
    prec: int = 1e-3,
) -> None:
    """Cast predictions from point cloud with instance and semantic predictions to the corresponding
    original point cloud (this is useful when the tree segments are going to be used in some downstream
    task, and the original data includes additional features not present in the prediction point cloud,
    which only includes xyz-coordinates, any additional input features, and the predictions/ground truth.
    While janky, due to the way the mmdet3d library functions, this is the most practical way...)

    NOTE: the two point clouds should be in the same coordinates (e.g. no offset)

    Args:
        pred_pc_path (str): path to the point cloud with predictions
        orig_pc_path (str): path to the original point cloud
        instance_pred_field (str): name of field with instance predictions
        score_field (str): name of field with instance scores
        semantic_pred_field (str): name of field with semantic predictions
        semantic_gt_field (str): name of field with semantic ground truth
        pred_file_type (str): file type of point cloud with predictions
        orig_file_type (str): file type of original point cloud
        ignore_classes (List[int], optional): list of semantic ground truth classes to ignore. Defaults to [].
        force_nn (bool, optional): if True, force using KD-Tree nearest neighbor search for casting predictions (can be extremely slow, or even infeasible for large point clouds).
        prec (int, optional)) decimal precision for matching points based on coordinate values. Defaults to 4.
    """
    pred_pc = read_pc(pred_pc_path, pred_file_type)
    # Ignored classes are removed from original pc, since prediction pc does not contain those points
    if unlabeled_mode:
        semantic_gt_field = None
        ignore_classes = []
    orig_pc = read_pc(
        orig_pc_path,
        orig_file_type,
        semantic_gt_field=semantic_gt_field,
        ignore_classes=ignore_classes,
    )
    pred_xyz = pred_pc.xyz.astype(np.float32)
    orig_xyz = orig_pc.xyz.astype(np.float32)
    if orig_file_type == "ply":
        orig_pc = orig_pc._data
    if force_nn:
        map_idx = _chunked_knn_search(pred_xyz, orig_xyz)
    else:
        map_idx = _voxelized_hash_matching(pred_xyz, orig_xyz, prec)
    # Map predicted instance IDs to the original points
    instance_preds = pred_pc[instance_pred_field][map_idx].astype(np.uint16)
    scores = pred_pc[score_field][map_idx].astype(np.float32)
    semantic_preds = pred_pc[semantic_pred_field][map_idx].astype(np.uint16)
    # Save results (in the same path as predictions, but file type of original point cloud. If file types are the same -> prediction overriden)
    save_path = f"{pred_pc_path[: -(len(pred_file_type) + 1)]}.{orig_file_type}"
    if orig_file_type == "ply":
        _save_to_ply(
            orig_pc,
            save_path,
            instance_pred_field,
            score_field,
            semantic_pred_field,
            instance_preds,
            scores,
            semantic_preds,
        )
    else:
        _save_to_las(
            orig_pc,
            save_path,
            instance_pred_field,
            score_field,
            semantic_pred_field,
            instance_preds,
            scores,
            semantic_preds,
        )


def _chunked_knn_search(
    pred_xyz: np.ndarray, orig_xyz: np.ndarray, chunk_size: int = 2_000_000
):
    ckdtree = cKDTree(pred_xyz)
    M = len(orig_xyz)

    map_idx = np.empty(M, dtype=np.int64)
    # Process data in chunks to keep RAM usage reasonable
    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        _, idx = ckdtree.query(orig_xyz[start:end], k=1)
        map_idx[start:end] = idx

    return map_idx


def _hash_xyz(q: np.ndarray) -> np.ndarray:
    return (
        (q[:, 0].astype(np.int64) << 42)
        ^ (q[:, 1].astype(np.int64) << 21)
        ^ q[:, 2].astype(np.int64)
    )


def _find_neighbor(
    orig_xyz_vox: np.ndarray, ind: int, offsets: np.ndarray, lookup: Dict[int, int]
):
    q = orig_xyz_vox[ind]
    for offset in offsets:
        h = _hash_xyz((q + offset).reshape(1, 3))[0]
        if h in lookup:
            return lookup[h]

    return -1


def _voxelized_hash_matching(
    pred_xyz: np.ndarray, orig_xyz: np.ndarray, prec: float = 1e-3
):
    # Quantize coordinates
    pred_xyz_vox = np.floor(pred_xyz / prec).astype(np.int32)
    orig_xyz_vox = np.floor(orig_xyz / prec).astype(np.int32)
    # Create hash by packing each xyz coordinate into a single int64 key
    pred_xyz_hash = _hash_xyz(pred_xyz_vox)
    orig_xyz_hash = _hash_xyz(orig_xyz_vox)
    # Build lookup table for B
    lookup = {h: i for i, h in enumerate(pred_xyz_hash)}
    # Initial matches
    map_idx = np.array([lookup.get(h, -1) for h in orig_xyz_hash], dtype=np.int64)
    # Corresponding points may quantize into different bins near edges resulting in failed get(). Fix this by checking neighboring voxels
    offsets = np.array(
        [
            [dx, dy, dz]
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if (dx, dy, dz) != (0, 0, 0)
        ]
    )
    for i in np.where(map_idx == -1)[0]:
        map_idx[i] = _find_neighbor(orig_xyz_vox, i, offsets, lookup)

    return map_idx


def _save_to_las(
    orig_pc: laspy.LasData,
    save_path: str,
    instance_pred_field: str,
    score_field: str,
    semantic_pred_field: str,
    instance_preds: np.ndarray,
    scores: np.ndarray,
    semantic_preds: np.ndarray,
) -> None:
    _add_las_field_if_not_exist(
        orig_pc,
        [instance_pred_field, semantic_pred_field, score_field],
        [np.uint16, np.uint16, np.float32],
    )
    orig_pc[instance_pred_field] = instance_preds
    orig_pc[score_field] = scores
    orig_pc[semantic_pred_field] = semantic_preds
    if (len(orig_pc.x) > _MAX_LAZ_SIZE) and (save_path[-3:] == "laz"):
        # laz files become corrupted with large file sizes, switch to las
        save_path = f"{save_path[:-1]}s"
    orig_pc.write(save_path)


def _add_las_field_if_not_exist(
    las: laspy.LasData,
    field: Union[str, List[str]],
    dtype: Union[np.dtype, List[np.dtype]],
) -> None:
    if not isinstance(field, List):
        field = [field]
    if not isinstance(dtype, List):
        dtype = [dtype]
    f_to_add = []
    for f_i, d_i in zip(field, dtype):
        if f_i not in las.point_format.dimension_names:
            f_to_add.append(laspy.ExtraBytesParams(f_i, type=d_i))
    las.add_extra_dims(f_to_add)


def _save_to_ply(
    orig_pc: np.ndarray,
    save_path: str,
    instance_pred_field: str,
    score_field: str,
    semantic_pred_field: str,
    instance_preds: np.ndarray,
    scores: np.ndarray,
    semantic_preds: np.ndarray,
) -> None:
    add_values = []
    add_fields = []
    add_types = []
    if instance_pred_field not in orig_pc.dtype.names:
        add_values.append(instance_preds)
        add_fields.append(instance_pred_field)
        add_types.append("u2")
    else:
        orig_pc[instance_pred_field] = instance_preds
    if score_field not in orig_pc.dtype.names:
        add_values.append(scores)
        add_fields.append(score_field)
        add_types.append("f4")
    else:
        orig_pc[score_field] = scores
    if semantic_pred_field not in orig_pc.dtype.names:
        add_values.append(semantic_preds)
        add_fields.append(semantic_pred_field)
        add_types.append("u2")
    else:
        orig_pc[semantic_pred_field] = semantic_preds
    # Add fields
    if len(add_values) > 0:
        dtype = [(field, t) for field, t in zip(add_fields, add_types)]
        new_data = np.zeros(len(add_values[0]), dtype=dtype)
        for field, val in zip(add_fields, add_values):
            new_data[field] = val
        # Merge new values to original data
        orig_pc = rfn.merge_arrays((orig_pc, new_data), flatten=True)

    elements = PlyElement.describe(orig_pc, "vertex")
    PlyData([elements], text=True).write(save_path)


if __name__ == "__main__":
    args = _parse_args()
    start_time = time.time()
    cast_all(
        config=args.config,
        data_config=args.data_config,
        pipeline_config=args.pipeline_config,
        unlabeled_mode=args.unlabeled_mode,
        force_nn=args.force_nn,
        prec=args.prec
    )
    end_time = time.time()
    print(f"==> Finished casting predictions in {(end_time - start_time):.3f} s!")
