import torch

import numpy as np

from mmengine.config import Config
from torch_cluster import knn_graph
from typing import Optional, Dict, Tuple, List


def match_pred_to_gt(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    iou_thresh: float,
    crown_ctgs: List[str],
    trees_gt: Optional[Dict] = None,
    save_ctg: bool = False,
) -> Tuple[Config, Optional[float]]:
    """Match precited tree instances to ground truth instances using IoU-based matching

    Args:
        predicted_labels (np.ndarray): predicted instance label for each point.
        gt_labels (np.ndarray): ground truth instance label for each point.
        iou_thresh (float): threshold for IoU-based matching of predictions and ground truth instances
        crown_ctgs (List[str]): list of crown category names.
        trees_gt (Optional[Dict]): Ground truth tree locations and crown categories. Defaults to None.
        save_ctg (bool, optional): If True, also save crown category information of matched instances. Defaults to False.

    Returns:
        Tuple[Config, Optional[float]]: Dictionary containing the matches and number of tps, fps and fns (+ optionally crown category tps) and sum of max ious (when matching method is set to iou)
    """
    # Get unique instance IDs
    unq_gt = np.unique(gt_labels[gt_labels != 0])
    unq_pred = np.unique(pred_labels[pred_labels != 0])
    n_gt, n_pred = len(unq_gt), len(unq_pred)
    # Remap instance labels to the range [0, N)
    gt_label_to_idx = {label: i for i, label in enumerate(unq_gt)}
    pred_label_to_idx = {label: i for i, label in enumerate(unq_pred)}
    gt_idx = np.array([gt_label_to_idx.get(x, -1) for x in gt_labels])
    pred_idx = np.array([pred_label_to_idx.get(x, -1) for x in pred_labels])
    # Build intersection matrix
    valid = (gt_idx >= 0) & (pred_idx >= 0)
    # Flatten 2D point indices (i, j) to 1D indices. The index pair (i, j) always has the same corresponding 1D index k
    pair_inds = np.ravel_multi_index((gt_idx[valid], pred_idx[valid]), (n_gt, n_pred))
    # Count how many times each pair (i, j) (casted to 1D) occurs -> this is equal to the intersection of the corresponding gt and prediction
    intersection = np.bincount(pair_inds, minlength=n_gt * n_pred).reshape(
        (n_gt, n_pred)
    )
    # Compute instance sizes of predictions and gts to get their unions
    gt_sizes = np.bincount(gt_idx[gt_idx >= 0], minlength=n_gt)
    pred_sizes = np.bincount(pred_idx[pred_idx >= 0], minlength=n_pred)
    union = gt_sizes[:, None] + pred_sizes[None, :] - intersection
    # Compute IoU matrix
    iou_matrix = np.where(union > 0, intersection / union, 0.0)
    if iou_matrix.size > 0:
        # Find matches with maximal IoU for each ground truth segment
        match_col_inds = np.argmax(iou_matrix, axis=1)
        match_row_inds = np.arange(iou_matrix.shape[0])
        # Filter out matches below IoU threshold
        match_ious = iou_matrix[match_row_inds, match_col_inds]
        sum_max_ious = match_ious.sum()
        valid_mask = match_ious >= iou_thresh
        match_row_inds, match_col_inds = (
            match_row_inds[valid_mask],
            match_col_inds[valid_mask],
        )
        gt_ind_matches = unq_gt[match_row_inds]
        pred_ind_matches = unq_pred[match_col_inds]
    else:
        # No predicted segments
        sum_max_ious = 0
        gt_ind_matches = []
        pred_ind_matches = []
    if iou_thresh < 0.5:
        # If IoU threshold is below 0.5, the matches are not guaranteed to be unique -> need to ensure this is the case:
        match_ious = match_ious[valid_mask]
        pred_dict = {}
        for gt_ind, pred_ind, iou in zip(gt_ind_matches, pred_ind_matches, match_ious):
            _, current_iou = pred_dict.get(pred_ind, (0, -np.inf))
            if iou > current_iou:
                pred_dict[pred_ind] = (gt_ind, iou)
        # Extract unique matches
        pred_ind_matches = np.array(list(pred_dict.keys()))
        gt_ind_matches = np.array([pred_dict[key][0] for key in pred_dict.keys()])
    # Save matches in a dict
    matches = _init_matches_dict(n_gt, crown_ctgs, trees_gt, save_ctg)
    for pred_ind, gt_ind in zip(pred_ind_matches, gt_ind_matches):
        matches.match_dict[int(pred_ind)] = int(gt_ind)
        matches.tp += 1
        if save_ctg:
            ctg = trees_gt["trees"][int(gt_ind)]["c"]
            matches.ctg_stats[ctg]["tp"] += 1
    # Predicted instances with no matches are false positives
    for pred_ind in unq_pred:
        if pred_ind not in pred_ind_matches:
            matches["fp"] += 1
    matches["fn"] = matches["n_gt"] - matches["tp"]

    return matches, sum_max_ious


def _init_matches_dict(
    n_gt: int, crown_ctgs: List[str], trees_gt: Optional[Dict], save_ctg: bool
) -> Config:
    """Initialize a dictionary object for saving information of matched segments

    Args:
        n_gt (int): number of ground truth instances in data
        crown_ctgs (List[str]): list of crown category names
        trees_gt (Optional[Dict]): tree ground truth information as a dictionary (required when save_ctg is set to True)
        save_ctg (bool): if true, also intialize a dictionary for saving crown category information

    Returns:
        Config: initialized dictionary for saving matches
    """
    matches = {"match_dict": {}, "tp": 0, "fp": 0, "fn": 0, "n_gt": n_gt}
    if save_ctg:
        ctg_stats = {}
        for ctg in crown_ctgs:
            ctg_stats[ctg] = {"tp": 0, "n_gt": trees_gt["n_trees"][ctg]}
        matches["ctg_stats"] = ctg_stats

    return Config(matches)


def get_stats_from_matches(
    matches_list: List[Config], crown_ctgs: List[str], save_ctg: bool
) -> Tuple[int, int, int, int, Optional[Dict[str, Tuple[int, int]]]]:
    """Fetch tp, fn and fp from a list of plot level match dictionaries

    Args:
        matches_list (List[DictConfig]): list of plot level match dictionaries
        crown_ctgs (List[str]): list of crown category names
        save_ctg (bool): if true, also compute the stats for crown categories

    Returns:
        Tuple[int, int, int, int, Optional[Dict[str, Tuple[int, int]]]]: tp, fp, fn, number of gt instances and (optionally) dict with number of tps and gt instances for each crown category
    """
    tp, fp, fn, n_gt = 0, 0, 0, 0
    ctg_stats = None
    if save_ctg:
        ctg_stats = {}
        for ctg in crown_ctgs:
            ctg_stats[ctg] = {"tp": 0, "n_gt": 0}
    for matches in matches_list:
        tp += matches.tp
        fp += matches.fp
        fn += matches.fn
        n_gt += matches.n_gt
        if save_ctg:
            for ctg in crown_ctgs:
                ctg_stats[ctg]["tp"] += matches.ctg_stats[ctg]["tp"]
                ctg_stats[ctg]["n_gt"] += matches.ctg_stats[ctg]["n_gt"]

    return tp, fp, fn, n_gt, ctg_stats


def match_pred_to_gt_ap(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    pred_confidences: Dict[int, float],
    iou_thresh: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """ScanNet style greedy matching (highest confidence first) for predicted instances and ground truth. Used for computing the average precision

    Args:
        predicted_labels (np.ndarray): predicted instance label for each point.
        gt_labels (np.ndarray): ground truth instance label for each point.
        pred_confidences (Dict[int, float]): confidence/score for each predicted instance (key, value = instance label, score)
        iou_thresh (float): threshold for IoU-based matching

    Returns:
        Tuple[np.ndarray, np.ndarray, int]: array where element at index i is True if prediction i is a true positive. Similarly, another array that contains the corresponding confidence value. Finally, the number of strict false negatives, i.e. ground truth segments with no match regardless of confidence threshold
    """
    # Begin by building an IoU matrix of the predicted segments and ground truth instances
    # Get unique instance IDs
    unq_gt = np.unique(gt_labels[gt_labels != 0])
    unq_pred = np.unique(pred_labels[pred_labels != 0])
    n_gt, n_pred = len(unq_gt), len(unq_pred)
    # Remap instance labels to the range [0, N)
    gt_label_to_idx = {label: i for i, label in enumerate(unq_gt)}
    pred_label_to_idx = {label: i for i, label in enumerate(unq_pred)}
    gt_idx = np.array([gt_label_to_idx.get(x, -1) for x in gt_labels])
    pred_idx = np.array([pred_label_to_idx.get(x, -1) for x in pred_labels])
    # Build intersection matrix
    valid = (gt_idx >= 0) & (pred_idx >= 0)
    # Flatten 2D point indices (i, j) to 1D indices. The index pair (i, j) always has the same corresponding 1D index k
    pair_inds = np.ravel_multi_index((gt_idx[valid], pred_idx[valid]), (n_gt, n_pred))
    # Count how many times each pair (i, j) (casted to 1D) occurs -> this is equal to the intersection of the corresponding gt and prediction
    intersection = np.bincount(pair_inds, minlength=n_gt * n_pred).reshape(
        (n_gt, n_pred)
    )
    # Compute instance sizes of predictions and gts to get their unions
    gt_sizes = np.bincount(gt_idx[gt_idx >= 0], minlength=n_gt)
    pred_sizes = np.bincount(pred_idx[pred_idx >= 0], minlength=n_pred)
    union = gt_sizes[:, None] + pred_sizes[None, :] - intersection
    # Compute IoU matrix
    iou_matrix = np.where(union > 0, intersection / union, 0.0)
    # Fetch confidence values for each predicted segment
    pred_conf = {}
    for i, pred in enumerate(unq_pred):
        conf = pred_confidences[int(pred)]
        # We use the index mapping of the pred labels instead of the original label as key
        pred_conf[i] = conf
    # Sort by confidence (ascending)
    sorted_preds = sorted(pred_conf.items(), key=lambda item: item[1], reverse=True)
    match_cands = {}
    for gt in unq_gt:
        gt_i = gt_label_to_idx[gt]
        match_cands[gt_i] = []
        for pred, _ in sorted_preds:
            iou = iou_matrix[gt_i, pred]
            if iou >= iou_thresh:
                match_cands[gt_i].append(pred)
    # Fetch indices of matching ground truth instances such that the IoU is above the given threshold
    # The ScanNet style matching is greedy in the sense that the match with the highest confidence is always
    # kept (regardless of which match has the highest IoU, provided that threshold is surpassed). Since the
    # matches are handled in a highest confidence first basis, any match such that the ground truth has
    # already been assigned to some other prediction is automatically a false positive
    matched_preds = {}
    for i, pred in enumerate(unq_pred):
        matched_preds[i] = {
            "conf": pred_conf[i],
            "is_tp": False,
        }
    # Track whether gt was matched
    gt_matched = np.zeros(n_gt, dtype=np.bool_)

    for gt_i, matches in match_cands.items():
        for pred in matches:
            if matched_preds[pred]["is_tp"]:
                continue
            if not gt_matched[gt_i]:
                gt_matched[gt_i] = True
                matched_preds[pred]["is_tp"] = True
                break  # matches sorted by confidence -> first found match is the best one

    strict_fn = n_gt - np.sum(gt_matched)
    matches = [pred["is_tp"] for _, pred in matched_preds.items()]
    scores = [float(pred["conf"]) for _, pred in matched_preds.items()]

    return matches, scores, strict_fn


def combine_ap_matches(
    matches_list: List[List[bool]],
    scores_list: List[List[float]],
    strict_fn_list: List[int],
) -> Tuple[List[float], List[float], int]:
    """Combine plot level ap matches into single lists

    Args:
        matches_list (List[List[bool]]): list containing the first output of match_pred_to_gt_ap() for each plot
        scores_list (List[List[float]]): list containing the second output of match_pred_to_gt_ap() for each plot
        strict_fn_list (List[int]): list containing the third output of match_pred_to_gt_ap() for each plot

    Returns:
        Tuple[List[float], List[float], int]: Tuple containing the flattened outputs
    """
    matches_total, scores_total, strict_fn_total = [], [], 0
    for matches, scores, strict_fn in zip(matches_list, scores_list, strict_fn_list):
        matches_total.extend(matches)
        scores_total.extend(scores)
        strict_fn_total += strict_fn

    return matches_total, scores_total, strict_fn_total


def get_graph_edges(
    original_points_list: List[np.ndarray],
    graph_k: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Construct kNN graphs for each point cloud in the provided list and return lists of source and targer nodes corresponding to edges

    Args:
        original_points_list (List[np.ndarray]): list of point cloud xyz-coordinates
        graph_k (int): number of neighbors k in the kNN graph

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: source and target nodes corresponding to edges for each point cloud
    """
    source_node_list = []
    target_node_list = []
    for original_points in original_points_list:
        original_points  = torch.from_numpy(original_points)
        edge_ind = knn_graph(original_points, k=graph_k, batch=None)
        source_nodes = edge_ind[0, :].numpy()
        target_nodes = edge_ind[1, :].numpy()
        source_node_list.append(source_nodes)
        target_node_list.append(target_nodes)

    return source_node_list, target_node_list
