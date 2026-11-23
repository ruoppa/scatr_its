import numpy as np

from typing import List, Tuple, Dict


def overall_accuracy(tp: int, n_gt: int) -> float:
    """Compute overall accuracy (only used for computing crown category-wise accuracies)

    Args:
        tp (int): Number of true positives
        n_gt (int): Number of ground truth instances

    Returns:
        float: Accuracy
    """
    if n_gt == 0:
        return 1.0  # If number of instances is 0, overall accuracy is 100% (everything succesfully detected)
    else:
        return tp / n_gt


def precision(tp: int, fp: int) -> float:
    """Compute precision

    Args:
        tp (int): Number of true positives
        fp (int): Number of false positives

    Returns:
        float: Precision
    """
    if (tp + fp) == 0:
        return 0.0  # Ensure no division by 0 happens. When there are no detections -> precision = 0
    else:
        return tp / (tp + fp)


def recall(tp: int, fn: int) -> float:
    """Compute precision

    Args:
        tp (int): Number of true positives
        fn (int): Number of false negatives

    Returns:
        float: Recall
    """
    if (tp + fn) == 0:
        return 1.0  # Ensure no division by 0 happens. When there are no true instances -> recall = 1
    else:
        return tp / (tp + fn)


def f1_score(tp: int, fp: int, fn: int) -> float:
    """Compute F1-score

    Args:
        tp (int): Number of true positives
        fp (int): Number of false positives
        fn (int): Number of false negatives

    Returns:
        float: F1-score
    """
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    if (prec + rec) == 0:
        return 0.0  # Ensure no division by 0 happens. When precision and recall are both 0 -> F1-score = 0
    else:
        return 2 * (prec * rec) / (prec + rec)


def coverage(n_gt: int, sum_max_ious: float) -> float:
    """Compute coverage (see e.g.: https://www.sciencedirect.com/science/article/pii/S0034425724000890)

    Args:
        n_gt (int): number of ground truth instances
        sum_max_ious (float): sum of maximum IoUs for each ground truth instances (max IoU achieved with any predicted segment)

    Returns:
        float: coverage
    """
    if n_gt == 0:
        return 1.0  # No ground truth segments -> full coverage
    else:
        return sum_max_ious / n_gt


def iou(
    semantic_pred: np.ndarray,
    semantic_gt: np.ndarray,
    num_semantic_classes: int,
    semantic_class_names: Dict[int, str],
) -> Dict[str, float]:
    semantic_pred, semantic_gt, semantic_class_names = _remap_labels(
        semantic_pred, semantic_gt, semantic_class_names
    )
    mask = (
        (semantic_gt >= 0) & (semantic_gt < num_semantic_classes) & (semantic_pred >= 0)
    )
    conf_matrix = np.bincount(
        num_semantic_classes * semantic_gt[mask].astype(np.int64)
        + semantic_pred[mask].astype(np.int64),
        minlength=num_semantic_classes**2,
    ).reshape(num_semantic_classes, num_semantic_classes)
    tp = np.diag(conf_matrix)
    o_acc = tp.sum() / conf_matrix.sum()
    fp = np.sum(conf_matrix, 1) - tp  # sum of rows
    fn = np.sum(conf_matrix, 0) - tp  # sum of columns
    m_acc = np.nanmean(tp / (tp + fn))
    iou_array = tp / (tp + fp + fn + 1e-8)
    m_iou = np.nanmean(iou_array)
    iou = {"oAcc": o_acc, "mAcc": m_acc, "mIoU": m_iou}
    for i, iou_i in enumerate(iou_array):
        iou[semantic_class_names[i]] = iou_i

    return iou


def average_precision(
    matches: List[float], scores: List[float], strict_fn: int
) -> float:
    """Compute average precision

    Args:
        matches (List[float]): first output of combine_ap_matches()
        scores (List[float]): second output of combine_ap_matches()
        strict_fn (int): third output of combine_ap_matches()

    Returns:
        float: dataset level average precision
    """
    # Code for computing AP from the official ScanNet GitHub repository:
    # https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py
    if len(matches) <= 0:
        return 0.0  # Edge case with no matches
    matches, scores = np.array(matches), np.array(scores)
    # sorting and cumsum
    score_arg_sort = np.argsort(scores)
    scores_sorted = scores[score_arg_sort]
    matches_sorted = matches[score_arg_sort]
    matches_sorted_cumsum = np.cumsum(matches_sorted)
    # unique thresholds
    (_, unique_indices) = np.unique(scores_sorted, return_index=True)
    num_prec_recall = len(unique_indices) + 1
    num_examples = len(scores_sorted)
    num_true_examples = matches_sorted_cumsum[-1]
    precision = np.zeros(num_prec_recall)
    recall = np.zeros(num_prec_recall)
    matches_sorted_cumsum = np.append(matches_sorted_cumsum, 0)
    for idx_res, idx_scores in enumerate(unique_indices):
        cumsum = matches_sorted_cumsum[idx_scores - 1]
        tp = num_true_examples - cumsum
        fp = num_examples - idx_scores - tp
        fn = cumsum + strict_fn
        p = float(tp) / (tp + fp)
        r = float(tp) / (tp + fn)
        precision[idx_res] = p
        recall[idx_res] = r
    precision[-1] = 1.0
    recall[-1] = 0.0
    recall_for_conv = np.copy(recall)
    recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
    recall_for_conv = np.append(recall_for_conv, 0.0)
    step_widths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], "valid")
    ap = np.dot(precision, step_widths)

    return ap


def _remap_labels(
    pred_labels: np.ndarray, gt_labels: np.ndarray, semantic_class_names: Dict[int, str]
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """Remap labels to a continuous range

    Args:
        pred_labels (np.ndarray): predicted labels
        gt_labels (np.ndarray): ground truth labels
        semantic_class_names (Dict[int, str]): name of each semantic class

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[int, str]]: remapped predicted and gt labels and class names
    """
    unique_labels = np.unique(np.concatenate([gt_labels, pred_labels]))
    max_label = unique_labels.max()
    lookup_table = np.full(max_label + 1, -1, dtype=np.int32)
    lookup_table[unique_labels] = np.arange(len(unique_labels))
    pred_remapped = lookup_table[pred_labels]
    gt_remapped = lookup_table[gt_labels]
    names_remapped = {
        i: semantic_class_names[int(label)] for i, label in enumerate(unique_labels)
    }

    return pred_remapped, gt_remapped, names_remapped


def boundary_recall(
    gt_labels: List[np.ndarray],
    sp_labels: List[np.ndarray],
    source: List[np.ndarray],
    target: List[np.ndarray],
) -> float:
    """Compute superpoint boundary recall as described in "Point Cloud Oversegmentation
    with Graph-Structured Deep Metric Learning", (Landrieu et al., 2019)

    Args:
        gt_labels (List[np.ndarray]): list of ground truth labels for each point cloud
        sp_labels (List[np.ndarray]): list of superpoint labels for each point cloud
        source (List[np.ndarray]): list of lists containing source indices for all edges in the graph used for generating the superpoints
        target (List[np.ndarray]): list of lists containing target indices for all edges in the graph used for generating the superpoints

    Returns:
        float: boundary recall
    """
    br_numerator = 0
    br_denominator = 0

    for gt, sp, src, tgt in zip(gt_labels, sp_labels, source, target):
        # Boolean masks for edges crossing superpoint boundaries and GT boundaries
        pred_transition = sp[src] != sp[tgt]
        is_transition = gt[src] != gt[tgt]
        br_num, br_denom = _get_br_parts(
            is_transition, _relax_edge_binary(pred_transition, src, tgt, gt.shape[0])
        )
        br_numerator += br_num
        br_denominator += br_denom

    # Avoid division by zero
    br = (br_numerator / br_denominator) if br_denominator > 0 else 0.0

    return br


def boundary_precision(
    gt_labels: List[np.ndarray],
    sp_labels: List[np.ndarray],
    source: List[np.ndarray],
    target: List[np.ndarray],
) -> float:
    """Compute superpoint boundary precision as described in "Point Cloud Oversegmentation
    with Graph-Structured Deep Metric Learning", (Landrieu et al., 2019)

    Args:
        gt_labels (List[np.ndarray]): list of ground truth labels for each point cloud
        sp_labels (List[np.ndarray]): list of superpoint labels for each point cloud
        source (List[np.ndarray]): list of lists containing source indices for all edges in the graph used for generating the superpoints
        target (List[np.ndarray]): list of lists containing target indices for all edges in the graph used for generating the superpoints

    Returns:
        float: boundary precision
    """
    bp_numerator = 0
    bp_denominator = 0

    for gt, sp, src, tgt in zip(gt_labels, sp_labels, source, target):
        # Boolean masks for edges crossing superpoint boundaries and GT boundaries
        pred_transition = sp[src] != sp[tgt]
        is_transition = gt[src] != gt[tgt]
        bp_num, bp_denom = _get_bp_parts(
            _relax_edge_binary(is_transition, src, tgt, gt.shape[0]), pred_transition
        )
        bp_numerator += bp_num
        bp_denominator += bp_denom

    # Avoid division by zero
    bp = (bp_numerator / bp_denominator) if bp_denominator > 0 else 0.0

    return bp


def _get_br_parts(
    is_transition: np.ndarray, pred_transition: np.ndarray
) -> Tuple[float, float]:
    br_num = ((is_transition == pred_transition) * is_transition).sum()
    br_denom = is_transition.sum()
    return br_num, br_denom


def _get_bp_parts(
    is_transition: np.ndarray, pred_transition: np.ndarray
) -> Tuple[float, float]:
    bp_num = ((is_transition == pred_transition) * pred_transition).sum()
    bp_denom = pred_transition.sum()
    return bp_num, bp_denom


def _relax_edge_binary(
    edge_binary: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    n_vertices: int,
    tolerance: int = 1,
) -> np.ndarray:
    """Relax a binary mask of transitions such that it also contains all edges within 'tolerance'. Code copied
    from: https://github.com/loicland/superpoint_graph/blob/ssp%2Bspg/supervized_partition/losses.py

    Args:
        edge_binary (np.ndarray): binary mask that is true for edges that are transitions
        source (np.ndarray): indices of source nodes
        target (np.ndarray): indices of target nodes
        n_vertices (int): number of vertices in the graph
        tolerance (int, optional): Tolerance to relax to. Defaults to 1.

    Returns:
        np.ndarray: relaxed binary mask
    """
    relaxed_binary = edge_binary.copy()
    transition_vertex = np.full((n_vertices,), 0, dtype="uint8")
    for _ in range(tolerance):
        transition_vertex[source[relaxed_binary.nonzero()]] = True
        transition_vertex[target[relaxed_binary.nonzero()]] = True
        relaxed_binary[transition_vertex[source]] = True
        relaxed_binary[transition_vertex[target] > 0] = True

    return relaxed_binary
