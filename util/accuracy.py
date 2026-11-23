import numpy as np

from tqdm import tqdm
from tabulate import tabulate
from mmengine.logging import MMLogger
from mmengine.logging import print_log
from typing import Dict, List, Optional
from util.match import (
    match_pred_to_gt,
    match_pred_to_gt_ap,
    get_stats_from_matches,
    combine_ap_matches,
    get_graph_edges,
)
from util.metrics import (
    overall_accuracy,
    precision,
    recall,
    f1_score,
    coverage,
    iou,
    average_precision,
    boundary_recall,
    boundary_precision,
)

_VALID_METRICS = ["precision", "recall", "f1", "ctg", "cov", "ap", "iou", "br", "bp", "rr"]


def get_accuracy_metrics(
    instance_pred_list: List[np.ndarray],
    instance_score_list: Optional[List[np.ndarray]],
    instance_gt_list: List[np.ndarray],
    semantic_pred_list: List[np.ndarray],
    semantic_gt_list: List[np.ndarray],
    metadata_list: List[Dict],
    iou_thresh: float,
    metrics: List[str],
    semantic_class_names: Dict[int, str],
    logger: Optional[MMLogger] = None,
    crown_ctgs: List[str] = ["A", "B", "C", "D"],
    superpoint_mode: bool = False,
    superpoint_counts: Optional[List[int]] = None,
    superpoint_labels_list: Optional[List[np.ndarray]] = None,
    original_points_list: Optional[List[np.ndarray]] = None,
    graph_k: Optional[int] = None,
    test_mode: bool = False,
) -> Dict:
    metrics = list(set(metrics))
    metrics = [m.lower() for m in metrics]
    assert len(metrics) > 0, "Metrics to compute is empty!"
    # Crown category accuracy cannot be computed if not in test mode
    if (not test_mode) and ("ctg" in metrics):
        metrics.remove("ctg")
    # BR, BP and RR cannot be computed if not in test mode AND superpoint mode
    if (not (test_mode and superpoint_mode)) and ("bp" in metrics):
        metrics.remove("bp")
    if (not (test_mode and superpoint_mode)) and ("br" in metrics):
        metrics.remove("br")
    if (not (test_mode and superpoint_mode)) and ("rr" in metrics):
        metrics.remove("rr")
    # AP cannot be computed in superpoint mode
    if superpoint_mode and ("ap" in metrics):
        metrics.remove("ap")
    assert iou_thresh > 0 and iou_thresh <= 1, (
        f"IoU threshold must be in the range (0, 1]. Got {iou_thresh} instead!"
    )
    for m in metrics:
        assert m in _VALID_METRICS, (
            f"'{m}' is not a valid metric! Supported metrics: {_VALID_METRICS}"
        )
    compute_inst = any(
        [m in metrics for m in ["precision", "recall", "f1", "cov", "ctg"]]
    )
    save_ctg = "ctg" in metrics
    compute_ap = "ap" in metrics
    compute_cov = "cov" in metrics
    compute_iou = "iou" in metrics
    compute_sp_stats = superpoint_counts is not None
    compute_boundary = ("br" in metrics) or ("bp" in metrics)
    compute_rr = compute_sp_stats and compute_boundary and ("rr" in metrics)
    if compute_ap:
        assert instance_score_list is not None, (
            "'instance_score_list' cannot be 'None' when AP is in metrics!"
        )
    if compute_sp_stats:
        assert len(superpoint_counts) == len(instance_gt_list), (
            f"length of 'superpoint_counts' does not match the number of scans ({len(superpoint_counts)} != {len(instance_gt_list)})!"
        )
    if compute_boundary:
        assert superpoint_labels_list is not None, (
            "'superpoint_labels_list' cannot be 'None' when BP or BR are in metrics!"
        )
        assert original_points_list is not None, (
            "'original_points_list' cannot be 'None' when BP or BR are in metrics!"
        )
        assert graph_k is not None, (
            "'graph_k' cannot be 'None' when BP or BR are in metrics!"
        )
    matches_list = []
    ap_matches_list = []
    ap_scores_list = []
    ap_strict_fn_list = []
    sum_max_ious_list = []
    # Match predicted instance to ground truth instances
    n_scans = len(instance_pred_list)
    if instance_score_list is None:
        instance_score_list = [None] * n_scans
    to_iterate = tqdm(
        zip(instance_pred_list, instance_gt_list, instance_score_list, metadata_list),
        desc="==> Matching predictions to ground truth",
        total=n_scans,
    )
    for pred_labels, gt_labels, pred_scores, metadata in to_iterate:
        if compute_inst:
            matches, sum_max_ious = match_pred_to_gt(
                pred_labels, gt_labels, iou_thresh, crown_ctgs, metadata, save_ctg
            )
            matches_list.append(matches)
        if compute_ap:
            ap_matches, ap_scores, ap_strict_fn = match_pred_to_gt_ap(
                pred_labels, gt_labels, pred_scores, iou_thresh
            )
            ap_matches_list.append(ap_matches)
            ap_scores_list.append(ap_scores)
            ap_strict_fn_list.append(ap_strict_fn)
        if compute_cov:
            sum_max_ious_list.append(sum_max_ious)
    # Fetch number of tp, fp, fn etc.
    print_log("==> Collecting number of matches", logger=logger)
    if compute_inst:
        tp, fp, fn, n_gt, ctg_stats = get_stats_from_matches(
            matches_list, crown_ctgs, save_ctg
        )
    if compute_cov:
        sum_max_ious = float(np.sum(sum_max_ious_list))
    # Compute required instance metrics
    instance_table = []
    instance_headers = []
    metric_dict = {}
    metric_suffix = _get_metric_suffix(iou_thresh)
    metric_prefix = _get_metric_prefix(superpoint_mode)
    print_log("==> Computing instance accuracy metrics", logger=logger)
    if "precision" in metrics:
        prec = precision(tp, fp) * 100
        instance_table.append(prec)
        instance_headers.append(f"{metric_prefix}Precision{metric_suffix}")
        metric_dict[f"{metric_prefix.lower()}precision"] = prec
    if "recall" in metrics:
        rec = recall(tp, fn) * 100
        instance_table.append(rec)
        instance_headers.append(f"{metric_prefix}Recall{metric_suffix}")
        metric_dict[f"{metric_prefix.lower()}recall"] = rec
    if "f1" in metrics:
        f1 = f1_score(tp, fp, fn) * 100
        instance_table.append(f1)
        instance_headers.append(f"{metric_prefix}F1-score{metric_suffix}")
        metric_dict[f"{metric_prefix.lower()}f1"] = f1
    if "cov" in metrics:
        cov = coverage(n_gt, sum_max_ious) * 100
        instance_table.append(cov)
        instance_headers.append(
            f"{metric_prefix}Cov"
        )  # Coverage not dependent on IoU threshold
        metric_dict[f"{metric_prefix.lower()}cov"] = cov
    if "ap" in metrics:
        ap_matches, ap_scores, ap_strict_fn = combine_ap_matches(
            ap_matches_list, ap_scores_list, ap_strict_fn_list
        )
        ap = average_precision(ap_matches, ap_scores, ap_strict_fn) * 100
        instance_table.append(ap)
        instance_headers.append(f"AP{metric_suffix}")
        metric_dict["ap"] = ap
    if save_ctg:
        print_log("==> Computing crown category-level accuracy metrics", logger=logger)
        ctg_table = []
        ctg_headers = []
        for ctg in crown_ctgs:
            tp_ctg = ctg_stats[ctg]["tp"]
            n_gt_ctg = ctg_stats[ctg]["n_gt"]
            if n_gt_ctg > 0:
                recall_ctg = overall_accuracy(tp_ctg, n_gt_ctg) * 100
                ctg_table.append(recall_ctg)
                ctg_headers.append(f"{ctg}: {metric_prefix}Recall{metric_suffix}")
                metric_dict[f"{metric_prefix.lower()}recall_{ctg.lower()}"] = recall_ctg
    if compute_iou:
        print_log("==> Computing semantic accuracy metrics", logger=logger)
        semantic_pred = np.concatenate(semantic_pred_list)
        semantic_gt = np.concatenate(semantic_gt_list)
        iou_dict = iou(
            semantic_pred, semantic_gt, len(semantic_class_names), semantic_class_names
        )
        iou_table = []
        iou_header = []
        for name, val in iou_dict.items():
            iou_table.append(val * 100)
            iou_header.append(f"{metric_prefix}{name}")
            metric_dict[f"{metric_prefix.lower()}{name.lower()}"] = val * 100
    # Compute superpoint statistics (if necessary)
    if compute_sp_stats:
        print_log("==> Computing superpoint count statistics", logger=logger)
        superpoint_counts = np.array(superpoint_counts)
        total_n_sp = float(np.sum(superpoint_counts))
        mean_n_sp = float(np.mean(superpoint_counts))
        min_n_sp = float(np.min(superpoint_counts))
        max_n_sp = float(np.max(superpoint_counts))
        sp_table = [total_n_sp, mean_n_sp, min_n_sp, max_n_sp]
        sp_header = ["Total", "Mean", "Min", "Max"]
        for name, val in zip(sp_header, sp_table):
            metric_dict[f"{name.lower()}_n_sp"] = val
        if compute_rr:
            point_counts = np.array([len(pc) for pc in original_points_list])
            total_n_points = float(np.sum(point_counts))
            rr_val = total_n_sp / total_n_points
            sp_table.append(rr_val * 100)
            sp_header.append("RR")
            metric_dict["rr"] = rr_val * 100
    # Compute superpoint boundary metrics (if necessary)
    if compute_boundary:
        print_log("==> Computing superpoint boundary accuracy metrics", logger=logger)
        boundary_table = []
        boundary_header = []
        source_node_list, target_node_list = get_graph_edges(
            original_points_list, graph_k
        )
        if "br" in metrics:
            br = (
                boundary_recall(
                    instance_gt_list,
                    superpoint_labels_list,
                    source_node_list,
                    target_node_list,
                )
                * 100
            )
            boundary_table.append(br)
            boundary_header.append("BR")
            metric_dict["br"] = br
        if "bp" in metrics:
            bp = (
                boundary_precision(
                    instance_gt_list,
                    superpoint_labels_list,
                    source_node_list,
                    target_node_list,
                )
                * 100
            )
            boundary_table.append(bp)
            boundary_header.append("BP")
            metric_dict["bp"] = bp

    # Log results
    if len(instance_table) > 0:
        _print_results(
            instance_table, instance_headers, "INSTANCE", superpoint_mode, logger
        )
    if save_ctg:
        _print_results(
            ctg_table, ctg_headers, "CROWN CATEGORY", superpoint_mode, logger
        )
    if compute_iou:
        _print_results(iou_table, iou_header, "SEMANTIC", superpoint_mode, logger)
    if compute_boundary:
        _print_results(
            boundary_table, boundary_header, "SUPERPOINT BOUNDARY", False, logger=logger
        )
    if compute_sp_stats:
        _print_results(
            sp_table,
            sp_header,
            "\n==> SUPERPOINT COUNT STATISTICS:\n",
            superpoint_mode,
            logger,
            override_title=True,
        )

    return metric_dict


def _get_metric_suffix(iou_thresh: float):
    iou_val = iou_thresh * 100
    iou_val_int = int(iou_val)
    if iou_val_int == iou_val:
        return f"@{iou_val_int}"
    else:
        return f"@{iou_val:.1f}"


def _get_metric_prefix(superpoint_mode: bool):
    if superpoint_mode:
        return "O"
    else:
        return ""


def _print_results(
    table: List,
    header: List,
    title: str,
    superpoint_mode: bool,
    logger: MMLogger,
    override_title: bool = False,
):
    if override_title:
        result_string = title
    elif superpoint_mode:
        result_string = f"\n==> {title} ORACLE ACCURACY METRICS:\n"
    else:
        result_string = f"\n==> {title} ACCURACY METRICS:\n"
    result_string += (
        tabulate([table], headers=header, floatfmt=".3f", tablefmt="grid") + "\n"
    )
    print_log(
        result_string,
        logger=logger,
    )
