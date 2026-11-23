import os
import yaml

import numpy as np

from typing import Optional, List, Dict, Tuple
from util import get_accuracy_metrics, read_pc
from mmengine.logging import MMLogger
from mmdet3d.evaluation.metrics import SegMetric
from mmdet3d.registry import METRICS


@METRICS.register_module()
class ForestMetric(SegMetric):
    """Metric for evaluating instance and semantic accuracy of forest data.
    NOTE: if not in test mode, all arguments related to prediction files are ignored.
    Essentially, in test mode the metric reads predicted results from file, while in
    other modes it computes the accuracy of the inputs

        Args:
            data_path (Optional[str]): path to directory containing files with predictions
            metadata_path (Optional[str]): path to directory containing metadata file (crown categories)
            iou_thresh (float): IoU threshold for matching prediction to ground truth.
            metrics (List[str]): metrics to compute. Supported: Supported metrics: [precision, recall, f1, ap, ctg]
            instance_gt_field (Optional[str]): Name of the field tree instance ground truths are stored in
            instance_pred_field (Optional[str]): Name of the field tree instance ground predictions are stored in
            confidence_field (Optional[str]): Name of the field tree instance scores/confidences are stored in
            semantic_gt_field (Optional[str]): Name of the field containing the semantic ground truth
            semantic_pred_field (Optional[str]): Name of the field containing the semantic predictions
            semantic_class_names (Dict[int, str]): Names of each semantic class
            file_type (Optional[str]): file type of prediction files. Supported: [las, laz, ply]
            ignore_classes (List[int]): list of semantic classes to ignore (points are removed before accuracy computations)
            filter_small_instances (bool, optional): If True, remove small instances. Defaults to False.
            min_instance_points (int, optional): Minimum number of points in an instance. Defaults to 40.
            min_instance_h (float, optional): Minimum height of instance. Defaults to 1.5.
            test_mode (bool, optional): If True, use test mode, where predictions are read from file instead. Defaults to False.
            unlabeled_mode (bool, optional): If True, skip metric computations (data has no ground truth labels). Defaults to False.
    """

    def __init__(
        self,
        data_path: Optional[str],
        metadata_path: Optional[str],
        iou_thresh: float,
        metric_list: List[str],
        instance_pred_field: Optional[str],
        confidence_field: Optional[str],
        instance_gt_field: Optional[str],
        semantic_pred_field: Optional[str],
        semantic_gt_field: Optional[str],
        semantic_class_names: Dict[int, str],
        file_type: Optional[str],
        ignore_classes: List[int] = [],
        filter_small_instances: bool = False,
        min_instance_points: int = 40,
        min_instance_h: float = 1.5,
        test_mode: bool = False,
        unlabeled_mode: bool = False,
        **kwargs,
    ):
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.iou_thresh = iou_thresh
        self.metric_list = metric_list
        self.instance_gt_field = instance_gt_field
        self.instance_pred_field = instance_pred_field
        self.confidence_field = confidence_field
        self.semantic_gt_field = semantic_gt_field
        self.semantic_pred_field = semantic_pred_field
        self.semantic_class_names = semantic_class_names
        self.file_type = file_type
        # self.test_files = test_files
        self.ignore_classes = ignore_classes
        if test_mode:
            self.filter_small_instances = filter_small_instances
        else:
            # Filtering disabled if not in test mode
            self.filter_small_instances = False
        self.min_instance_points = min_instance_points
        self.min_instance_h = min_instance_h
        self.test_mode = test_mode
        self.unlabeled_mode = unlabeled_mode
        super().__init__(**kwargs)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (List): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        if not self.unlabeled_mode:
            logger = MMLogger.get_current_instance()

            instance_gt_list = []
            instance_pred_list = []
            instance_score_list = []
            semantic_pred_list = []
            semantic_gt_list = []
            metadata_list = []

            if not self.test_mode:
                for eval_ann, single_pred_results in results:
                    if "originids" in single_pred_results:
                        eval_ann["pts_semantic_mask"] = eval_ann["pts_semantic_mask"][
                            single_pred_results["originids"]
                        ]
                        eval_ann["pts_instance_mask"] = eval_ann["pts_instance_mask"][
                            single_pred_results["originids"]
                        ]
                    # Store semantic labels
                    semantic_gt_list.append(eval_ann["pts_semantic_mask"].copy())
                    semantic_pred_list.append(
                        single_pred_results["pts_semantic_mask"][0].copy()
                    )
                    # Store instance labels
                    instance_gt_list.append(eval_ann["pts_instance_mask"].copy())
                    n_points = instance_gt_list[-1].shape[0]
                    # Extract instance labels from boolean masks
                    instance_masks = single_pred_results["pts_instance_mask"][0].copy()
                    instance_scores = single_pred_results["instance_scores"].copy()
                    instance_labels, score_dict = self._instance_masks_to_labels(
                        n_points, instance_masks, instance_scores
                    )
                    instance_pred_list.append(instance_labels)
                    instance_score_list.append(score_dict)
                    # No metadata if not in test mode
                    metadata_list.append(None)
            else:
                # Read predictions from file and store results
                for _, single_pred_results in results:
                    filename = single_pred_results["filename"]
                    filepath = os.path.join(
                        self.data_path, f"{filename}.{self.file_type}"
                    )
                    pc = read_pc(
                        filepath,
                        self.file_type,
                        self.instance_pred_field,
                        self.instance_gt_field,
                        self.semantic_gt_field,
                        self.ignore_classes,
                        self.filter_small_instances,
                        self.min_instance_points,
                        self.min_instance_h,
                    )
                    # Store semantic labels
                    semantic_gt_list.append(pc[self.semantic_gt_field])
                    semantic_pred_list.append(pc[self.semantic_pred_field])
                    # Store instance labels
                    instance_gt_list.append(pc[self.instance_gt_field])
                    instance_pred_list.append(pc[self.instance_pred_field])
                    score_dict = self._score_dict_from_pc(
                        pc, self.instance_pred_field, self.confidence_field
                    )
                    instance_score_list.append(score_dict)
                    # Fetch plot metadata (if available, and correctly formatted)
                    if self.metadata_path is not None:
                        with open(self.metadata_path, "r") as f:
                            # Metadata contains integer keys so cannot use mmengine Config to load
                            metadata = yaml.safe_load(f)
                        try:
                            plot_id = int(filename.split("_")[-1])
                            metadata_list.append(metadata[plot_id])
                        except Exception:
                            metadata_list.append(None)
                    else:
                        metadata_list.append(None)

            # Compute accuracy metrics
            metric_dict = get_accuracy_metrics(
                instance_pred_list,
                instance_score_list,
                instance_gt_list,
                semantic_pred_list,
                semantic_gt_list,
                metadata_list,
                self.iou_thresh,
                self.metric_list,
                self.semantic_class_names,
                logger=logger,
                test_mode=self.test_mode,
            )

            return metric_dict
        else:
            return {}

    @staticmethod
    def _instance_masks_to_labels(
        n_points: int, instance_masks: List[np.ndarray], instance_scores: np.ndarray
    ) -> Tuple[np.ndarray, Dict[int, float]]:
        """Convert a list of boolean instance masks into a single array containing the predicted
        instance id for each point

        Args:
            n_points (int): number of points in the current point cloud
            instance_masks (List[np.ndarray]): List of boolean instance masks.
            instance_scores (np.ndarray): Instance score of each mask.

        Returns:
            np.ndarray: Predicted instance label for each point and score dictionary with (key, value = instance label, score)
        """
        labels = np.zeros(n_points, dtype=np.uint16)
        score_dict = {}
        # Sort the instance masks by score in descending order
        sorted_instance_masks = sorted(
            zip(instance_masks, instance_scores), key=lambda x: -x[1]
        )
        taken_flag = np.zeros(n_points, dtype=np.bool_)
        for i, mask_n_score in enumerate(sorted_instance_masks):
            mask, score = mask_n_score
            mask = mask & ~taken_flag
            labels[mask] = i + 1
            score_dict[i + 1] = score

        return labels, score_dict

    @staticmethod
    def _score_dict_from_pc(
        pc, instace_pred_field: str, score_field: str
    ) -> Dict[int, float]:
        unq_labels = np.unique(pc[instace_pred_field])
        score_dict = {}
        for label in unq_labels:
            if label != 0:
                score = pc[score_field][pc[instace_pred_field] == label][0]
                score_dict[int(label)] = float(score)

        return score_dict


@METRICS.register_module()
class SuperpointMetric(SegMetric):
    """Metric for generated superpoints. Given superpoint labels, this will compute so-called Oracle metrics
    (e.g. ORecall, OPrecision, OF1-score) for forest data, by assigning each superpoint the majority ground truth
    label it contains and then computing metrics as usual. The metrics are the upper bound accuracy that is
    achievable when using said superpoints to aggregate point-level features.

        Args:
            data_path (Optional[str]): path to directory containing files with predictions
            superpoint_path (Optional[str]): path to directory containing generated superpoint labels.
            metadata_path (Optional[str]): path to directory containing metadata file (crown categories)
            iou_thresh (float): IoU threshold for matching prediction to ground truth.
            metrics (List[str]): metrics to compute. Supported: Supported metrics: [precision, recall, f1, ap, ctg]
            instance_gt_field (Optional[str]): Name of the field tree instance ground truths are stored in
            semantic_gt_field (Optional[str]): Name of the field containing the semantic ground truth
            semantic_class_names (Dict[int, str]): Names of each semantic class
            file_type (Optional[str]): file type of prediction files. Supported: [las, laz, ply]
            ignore_classes (List[int]): list of semantic classes to ignore (points are removed before accuracy computations)
            filter_small_instances (bool, optional): If True, remove small instances. Defaults to False.
            min_instance_points (int, optional): Minimum number of points in an instance. Defaults to 40.
            min_instance_h (float, optional): Minimum height of instance. Defaults to 1.5.
            test_mode (bool, optional): If True, use test mode, where predictions are read from file instead. Defaults to False.
            unlabeled_mode (bool, optional): If True, skip metric computations (data has no ground truth labels). Defaults to False.
    """

    def __init__(
        self,
        data_path: Optional[str],
        superpoint_path: Optional[str],
        metadata_path: Optional[str],
        iou_thresh: float,
        metric_list: List[str],
        instance_gt_field: Optional[str],
        semantic_gt_field: Optional[str],
        semantic_class_names: Dict[int, str],
        file_type: Optional[str],
        ignore_classes: List[int] = [],
        semantic_class_mapping: Optional[Dict[int, int]] = None,
        graph_k: Optional[int] = None,
        test_mode: bool = False,
        unlabeled_mode: bool = False,
        **kwargs,
    ):
        self.data_path = data_path
        self.superpoint_path = superpoint_path
        self.metadata_path = metadata_path
        self.iou_thresh = iou_thresh
        self.metric_list = metric_list
        self.instance_gt_field = instance_gt_field
        self.semantic_gt_field = semantic_gt_field
        self.semantic_class_names = semantic_class_names
        self.file_type = file_type
        self.ignore_classes = ignore_classes
        self.semantic_class_mapping = semantic_class_mapping
        self.graph_k = graph_k
        self.test_mode = test_mode
        self.unlabeled_mode = unlabeled_mode
        super().__init__(**kwargs)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (List): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        if not self.unlabeled_mode:
            logger = MMLogger.get_current_instance()

            instance_gt_list = []
            instance_pred_list = []
            semantic_pred_list = []
            semantic_gt_list = []
            metadata_list = []
            superpoint_counts = []
            superpoint_labels_list = []
            original_points_list = []

            if not self.test_mode:
                for eval_ann, single_pred_results in results:
                    if "originids" in single_pred_results:
                        eval_ann["pts_semantic_mask"] = eval_ann["pts_semantic_mask"][
                            single_pred_results["originids"]
                        ]
                        eval_ann["pts_instance_mask"] = eval_ann["pts_instance_mask"][
                            single_pred_results["originids"]
                        ]
                    superpoint_labels = single_pred_results["pts_superpoint_labels"][
                        0
                    ].copy()
                    # Store semantic gt labels
                    semantic_gt_labels = eval_ann["pts_semantic_mask"].copy()
                    semantic_gt_list.append(semantic_gt_labels)
                    # Generate oracle semantic labels for superpoints
                    semantic_pred_list.append(
                        self._oracle_labels(superpoint_labels, semantic_gt_labels)
                    )
                    # Store instance gt labels
                    instance_gt_labels = eval_ann["pts_instance_mask"].copy()
                    instance_gt_list.append(instance_gt_labels)
                    # Generate oracle instance labels for superpoints
                    instance_pred_list.append(
                        self._oracle_labels(superpoint_labels, instance_gt_labels)
                    )
                    # No metadata if not in test mode
                    metadata_list.append(None)
                    # Store number of superpoints in current scan
                    superpoint_counts.append(len(np.unique(superpoint_labels)))
                # Boundary metrics are not supported when not in test mode
                superpoint_labels_list = None
                original_points_list = None
            else:
                # Read predictions from file and store results
                for _, single_pred_results in results:
                    filename = single_pred_results["filename"]
                    filepath = os.path.join(
                        self.data_path, f"{filename}.{self.file_type}"
                    )
                    pc = read_pc(
                        filepath,
                        self.file_type,
                        None,
                        self.instance_gt_field,
                        self.semantic_gt_field,
                        self.ignore_classes,
                        semantic_class_mapping=self.semantic_class_mapping,
                    )
                    superpoint_labels = np.load(
                        os.path.join(self.superpoint_path, f"{filename}.npy")
                    )
                    # Store semantic labels
                    semantic_gt_labels = pc[self.semantic_gt_field].copy()
                    semantic_gt_list.append(semantic_gt_labels)
                    semantic_pred_list.append(
                        self._oracle_labels(superpoint_labels, semantic_gt_labels)
                    )
                    # Store instance labels
                    instance_gt_labels = pc[self.instance_gt_field].copy()
                    instance_gt_list.append(instance_gt_labels)
                    instance_pred_list.append(
                        self._oracle_labels(superpoint_labels, instance_gt_labels)
                    )
                    # Fetch plot metadata (if available, and correctly formatted)
                    if self.metadata_path is not None:
                        with open(self.metadata_path, "r") as f:
                            # Metadata contains integer keys so cannot use mmengine Config to load
                            metadata = yaml.safe_load(f)
                        try:
                            plot_id = int(filename.split("_")[-1])
                            metadata_list.append(metadata[plot_id])
                        except Exception:
                            metadata_list.append(None)
                    else:
                        metadata_list.append(None)
                    # Store number of superpoints in current scan
                    superpoint_counts.append(len(np.unique(superpoint_labels)))
                    # Store superpoint labels and original points for boundary metric computation
                    superpoint_labels_list.append(superpoint_labels)
                    original_points_list.append(pc.xyz)

            # Compute accuracy metrics
            metric_dict = get_accuracy_metrics(
                instance_pred_list,
                None,
                instance_gt_list,
                semantic_pred_list,
                semantic_gt_list,
                metadata_list,
                self.iou_thresh,
                self.metric_list,
                self.semantic_class_names,
                logger=logger,
                superpoint_mode=True,
                superpoint_counts=superpoint_counts,
                superpoint_labels_list=superpoint_labels_list,
                original_points_list=original_points_list,
                graph_k=self.graph_k,
                test_mode=self.test_mode,
            )

            return metric_dict
        else:
            return {}

    @staticmethod
    def _oracle_labels(
        superpoint_labels: np.ndarray, gt_labels: np.ndarray
    ) -> np.ndarray:
        """Generate oracle labels by assigning each point the majority label within the corresponding superpoint

        Args:
            superpoint_labels (np.ndarray): superpoint label for each point
            gt_labels (np.ndarray): ground truth label (instance or semantic) for each point

        Returns:
            np.ndarray: oracle labels
        """
        sp_ids, sp_inverse = np.unique(superpoint_labels, return_inverse=True)
        n_sp = sp_ids.shape[0]
        gt_ids, gt_inverse = np.unique(gt_labels, return_inverse=True)
        n_gt = gt_ids.shape[0]
        # Count occurrences of (superpoint, gt_label) pairs
        counts = np.zeros((n_sp, n_gt), dtype=np.int64)
        np.add.at(counts, (sp_inverse, gt_inverse), 1)
        # Majority vote for each superpoint
        majority_gt_sp = counts.argmax(axis=1)
        # Map back to original labels
        majority_labels = gt_ids[majority_gt_sp]
        oracle_labels = majority_labels[sp_inverse]

        return oracle_labels
