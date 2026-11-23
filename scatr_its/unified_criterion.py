import torch

from util import InstanceData_
from typing import Dict, List, Optional
from mmdet3d.registry import MODELS


@MODELS.register_module()
class ForestUnifiedCriterion:
    """Simply call semantic and instance criterions.

    Args:
        num_semantic_classes (int): Number of semantic classes.
        sem_criterion (Dict): Class for semantic loss calculation.
        inst_criterion (Dict): Class for instance loss calculation.
        instance_only (bool, optional): If True, the predictions only contain instances. Defaults to False.
    """

    def __init__(
        self,
        num_semantic_classes: int,
        sem_criterion: Optional[Dict],
        inst_criterion: Dict,
        instance_only: bool = False,
    ):
        self.instance_only = instance_only
        if not instance_only:
            assert num_semantic_classes > 0, (
                f"'num_semantic_classes' must be greater than zero when 'instance_only' is not set. Got '{num_semantic_classes}' instead!"
            )
            assert sem_criterion is not None, (
                "'sem_criterion' must not be None when 'instance_only' is not set!"
            )
            self.num_semantic_classes = num_semantic_classes
            self.sem_criterion = MODELS.build(sem_criterion)
        else:
            self.num_semantic_classes = 0
            self.sem_criterion = None
        self.inst_criterion = MODELS.build(inst_criterion)
        self.superpoint_size_aware_inst = inst_criterion.get(
            "superpoint_size_aware", False
        )
        self.superpoint_size_aware_sem = sem_criterion.get(
            "superpoint_size_aware", False
        )

    def __call__(self, pred: Dict, insts: List, epoch: Optional[int] = None) -> Dict:
        """Calculate loss.

        Args:
            pred (Dict):
                * List `cls_preds` of shape len batch_size, each of shape (n_queries, n_classes + 1)
                * List `scores` of len batch_size each of shape (n_queries, 1)
                * List `masks` of len batch_size each of shape (n_queries, n_points)
                * Dict `aux_preds` with list of cls_preds, scores, and masks
            insts (List): Ground truth of len batch_size, each InstanceData_ with
                * `sp_inst_masks` of shape (n_gts_i, n_points_i)
                * `sp_sem_masks` of shape (n_classes + 1, n_points_i)
                * `labels_3d` of shape (n_gts_i + n_classes + 1,).
            epoch (Optional[int]): Current training epoch. Defaults to None.

        Returns:
            Dict: with semantic and instance loss values.
        """
        if self.instance_only:
            return self._compute_criterion_instance_only(
                pred=pred, insts=insts, epoch=epoch
            )
        else:
            return self._compute_criterion(pred=pred, insts=insts, epoch=epoch)

    def _compute_criterion(
        self, pred: Dict, insts: List, epoch: Optional[int] = None
    ) -> Dict:
        pred_masks = pred["masks"]
        pred_scores = pred["scores"]
        sem_preds = []
        sem_gts = []
        inst_gts = []
        if self.superpoint_size_aware_inst:
            sp_sizes_inst = []
        else:
            sp_sizes_inst = None
        if self.superpoint_size_aware_sem:
            sp_sizes_sem = []
        else:
            sp_sizes_sem = None
        n = self.num_semantic_classes
        for i in range(len(pred_masks)):
            sem_preds.append(pred_masks[i][-n:, :])
            pred_masks[i] = pred_masks[i][:-n, :]
            pred_scores[i] = pred_scores[i][:-n, :]
            sem_gt = InstanceData_()
            inst_gt = InstanceData_()
            sem_gt.sp_masks = insts[i].sp_sem_masks
            sem_gts.append(sem_gt)
            inst_gt.sp_masks = insts[i].sp_inst_masks
            inst_gt.labels_3d = insts[i].labels_3d
            inst_gt.ratio_inspoint = insts[i].ratio_inspoint
            n_gts = inst_gt.sp_masks.shape[0]
            n_queries = pred_masks[i].shape[0]
            query_inslabel = insts[i].query_inslabel
            query_masks = torch.zeros(
                (n_queries, n_gts), dtype=torch.bool, device=inst_gt.sp_masks.device
            )
            valid_queries = query_inslabel != -1
            query_masks[valid_queries, query_inslabel[valid_queries]] = True
            inst_gt.query_masks = query_masks
            inst_gts.append(inst_gt)
            if self.superpoint_size_aware_inst:
                sp_sizes_inst.append(insts[i].superpoint_sizes)
            if self.superpoint_size_aware_sem:
                sp_sizes_sem.append(insts[i].superpoint_sizes)

        if "aux_outputs" in pred:
            sem_aux_outputs = []
            for aux_outputs in pred["aux_outputs"]:
                sem_aux_outputs.append(self._prepare_aux_outputs(aux_outputs))

        loss = self.inst_criterion(pred, inst_gts, sp_sizes_inst, epoch)
        loss.update(
            self.sem_criterion(
                {"masks": sem_preds, "aux_outputs": sem_aux_outputs},
                sem_gts,
                sp_sizes_sem,
            )
        )

        return loss

    def _compute_criterion_instance_only(
        self, pred: Dict, insts: List, epoch: Optional[int] = None
    ) -> Dict:
        pred_masks = pred["masks"]
        inst_gts = []
        if self.superpoint_size_aware_inst:
            sp_sizes_inst = []
        else:
            sp_sizes_inst = None
        for i in range(len(pred_masks)):
            inst_gt = InstanceData_()
            inst_gt.sp_masks = insts[i].sp_inst_masks
            inst_gt.labels_3d = insts[i].labels_3d
            inst_gt.ratio_inspoint = insts[i].ratio_inspoint
            n_gts = inst_gt.sp_masks.shape[0]
            n_queries = pred_masks[i].shape[0]
            query_inslabel = insts[i].query_inslabel
            query_masks = torch.zeros(
                (n_queries, n_gts), dtype=torch.bool, device=inst_gt.sp_masks.device
            )
            valid_queries = query_inslabel != -1
            query_masks[valid_queries, query_inslabel[valid_queries]] = True
            inst_gt.query_masks = query_masks
            inst_gts.append(inst_gt)
            if self.superpoint_size_aware_inst:
                sp_sizes_inst.append(insts[i].superpoint_sizes)

        loss = self.inst_criterion(pred, inst_gts, sp_sizes_inst, epoch)

        return loss

    def _prepare_aux_outputs(self, aux_outputs: Dict) -> Dict:
        """Prepare aux outputs for intermediate layers.

        Args:
            aux_outputs (Dict):
                * List `cls_preds` of shape len batch_size, each of shape (n_queries, n_classes + 1)
                * List `scores` of len batch_size each of shape (n_queries, 1)
                * List `masks` of len batch_size each of shape (n_queries, n_points).

        Returns:
            Dict: with semantic predictions.
        """
        pred_masks = aux_outputs["masks"]
        pred_scores = aux_outputs["scores"]

        sem_preds = []
        n = self.num_semantic_classes
        for i in range(len(pred_masks)):
            sem_preds.append(pred_masks[i][-n:, :])
            pred_masks[i] = pred_masks[i][:-n, :]
            pred_scores[i] = pred_scores[i][:-n, :]

        return {"masks": sem_preds}
