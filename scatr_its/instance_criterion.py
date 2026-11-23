import torch

import torch.nn.functional as F

from util import InstanceData_
from torch import Tensor
from typing import List, Callable, Tuple, Dict, Optional
from scipy.optimize import linear_sum_assignment
from scatr_its.tools import get_iou
from scatr_its.losses import batch_sigmoid_bce_loss, batch_dice_loss, dice_loss
from mmdet3d.registry import MODELS, TASK_UTILS


@MODELS.register_module()
class ForestInstanceCriterion:
    """Instance criterion for ForestFormer3D

    Args:
        matcher (Callable): Class for matching queries with gt.
        loss_weight (List[float]): 4 weights for query classification, mask bce, mask dice, and score losses.
        fix_dice_loss_weight (bool): Whether to fix dice loss for batch_size != 4.
        iter_matcher (bool): Whether to use separate matcher for each decoder layer.
        fix_mean_loss (bool): Whether to use .mean() instead of .sum() for mask losses.

    """

    def __init__(
        self,
        matcher: Callable,
        loss_weight: List[float],
        fix_dice_loss_weight: float,
        iter_matcher: bool,
        fix_mean_loss: bool = False,
        superpoint_size_aware: bool = False,
    ):
        self.matcher = TASK_UTILS.build(matcher)
        self.loss_weight = loss_weight
        self.fix_dice_loss_weight = fix_dice_loss_weight
        self.iter_matcher = iter_matcher
        self.fix_mean_loss = fix_mean_loss
        self.superpoint_size_aware = superpoint_size_aware

    def _get_src_permutation_idx(self, indices: Tensor):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])

        return batch_idx, src_idx

    def get_layer_loss(
        self,
        aux_outputs: Dict,
        insts: List,
        indices: Optional[Tensor] = None,
        sp_sizes: Optional[List] = None,
        epoch: Optional[int] = None,
    ) -> Tensor:
        """Per layer auxiliary loss.

        Args:
            aux_outputs (Dict):
                * List `cls_preds` of shape len batch_size, each of shape (n_queries, n_classes + 1)
                * List `scores` of len batch_size each of shape (n_queries, 1)
                * List `masks` of len batch_size each of shape (n_queries, n_points)
            insts (List): Ground truth of len batch_size, each InstanceData_ with
                * `sp_masks` of shape (n_gts_i, n_points_i)
                * `labels_3d` of shape (n_gts_i,)
                * `query_masks` of shape (n_gts_i, n_queries_i).
            indices (Optional[List]): indices of matched prediction and ground truth instances
            sp_sizes (Optional[List]): Superpoint sizes of len batch_size, each is a Tensor where index i contains the number of points in superpoint i
            epoch (Optional[int]): Current training epoch. Defaults to None.

        Returns:
            Tensor: loss value.
        """
        if not self.superpoint_size_aware:
            sp_sizes = [None] * len(insts)
        pred_scores = aux_outputs["scores"]
        pred_masks = aux_outputs["masks"]

        if indices is None:
            indices = []
            for i in range(len(insts)):
                pred_instances = InstanceData_(masks=pred_masks[i])
                gt_instances = InstanceData_(
                    labels=insts[i].labels_3d, masks=insts[i].sp_masks
                )
                if insts[i].get("query_masks") is not None:
                    gt_instances.query_masks = insts[i].query_masks
                indices.append(
                    self.matcher(
                        pred_instances, gt_instances, sp_size=sp_sizes[i], epoch=epoch
                    )
                )

        # 3 other losses
        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for mask, score, inst, sp_size, (idx_q, idx_gt) in zip(
            pred_masks, pred_scores, insts, sp_sizes, indices
        ):
            if len(inst) == 0:
                continue
            pred_mask = mask[idx_q]
            # No need to distinguish between positive and negative samples, compute for all masks
            tgt_mask = torch.zeros_like(pred_mask)
            if (idx_gt != -1).any():
                # For positive samples, assign the ground truth mask
                tgt_mask[idx_gt != -1] = inst.sp_masks[idx_gt[idx_gt != -1]].float()

            # Compute mask losses (both positive and negative)
            bce_loss_i = F.binary_cross_entropy_with_logits(
                pred_mask, tgt_mask, reduction="none"
            )
            if self.superpoint_size_aware:
                # Weight loss elements by superpoint sizes
                weights = sp_size.unsqueeze(0)
                bce_loss_i = (bce_loss_i * weights).sum() / weights.expand_as(
                    bce_loss_i
                ).sum()
            else:
                bce_loss_i = bce_loss_i.mean()
            mask_bce_losses.append(bce_loss_i)
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask, sp_size))

            # Calculate score losses
            if score is not None:
                pred_score = score[idx_q]
                # Default score for negative samples is 0
                tgt_score = torch.zeros_like(pred_score)
                if (idx_gt != -1).any():
                    # For positive samples, use IoU as the ground truth score
                    with torch.no_grad():
                        tgt_score[idx_gt != -1] = get_iou(
                            pred_mask[idx_gt != -1],
                            tgt_mask[idx_gt != -1],
                            inst.ratio_inspoint[idx_gt[idx_gt != -1]],
                            sp_size,
                        ).unsqueeze(1)

                # Append score loss for both positive and negative samples
                score_losses.append(F.mse_loss(pred_score, tgt_score))

        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = 0

        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum() / len(pred_masks)

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4

            if self.fix_mean_loss:
                mask_bce_loss = mask_bce_loss * len(pred_masks) / len(mask_bce_losses)
                mask_dice_loss = (
                    mask_dice_loss * len(pred_masks) / len(mask_dice_losses)
                )
        else:
            mask_bce_loss = 0
            mask_dice_loss = 0

        loss = (
            self.loss_weight[0] * mask_bce_loss
            + self.loss_weight[1] * mask_dice_loss
            + self.loss_weight[2] * score_loss
        )

        return loss

    def __call__(
        self,
        pred: Dict,
        insts: List,
        sp_sizes: Optional[List] = None,
        epoch: Optional[int] = None,
    ) -> Dict:
        """Loss main function.

        Args:
            pred (Dict):
                * List `cls_preds` of shape len batch_size, each of shape (n_queries, n_classes + 1)
                * List `scores` of len batch_size each of shape (n_queries, 1)
                * List `masks` of len batch_size each of shape (n_queries, n_points)
                * Dict `aux_preds` with list of cls_preds, scores, and masks.
            insts (List): Ground truth of len batch_size, each InstanceData_ with
                * `sp_masks` of shape (n_gts_i, n_points_i)
                * `labels_3d` of shape (n_gts_i,)
                * `query_masks` of shape (n_gts_i, n_queries_i).
            sp_sizes (Optional[List]): Superpoint sizes of len batch_size, each is a Tensor where index i contains the number of points in superpoint i
            epoch (Optional[int]): Current training epoch. Defaults to None.

        Returns:
            Dict: with instance loss value.
        """
        if self.superpoint_size_aware:
            assert sp_sizes is not None, (
                "'sp_sizes' should not be None when 'superpoint_size_aware' is set to True!"
            )
        else:
            sp_sizes = [None] * len(insts)
        pred_scores = pred["scores"]
        pred_masks = pred["masks"]
        # match
        indices = []
        for i in range(len(insts)):
            pred_instances = InstanceData_(masks=pred_masks[i])
            gt_instances = InstanceData_(
                labels=insts[i].labels_3d, masks=insts[i].sp_masks
            )
            if insts[i].get("query_masks") is not None:
                gt_instances.query_masks = insts[i].query_masks
            indices.append(
                self.matcher(
                    pred_instances, gt_instances, sp_size=sp_sizes[i], epoch=epoch
                )
            )

        # 3 other losses
        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for mask, score, inst, sp_size, (idx_q, idx_gt) in zip(
            pred_masks, pred_scores, insts, sp_sizes, indices
        ):
            if len(inst) == 0:
                continue
            pred_mask = mask[idx_q]
            # No need to distinguish between positive and negative samples, compute for all masks
            tgt_mask = torch.zeros_like(pred_mask)
            if (idx_gt != -1).any():
                # For positive samples, assign the ground truth mask
                tgt_mask[idx_gt != -1] = inst.sp_masks[idx_gt[idx_gt != -1]].float()
            # Compute mask losses (both positive and negative)
            bce_loss_i = F.binary_cross_entropy_with_logits(
                pred_mask, tgt_mask, reduction="none"
            )
            if self.superpoint_size_aware:
                # Weight loss elements by superpoint sizes
                weights = sp_size.unsqueeze(0)
                bce_loss_i = (bce_loss_i * weights).sum() / weights.expand_as(
                    bce_loss_i
                ).sum()
            else:
                bce_loss_i = bce_loss_i.mean()
            mask_bce_losses.append(bce_loss_i)
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask, sp_size))

            # Calculate score losses
            if score is not None:
                pred_score = score[idx_q]

                tgt_score = torch.zeros_like(pred_score)
                if (idx_gt != -1).any():
                    # For positive samples, use IoU as the ground truth score
                    with torch.no_grad():
                        tgt_score[idx_gt != -1] = get_iou(
                            pred_mask[idx_gt != -1],
                            tgt_mask[idx_gt != -1],
                            inst.ratio_inspoint[idx_gt[idx_gt != -1]],
                            sp_size,
                        ).unsqueeze(1)

                # Append score loss for both positive and negative samples
                score_losses.append(F.mse_loss(pred_score, tgt_score))

        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = 0

        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum()

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4

            if self.fix_mean_loss:
                mask_bce_loss = mask_bce_loss * len(pred_masks) / len(mask_bce_losses)
                mask_dice_loss = (
                    mask_dice_loss * len(pred_masks) / len(mask_dice_losses)
                )
        else:
            mask_bce_loss = 0
            mask_dice_loss = 0

        loss = (
            self.loss_weight[0] * mask_bce_loss
            + self.loss_weight[1] * mask_dice_loss
            + self.loss_weight[2] * score_loss
        )

        if "aux_outputs" in pred:
            if self.iter_matcher:
                indices = None
            for i, aux_outputs in enumerate(pred["aux_outputs"]):
                loss += self.get_layer_loss(aux_outputs, insts, indices, sp_sizes, epoch)

        return {"inst_loss": loss}


@TASK_UTILS.register_module()
class QueryClassificationCost:
    """Classification cost for queries.

    Args:
        weigth (float): Weight of the cost.
    """

    def __init__(self, weight: float):
        self.weight = weight

    def __call__(
        self, pred_instances: InstanceData_, gt_instances: InstanceData_, **kwargs
    ) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (InstanceData_): Predicted instances which must contain `masks` of shape (n_queries, n_points).
            gt_instances (InstanceData_): Ground truth which must contain `masks` of shape (n_gts, n_points).

        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        scores = pred_instances.scores.softmax(-1)
        cost = -scores[:, gt_instances.labels]

        return cost * self.weight


@TASK_UTILS.register_module()
class MaskBCECost:
    """Sigmoid BCE cost for masks.

    Args:
        weigth (float): Weight of the cost.
    """

    def __init__(self, weight: float):
        self.weight = weight

    def __call__(
        self, pred_instances: InstanceData_, gt_instances: InstanceData_, **kwargs
    ) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (InstanceData_): Predicted instances which must contain `masks` of shape (n_queries, n_points).
            gt_instances (InstanceData_): Ground truth which must contain `masks` of shape (n_gts, n_points).

        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        cost = batch_sigmoid_bce_loss(pred_instances.masks, gt_instances.masks.float())

        return cost * self.weight


@TASK_UTILS.register_module()
class MaskDiceCost:
    """Dice cost for masks.

    Args:
        weigth (float): Weight of the cost.
    """

    def __init__(self, weight: float):
        self.weight = weight

    def __call__(
        self, pred_instances: InstanceData_, gt_instances: InstanceData_, **kwargs
    ) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (InstanceData_): Predicted instances which must contain `masks` of shape (n_queries, n_points).
            gt_instances (InstanceData_): Ground truth which must contain `masks` of shape (n_gts, n_points).

        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        cost = batch_dice_loss(pred_instances.masks, gt_instances.masks.float())

        return cost * self.weight


@TASK_UTILS.register_module()
class HungarianMatcher:
    """Hungarian matcher.

    Args:
        costs (List[ConfigDict]): Cost functions.
    """

    def __init__(self, costs: List[Dict]):
        self.costs = []
        for cost in costs:
            self.costs.append(TASK_UTILS.build(cost))

    @torch.no_grad()
    def __call__(
        self, pred_instances: InstanceData_, gt_instances: InstanceData_, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Compute match cost.

        Args:
            pred_instances (InstanceData_): Predicted instances which can contain `masks` of shape (n_queries, n_points), `scores` of shape (n_queries, n_classes + 1),
            gt_instances (InstanceData_): Ground truth which can contain `labels` of shape (n_gts,), `masks` of shape (n_gts, n_points), `query_masks` of shape (n_gts, n_queries).

        Returns:
            Tuple[Tensor, Tensor]:
                * Query ids of shape (n_matched,),
                * Object ids of shape (n_matched,).
        """
        labels = gt_instances.labels
        n_gts = len(labels)
        if n_gts == 0:
            return labels.new_empty((0,)), labels.new_empty((0,))

        cost_values = []
        for cost in self.costs:
            cost_values.append(
                cost(pred_instances, gt_instances)
            )  # QueryClassificationCost, MaskBCECost, MaskDiceCost
        cost_value = torch.stack(cost_values).sum(dim=0)
        query_ids, object_ids = linear_sum_assignment(cost_value.cpu().numpy())

        return labels.new_tensor(query_ids), labels.new_tensor(object_ids)


@TASK_UTILS.register_module()
class SparseMatcher:
    """Match only queries to their including objects.

    Args:
        costs (List[Callable]): Cost functions.
        topk (int): Limit topk matches per query.
    """

    def __init__(self, costs: List[Callable], topk: int):
        self.topk = topk
        self.costs = []
        self.inf = 1e8
        for cost in costs:
            self.costs.append(TASK_UTILS.build(cost))

    @torch.no_grad()
    def __call__(
        self, pred_instances: InstanceData_, gt_instances: InstanceData_, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Compute match cost.

        Args:
            pred_instances (InstanceData_): Predicted instances which can contain `masks` of shape (n_queries, n_points), `scores` of shape (n_queries, n_classes + 1),
            gt_instances (InstanceData_): Ground truth which can contain `labels` of shape (n_gts,), `masks` of shape (n_gts, n_points), `query_masks` of shape (n_gts, n_queries).

        Returns:
            Tuple[Tensor, Tensor]:
                * Query ids of shape (n_matched,),
                * Object ids of shape (n_matched,).
        """
        labels = gt_instances.labels
        n_gts = len(labels)
        if n_gts == 0:
            return labels.new_empty((0,)), labels.new_empty((0,))

        cost_values = []
        for cost in self.costs:
            cost_values.append(cost(pred_instances, gt_instances))
        # of shape (n_queries, n_gts)
        cost_value = torch.stack(cost_values).sum(dim=0)
        cost_value = torch.where(gt_instances.query_masks, cost_value, self.inf)

        values = torch.topk(
            cost_value, self.topk + 1, dim=0, sorted=True, largest=False
        ).values[-1:, :]
        ids = torch.argwhere(cost_value < values)

        return ids[:, 0], ids[:, 1]


@TASK_UTILS.register_module()
class One2ManyMatcher:
    """Match queries to their corresponding ground truth objects using query_masks."""

    def __init__(self):
        self.inf = 1e8

    @torch.no_grad()
    def __call__(
        self, pred_instances: InstanceData_, gt_instances: InstanceData_, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Match pre masks to GT based on query_masks.

        Args:
            pred_instances (InstanceData_): Predicted instances which can contain `masks` of shape (n_queries, n_points), `scores` of shape (n_queries, n_classes + 1),
            gt_instances (InstanceData_): Ground truth which can contain `labels` of shape (n_gts,), `masks` of shape (n_gts, n_points), `query_masks` of shape (n_queries, n_gts).

        Returns:
            Tuple[Tensor, Tensor]:
                * Query ids of shape (n_queries,),
                * GT ids of shape (n_queries,) with -1 for unmatched queries.
        """
        query_masks = gt_instances.query_masks
        n_queries, n_gts = query_masks.shape
        # If there are no GTs, return all queries as unmatched (-1)
        if n_gts == 0:
            return torch.arange(n_queries), torch.full((n_queries,), -1)
        # Initialize arrays for storing results
        query_ids = torch.arange(n_queries).to(query_masks.device)
        gt_ids = torch.full((n_queries,), -1, dtype=torch.long).to(query_masks.device)
        # For each query, check if it's matched with any GT in query_masks
        matched_queries = torch.any(query_masks, dim=1).to(query_masks.device)
        gt_ids[matched_queries] = torch.argmax(
            query_masks[matched_queries].float(), dim=1
        )

        return query_ids, gt_ids


@TASK_UTILS.register_module()
class One2TopKMatcher:
    """Match queries to the corresponding ground truth objects, but only take into account
    the top k (based on IoU) best matches for each ground truth instance
    """

    def __init__(
        self,
        topk: int,
        superpoint_size_aware: bool = False,
        warmup: Optional[int] = None,
        eps: float = 1e-6,
    ):
        self.topk = topk
        self.superpoint_size_aware = superpoint_size_aware
        self.warmup = warmup
        self._use_warmup = warmup is not None
        self.eps = eps

    @torch.no_grad()
    def __call__(
        self,
        pred_instances: InstanceData_,
        gt_instances: InstanceData_,
        sp_size: Optional[Tensor] = None,
        epoch: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        if self._use_warmup:
            assert epoch is not None, (
                "'epoch' should not be 'None' when 'warmup' is set!"
            )
        pred_masks = pred_instances.masks
        gt_masks = gt_instances.masks
        query_masks = gt_instances.query_masks

        n_queries, n_gts = query_masks.shape
        device = pred_masks.device
        query_ids = torch.arange(n_queries, device=device)
        gt_ids = torch.full((n_queries,), -1, dtype=torch.long, device=device)
        if n_gts == 0:
            return query_ids, gt_ids
        # Generate initial matches the samy way as One2ManyMatcher
        matched_queries = torch.any(query_masks, dim=1).to(device)
        gt_ids[matched_queries] = torch.argmax(
            query_masks[matched_queries].float(), dim=1
        )
        # During warmup, default to One2ManyMatcher behavior
        if self._use_warmup and (epoch <= self.warmup):
            return query_ids, gt_ids
        # Compute IoU between all predicted masks (queries) and ground truth masks
        pred_masks = pred_masks.sigmoid()
        binarized_pred_masks = (pred_masks > 0.5).float()
        binarized_gt_masks = (gt_masks > 0.5).float()
        if self.superpoint_size_aware:
            # Weight by superpoint-size for more accurate IoU
            intersection = torch.einsum(
                "qn,gn,n->qg", binarized_pred_masks, binarized_gt_masks, sp_size
            )
            pred_area = torch.einsum("qn,n->q", binarized_pred_masks, sp_size)
            gt_area = torch.einsum("gn,n->g", binarized_gt_masks, sp_size)
            union = (
                pred_area.unsqueeze(1) + gt_area.unsqueeze(0) - intersection + self.eps
            )
        else:
            intersection = torch.einsum(
                "qn,gn->qg", binarized_pred_masks, binarized_gt_masks
            )
            union = (
                binarized_pred_masks.sum(dim=1, keepdim=True)
                + binarized_gt_masks.sum(dim=1)
                - intersection
                + self.eps
            )
        iou = intersection / union
        # For each ground truth, select k best queries (based on IoU)
        keep_mask = torch.zeros(n_queries, dtype=torch.bool, device=device)
        for g in range(n_gts):
            candidate_queries = torch.where(gt_ids == g)[0]
            if candidate_queries.numel() == 0:
                continue
            # IoU for candidates only
            candidate_ious = iou[candidate_queries, g]
            k = min(self.topk, candidate_queries.numel())
            topk_ind = torch.topk(candidate_ious, k=k, dim=0).indices
            selected_queries = candidate_queries[topk_ind]
            keep_mask[selected_queries] = True
        # Final mask for queries to keep. We retain:
        # 1. Queries with no corresponding ground truth instance (background, index set to -1)
        # 2. Top k matched queries for each instance
        final_mask = (gt_ids == -1) | keep_mask
        query_ids = query_ids[final_mask]
        gt_ids = gt_ids[final_mask]

        return query_ids, gt_ids
