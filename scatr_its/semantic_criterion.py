import torch

from typing import Dict, List, Optional
from mmdet3d.registry import MODELS


@MODELS.register_module()
class ForestSemanticCriterion:
    """Semantic criterion for forest data.

    Args:
        loss_weight (float): loss weight.
        seg_loss (Dict): loss config.
    """

    def __init__(
        self,
        loss_weight: float,
        seg_loss: Dict = dict(type="mmdet.CrossEntropyLoss", use_sigmoid=True),
        superpoint_size_aware: bool = False,
    ):
        self.superpoint_size_aware = superpoint_size_aware
        if self.superpoint_size_aware:
            seg_loss["reduction"] = "none"
        self.seg_loss = MODELS.build(seg_loss)
        self.loss_weight = loss_weight

    def get_layer_loss(
        self,
        layer: int,
        aux_outputs: Dict,
        insts: List,
        sp_sizes: Optional[List] = None,
    ) -> Dict:
        """Calculate loss at intermediate level.

        Args:
            layer (int): transformer layer number
            aux_outputs (Dict): Predictions with List `masks` of len batch_size, each of shape (n_points_i, n_classes + 1).
            insts (List): Ground truth of len batch_size, each InstanceData_ with `sp_masks` of shape (n_classes + 1, n_points_i).

        Returns:
            Dict: with semantic loss value.
        """
        if not self.superpoint_size_aware:
            sp_sizes = [None] * len(insts)
        pred_masks = aux_outputs["masks"]
        seg_losses = []
        for pred_mask, gt_mask, sp_size in zip(pred_masks, insts, sp_sizes):
            seg_loss_i = self.seg_loss(pred_mask.T, gt_mask.sp_masks.float().argmax(0))
            if self.superpoint_size_aware:
                # Weight loss elements by superpoint sizes
                seg_loss_i = seg_loss_i.sum(dim=1)
                seg_loss_i = (seg_loss_i * sp_size).sum() / sp_size.sum()
            else:
                seg_loss_i = seg_loss_i.mean()
            seg_losses.append(seg_loss_i)

        seg_loss = self.loss_weight * torch.mean(torch.stack(seg_losses))

        return {f"layer_{layer}_seg_loss": seg_loss}

    def __call__(
        self, pred: Dict, insts: List, sp_sizes: Optional[List] = None
    ) -> Dict:
        """Calculate loss.

        Args:
            pred (Dict): Predictions with List `masks` of len batch_size, each of shape (n_points_i, n_classes + 1).
            insts (List): Ground truth of len batch_size, each InstanceData_ with `sp_masks` of shape (n_classes + 1, n_points_i).

        Returns:
            Dict: with semantic loss value.
        """
        if not self.superpoint_size_aware:
            sp_sizes = [None] * len(insts)
        pred_masks = pred["masks"]
        seg_losses = []
        for pred_mask, gt_mask, sp_size in zip(pred_masks, insts, sp_sizes):
            seg_loss_i = self.seg_loss(pred_mask.T, gt_mask.sp_masks.float().argmax(0))
            if self.superpoint_size_aware:
                # Weight loss elements by superpoint sizes
                seg_loss_i = seg_loss_i.sum(dim=1)
                seg_loss_i = (seg_loss_i * sp_size).sum() / sp_size.sum()
            else:
                seg_loss_i = seg_loss_i.mean()
            seg_losses.append(seg_loss_i)

        seg_loss = self.loss_weight * torch.mean(torch.stack(seg_losses))
        loss = {"last_layer_seg_loss": seg_loss}

        if "aux_outputs" in pred:
            for i, aux_outputs in enumerate(pred["aux_outputs"]):
                loss_i = self.get_layer_loss(i, aux_outputs, insts, sp_sizes)
                loss.update(loss_i)

        return loss
