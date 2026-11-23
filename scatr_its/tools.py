import torch

import numpy as np

from torch import Tensor, nn
from typing import Optional, Tuple, Callable, List


def mask_matrix_nms(
    masks: Tensor,
    labels: Tensor,
    scores: Tensor,
    filter_thr: float = -1,
    nms_pre: int = -1,
    max_num: int = -1,
    kernel: str = "gaussian",
    sigma: float = 2.0,
    mask_area: Optional[Tensor] = None,
    superpoint_sizes: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Matrix NMS for multi-class masks. (a copy of mmdet/models/layers/matrix_nms.py. with the input shape of the `masks` tensor changed.)

    Args:
        masks (Tensor): Has shape (num_instances, m)
        labels (Tensor): Labels of corresponding masks, has shape (num_instances,).
        scores (Tensor): Mask scores of corresponding masks, has shape (num_instances).
        filter_thr (float): Score threshold to filter the masks after matrix nms, -1 = do not use. Defaults to -1.
        nms_pre (int): The max number of instances to do the matrix nms, -1 = do not use. Defaults to -1.
        max_num (int): If there are more than max_num masks after matrix, only top max_num will be kept, -1 = do not use. Defaults to -1.
        kernel (str): 'linear' or 'gaussian'. Defaults to "gaussian"
        sigma (float): std in gaussian method. Defaults to 2.0.
        mask_area (Tensor): The sum of seg_masks. Defaults to None.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: Processed mask results.
            - scores (Tensor): Updated scores, has shape (n,).
            - labels (Tensor): Remaining labels, has shape (n,).
            - masks (Tensor): Remaining masks, has shape (n, m).
            - keep_inds (Tensor): The indices of the remaining mask in the input mask, has shape (n,).
    """
    assert len(labels) == len(masks) == len(scores)
    if len(labels) == 0:
        return (
            scores.new_zeros(0),
            labels.new_zeros(0),
            masks.new_zeros(0, *masks.shape[-1:]),
            labels.new_zeros(0),
        )
    if mask_area is None:
        if superpoint_sizes is not None:
            mask_area = (masks * superpoint_sizes).sum(1).float()
        else:
            mask_area = masks.sum(1).float()
    else:
        assert len(masks) == len(mask_area)

    # sort and keep top nms_pre
    scores, sort_inds = torch.sort(scores, descending=True)

    keep_inds = sort_inds
    if nms_pre > 0 and len(sort_inds) > nms_pre:
        sort_inds = sort_inds[:nms_pre]
        keep_inds = keep_inds[:nms_pre]
        scores = scores[:nms_pre]
    masks = masks[sort_inds]
    mask_area = mask_area[sort_inds]
    labels = labels[sort_inds]

    num_masks = len(labels)
    if superpoint_sizes is not None:
        weighted_masks = masks * superpoint_sizes.float()
        inter_matrix = weighted_masks.matmul(masks.T)
    else:
        flatten_masks = masks.reshape(num_masks, -1).float()
        inter_matrix = torch.mm(flatten_masks, flatten_masks.transpose(1, 0))
    expanded_mask_area = mask_area.expand(num_masks, num_masks)
    # Upper triangle iou matrix.
    iou_matrix = (
        inter_matrix
        / (expanded_mask_area + expanded_mask_area.transpose(1, 0) - inter_matrix)
    ).triu(diagonal=1)
    # label_specific matrix.
    expanded_labels = labels.expand(num_masks, num_masks)
    # Upper triangle label matrix.
    label_matrix = (expanded_labels == expanded_labels.transpose(1, 0)).triu(diagonal=1)
    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(num_masks, num_masks).transpose(1, 0)
    # IoU decay
    decay_iou = iou_matrix * label_matrix
    # Calculate the decay_coefficient
    if kernel == "gaussian":
        decay_matrix = torch.exp(-1 * sigma * (decay_iou**2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou**2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == "linear":
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError(f"{kernel} kernel is not supported in matrix nms!")
    # update the score.
    scores = scores * decay_coefficient

    if filter_thr > 0:
        keep = scores >= filter_thr
        keep_inds = keep_inds[keep]
        if not keep.any():
            return (
                scores.new_zeros(0),
                labels.new_zeros(0),
                masks.new_zeros(0, *masks.shape[-1:]),
                labels.new_zeros(0),
            )
        masks = masks[keep]
        scores = scores[keep]
        labels = labels[keep]

    # sort and keep top max_num
    scores, sort_inds = torch.sort(scores, descending=True)
    keep_inds = keep_inds[sort_inds]
    if max_num > 0 and len(sort_inds) > max_num:
        sort_inds = sort_inds[:max_num]
        keep_inds = keep_inds[:max_num]
        scores = scores[:max_num]
    masks = masks[sort_inds]
    labels = labels[sort_inds]

    return scores, labels, masks, keep_inds


def get_iou(
    inputs: Tensor,
    targets: Tensor,
    ratio: Optional[float],
    sp_size: Optional[Tensor] = None,
) -> Tensor:
    """IoU for to equal shape masks.

    Args:
        inputs (Tensor): of shape (n_gts, n_points).
        targets (Tensor): of shape (n_gts, n_points).

    Returns:
        Tensor: IoU of shape (n_gts,).
    """
    if ratio is None:
        # Assume that instance is fully visible if ratio is None
        ratio = 1.0
    inputs = inputs.sigmoid()
    binarized_inputs = (inputs >= 0.5).float()
    targets = (targets > 0.5).float()
    if sp_size is not None:
        # Weight by superpoint size
        intersection = (binarized_inputs * targets * sp_size).sum(-1)
        union = (
            (targets * sp_size).sum(-1) / ratio
            + (binarized_inputs * sp_size).sum(-1)
            - intersection
        )
    else:
        intersection = (binarized_inputs * targets).sum(-1)
        union = targets.sum(-1) / ratio + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)

    return score


def generate_cylindrical_regions(
    points: Tensor, step_size: float
) -> List[Tuple[float, float]]:
    """Generate a list of cylinder centers for inference

    Args:
        points (Tensor): Original point coordinates.
        step_size (float): Step size for the cylindrical moving window during inference.

    Returns:
        List[Tuple[float, float]]: List of (x, y) coordinate corresponding to cylinder centers
    """
    x_coords = points[:, 0].cpu()
    y_coords = points[:, 1].cpu()
    x_min, x_max = x_coords.min().item(), x_coords.max().item()
    y_min, y_max = y_coords.min().item(), y_coords.max().item()
    regions = []
    x = x_min
    while x <= x_max:
        y = y_min
        while y <= y_max:
            regions.append((x, y))
            y += step_size
        x += step_size

    return regions


def generate_cylindrical_regions_max_stride(
    points: Tensor,
    radius: float,
    voxel_size: float,
    max_object_size: float,
    stride_alpha: float = 0.9,
    downsample_n_points: Optional[int] = 500_000,
) -> List[Tuple[float, float]]:
    """Generate a list of cylinder centers for inference with maximum stride that attempts to minimize the number of cylinders (while covering the whole input area)

    Args:
        points (Tensor): Original point coordinates.
        radius (float): Radius of the cylinders.
        voxel_size (float): Voxel size used by the model.
        max_object_size (float): Estimated maximum width of objects in the data (in meters).
        downsample_n_points (Optional[int]): Number of points to downsample to before estimating whether entire plot can fit within cylinder. Defaults to 500_000.

    Raises:
        ValueError: _description_

    Returns:
        List[Tuple[float, float]]: _description_
    """
    border_th = 2 * voxel_size
    usable_radius = radius - border_th - max_object_size
    if usable_radius <= 0:
        raise ValueError(
            "Usable radius <= 0! Reduce 'voxel_size', 'max_object_size' or increase input radius."
        )
    # Optimal step size for full coverage
    step_size = stride_alpha * np.sqrt(2) * usable_radius
    # Extract point cloud bounding box
    x_coords = points[:, 0].cpu()
    y_coords = points[:, 1].cpu()
    x_min, x_max = x_coords.min().item(), x_coords.max().item()
    y_min, y_max = y_coords.min().item(), y_coords.max().item()
    # First check if the bounding box of the entire plot could be fit inside the radius
    x_center = x_min + 0.5 * (x_max - x_min)
    y_center = y_min + 0.5 * (y_max - y_min)
    bb_fits_cylinder = True
    for x in [x_min, x_max]:
        for y in [y_min, y_max]:
            dist = np.sqrt((x_center - x) ** 2 + (y_center - y) ** 2)
            if dist >= radius:
                bb_fits_cylinder = False
    # Also check if the entire plot could be fit inside the radius if we place the cylinder at its (approximate) center
    n_points = len(x_coords)
    downsample_n_points = min(downsample_n_points, n_points)
    downsample_ind = torch.randint(
        0, n_points, (downsample_n_points,), device=x_coords.device
    )
    x_coords_downsampled = x_coords[downsample_ind]
    y_coords_downsampled = y_coords[downsample_ind]
    max_dist_center = (
        torch.sqrt(
            (x_coords_downsampled - x_center) ** 2
            + (y_coords_downsampled - y_center) ** 2
        )
        .max()
        .item()
    )
    plot_fits_centered_cylinder = max_dist_center <= radius

    centers = []
    if bb_fits_cylinder or plot_fits_centered_cylinder:
        centers.append((x_center, y_center))
        return centers

    # Start at exact bounding box edge
    cx = x_min
    while cx <= x_max:
        cy = y_min
        while cy <= y_max:
            centers.append((cx, cy))
            cy += step_size
        # Ensure last row covers top edge
        if cy - step_size < y_max:
            # Center of last cylinder is below the upper border of the bounding box, check if full are can be covered
            extreme_point_x = cx + 0.5 * step_size
            dist_extreme_point = np.sqrt(
                (extreme_point_x - cx) ** 2 + (y_max - (cy - step_size)) ** 2
            )
            if dist_extreme_point >= usable_radius:
                centers.append((cx, cy))

        cx += step_size
    # Ensure last column covers right edge
    if cx - step_size < x_max:
        cy = y_min
        extreme_point_y = cy + 0.5 * step_size
        dist_extreme_point = np.sqrt(
            (x_max - (cx - step_size)) ** 2 + (extreme_point_y - cy) ** 2
        )
        if dist_extreme_point >= usable_radius:
            while cy <= y_max:
                centers.append((cx, cy))
                cy += step_size
            if cy - step_size < y_max:
                extreme_point_x = cx + 0.5 * step_size
                dist_extreme_point = np.sqrt(
                    (extreme_point_x - cx) ** 2 + (y_max - (cy - step_size)) ** 2
                )
                if dist_extreme_point >= usable_radius:
                    centers.append((cx, cy))

    return centers


class BaseModule(nn.Module):
    """Base module class with some basic additions to the pytorch Module class"""

    @property
    def nb_params(self):
        """This property is used to return the number of trainable parameters for a given layer
        It is useful for debugging and reproducibility.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self._nb_params = sum([np.prod(p.size()) for p in model_parameters])

        return self._nb_params


def MLP(
    channels: List[int],
    activation: Callable = nn.LeakyReLU(0.2),
    bn_momentum: float = 0.1,
    bias: bool = True,
) -> nn.Sequential:
    """Basic MLP consisting of linear layers, batch normalization and activation

    Args:
        channels (List[int]): List of input channel sizes
        activation (Callable, optional): Activation function. Defaults to nn.LeakyReLU(0.2).
        bn_momentum (float, optional): Batch norm momentum. Defaults to 0.1.
        bias (bool, optional): If True, use bias in linear layers. Defaults to True.

    Returns:
        nn.Sequential: MLP as a nn.Sequential object
    """
    return nn.Sequential(
        *[
            nn.Sequential(
                nn.Linear(channels[i - 1], channels[i], bias=bias),
                FastBatchNorm1d(channels[i], momentum=bn_momentum),
                activation,
            )
            for i in range(1, len(channels))
        ]
    )


class FastBatchNorm1d(BaseModule):
    def __init__(self, num_features: int, momentum: float = 0.1, **kwargs):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features, momentum=momentum, **kwargs)

    def _forward_dense(self, x: Tensor):
        return self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)

    def _forward_sparse(self, x: Tensor):
        """Batch norm 1D is not optimised for 2D tensors. The first dimension is supposed to be
        the batch and therefore not very large. So we introduce a custom version that leverages BatchNorm1D
        in a more optimised way
        """
        x = x.unsqueeze(2)
        x = x.transpose(0, 2)
        x = self.batch_norm(x)
        x = x.transpose(0, 2)

        return x.squeeze(dim=2)

    def forward(self, x: Tensor):
        if x.dim() == 2:
            return self._forward_sparse(x)
        elif x.dim() == 3:
            return self._forward_dense(x)
        else:
            raise ValueError("Non supported number of dimensions {}".format(x.dim()))


class Seq(nn.Sequential):
    def __init__(self):
        super().__init__()
        self._num_modules = 0

    def append(self, module: nn.Module):
        self.add_module(str(self._num_modules), module)
        self._num_modules += 1

        return self
