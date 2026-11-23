import torch

import torch.nn.functional as F

from torch import Tensor
from typing import Dict, Optional, Tuple
from itertools import filterfalse
from torch_scatter import scatter


def discriminative_loss(
    embedding_logits: Tensor,
    gt_instance_masks: Tensor,
    batch: Tensor,
    feature_dim: int,
) -> Dict[str, Tensor]:
    loss = []
    loss_var = []
    loss_dist = []
    loss_reg = []
    batch_size = torch.unique(batch)
    for s in batch_size:
        batch_mask = batch == s
        sample_gt_instances = gt_instance_masks[batch_mask]
        sample_embed_logits = embedding_logits[batch_mask]
        sample_loss, sample_loss_var, sample_loss_dist, sample_loss_reg = (
            _discriminative_loss_single(
                sample_embed_logits, sample_gt_instances, feature_dim
            )
        )
        loss.append(sample_loss)
        loss_var.append(sample_loss_var)
        loss_dist.append(sample_loss_dist)
        loss_reg.append(sample_loss_reg)
    loss = torch.stack(loss)
    loss_var = torch.stack(loss_var)
    loss_dist = torch.stack(loss_dist)
    loss_reg = torch.stack(loss_reg)

    return {
        "ins_loss": torch.mean(loss),
        "ins_var_loss": torch.mean(loss_var),
        "ins_dist_loss": torch.mean(loss_dist),
        "ins_reg_loss": torch.mean(loss_reg),
    }


def _discriminative_loss_single(
    prediction: Tensor,
    ground_truth: Tensor,
    feature_dim: int,
    delta_v: float = 0.5,
    delta_d: float = 1.5,
    param_var: float = 1.0,
    param_dist: float = 1.0,
    param_reg: float = 0.001,
) -> Tensor:
    """Discriminative loss for a single prediction/label pair.

    Args:
        prediction (Tensor): Inference of network.
        ground_truth (Tensor): Ground truth instance labels.
        feature_dim (int): Feature dimension of prediction.
        delta_v (float, optional): Cutoff variance distance. Defaults to 0.5.
        delta_d (float, optional): Cutoff cluster distance. Defaults to 1.5.
        param_var (float, optional): Weight for intra cluster variance. Defaults to 1.0.
        param_dist (float, optional): Weight for inter cluster variance. Defaults to 1.0.
        param_reg (float, optional): Weight regularization. Defaults to 0.001.

    Returns:
        Tensor: discriminative loss
    """

    # Reshape so pixels are aligned along a vector
    reshaped_pred = torch.reshape(prediction, (-1, feature_dim))
    # Count instances
    unique_labels, unique_id, counts = torch.unique(
        ground_truth, return_inverse=True, return_counts=True
    )
    num_instances = unique_labels.size()
    segmented_sum = scatter(reshaped_pred, unique_id, dim=0, reduce="sum")

    mu = torch.div(segmented_sum, (torch.reshape(counts, (-1, 1)) + 1e-8))
    unique_id_t = unique_id.unsqueeze(1)

    unique_id_t = unique_id_t.expand(unique_id_t.size()[0], mu.size()[-1])
    mu_expand = torch.gather(mu, 0, unique_id_t)

    tmp_distance = reshaped_pred - mu_expand
    distance = torch.norm(tmp_distance, p=1, dim=1)
    distance = torch.subtract(distance, delta_v)
    distance = torch.clip(distance, min=0.0)
    distance = torch.square(distance)
    l_var = scatter(distance, unique_id, dim=0, reduce="sum")
    l_var = torch.div(l_var, counts + 1e-8)
    l_var = torch.sum(l_var)
    l_var = torch.div(l_var, float(num_instances[0]))
    # Get distance for each pair of clusters
    mu_interleaved_rep = mu.repeat(num_instances[0], 1)
    mu_band_rep = mu.repeat(1, num_instances[0])
    mu_band_rep = torch.reshape(
        mu_band_rep, (num_instances[0] * num_instances[0], feature_dim)
    )
    mu_diff = torch.subtract(mu_band_rep, mu_interleaved_rep)
    # Filter out zeros from same cluster subtraction
    eye = torch.eye(num_instances[0])
    diff_cluster_mask = torch.eq(eye, 0)
    diff_cluster_mask = torch.reshape(diff_cluster_mask, (-1,))
    mu_diff_bool = mu_diff[diff_cluster_mask]
    mu_norm = torch.norm(mu_diff_bool, p=1, dim=1)
    mu_norm = torch.subtract(torch.mul(delta_d, 2.0), mu_norm)
    mu_norm = torch.clip(mu_norm, min=0.0)
    mu_norm = torch.square(mu_norm)

    l_dist = torch.mean(mu_norm)

    if num_instances[0] == 1:
        l_dist = torch.tensor(0).cuda()
    l_reg = torch.mean(torch.norm(mu, p=1, dim=1))

    if num_instances[0] == 0:
        l_var = torch.tensor(0).cuda()
        l_dist = torch.tensor(0).cuda()
        l_reg = torch.tensor(0).cuda()

    param_scale = 1.0
    l_var = param_var * l_var
    l_dist = param_dist * l_dist
    l_reg = param_reg * l_reg

    loss = param_scale * (l_var + l_dist + l_reg)

    return loss, l_var, l_dist, l_reg


def batch_sigmoid_bce_loss(inputs, targets):
    """Sigmoid BCE loss.

    Args:
        inputs: of shape (n_queries, n_points).
        targets: of shape (n_gts, n_points).

    Returns:
        Tensor: Loss of shape (n_queries, n_gts).
    """
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    pos_loss = torch.einsum("nc,mc->nm", pos, targets)
    neg_loss = torch.einsum("nc,mc->nm", neg, (1 - targets))

    return (pos_loss + neg_loss) / inputs.shape[1]


def batch_dice_loss(inputs: Tensor, targets: Tensor):
    """Dice loss.

    Args:
        inputs: of shape (n_queries, n_points).
        targets: of shape (n_gts, n_points).

    Returns:
        Tensor: Loss of shape (n_queries, n_gts).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)

    return loss


def dice_loss(inputs: Tensor, targets: Tensor, sp_size: Optional[Tensor] = None):
    """Compute the DICE loss, similar to generalized IoU for masks.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs.
            Stores the binary classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        Tensor: loss value.
    """
    inputs = inputs.sigmoid()

    if sp_size is not None:
        # Weight elements by superpoint size
        numerator = 2 * (inputs * targets * sp_size).sum(-1)
        denominator = (inputs * sp_size).sum(-1) + (targets * sp_size).sum(-1)
    else:
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)

    return loss.mean()


def lovasz_hinge_loss(
    logits: Tensor, labels: Tensor, per_scene: bool = True, ignore: Optional[int] = None
) -> Tensor:
    """ Compute binary Lovasz hinge loss (see: https://arxiv.org/abs/1705.08790)

    Args:
        logits (Tensor): Logits at each point/voxel (between -infinity and +infinity). Of shape (B, N).
        labels: (Tensor): Binary ground truth masks (0 or 1) for each point. Of shape (B, N).
        per_scene (bool): Compute the loss per scene instead of per batch
        ignore (Optional[int]): class id for points to ignore during loss computation. Defaults to None.
    
    Returns:
        Tensor: loss value.
    """
    if per_scene:
        loss = _mean(
            _lovasz_hinge_flat(
                *_flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore)
            )
            for log, lab in zip(logits, labels)
        )
    else:
        loss = _lovasz_hinge_flat(*_flatten_binary_scores(logits, labels, ignore))

    return loss


def _lovasz_grad(gt_sorted: Tensor) -> Tensor:
    """Compute gradient of the Lovasz extension w.r.t sorted errors. See Alg. 1 in paper

    Args:
        gt_sorted (Tensor): Ground trut labels sorted in a descending order based on the corresponding logit error magnitudes.

    Returns:
        Tensor: Gradient of the lovasz extension of the Jaccard loss.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]

    return jaccard


def _lovasz_hinge_flat(logits: Tensor, labels: Tensor) -> Tensor:
    """Binary Lovasz hinge loss
    
    Args:
        logits (Tensor): Logits at each prediction (between -infinity and +infinity). Of shape (N,).
        labels (Tensor): Binary ground truth labels (0 or 1). Of shape (N,).
    
    Returns:
        Tensor: Binary lovasz hinge loss for the given scene.
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)

    return loss


def _flatten_binary_scores(
    scores: Tensor, labels: Tensor, ignore: Optional[int] = None
) -> Tuple[Tensor, Tensor]:
    """Flatten multi dimensional tensors of logits/labels into 1D.

    Args:
        scores (Tensor): Logits for a signle scene.
        labels (Tensor): Binary labels for a signle scene.
        ignore (Optional[int], optional): Class id for points to ignore during loss computation. Defaults to None.

    Returns:
        Tuple[Tensor, Tensor]: flattened and filtered scores and labels.
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]

    return vscores, vlabels


def _isnan(x) -> bool:
    return x != x


def _mean(values: Tensor, ignore_nan: bool = False) -> Tensor:
    values = iter(values)
    if ignore_nan:
        values = filterfalse(_isnan, values)
    try:
        n = 1
        acc = next(values)
    except StopIteration:
        raise ValueError("Empty mean!")
    for n, v in enumerate(values, 2):
        acc += v
    if n == 1:
        return acc

    return acc / n
