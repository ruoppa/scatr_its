import torch

import torch.nn.functional as F

from torch import Tensor, nn
from typing import Callable, Dict
from mmdet3d.registry import MODELS


@MODELS.register_module()
class DynamicQueryModule(nn.Module):
    def __init__(
        self,
        bins: Dict[int, int],
        in_channels: int,
        hidden_channels: int = 128,
        ordinal: bool = False,
        activation: Callable = nn.LeakyReLU(0.2),
        bias: bool = True,
    ):
        """Module for dynamically predicting a query count suitable for the density of the given scene

        Args:
            bins (Dict[int, int]): Lower bound of each bin (instance count) and the corresponding query count. The lower bound of the next bin serves as the upper bound for each bin, with inf used for the last bin.
            in_channels (int): Dimensionality of the input features.
            hidden_channels (int, optional): Number of hidden channels. Defaults to 128.
            ordinal (bool, optional): If True, use ordinal classification. Defaults to False.
            activation (Callable, optional): Actication function for the MLP. Defaults to nn.LeakyReLU(0.2).
            bias (bool, optional): If True, use a bias in the linear layers of the MLP. Defaults to True.
        """
        super().__init__()
        self.in_channels = in_channels
        self.bins = bins
        bin_thresholds = torch.tensor(list(bins.keys()))
        bin_values = torch.tensor(list(bins.values()))
        sort_ind = torch.argsort(bin_thresholds)
        self.register_buffer("bin_thresholds", bin_thresholds[sort_ind])
        self.register_buffer("bin_values", bin_values[sort_ind])
        assert self.bin_thresholds[0] == 0, (
            f"The lowest density threshold should be zero, got '{self.bin_thresholds[0]}' instead!"
        )
        assert self.bin_values[0] >= 0, (
            f"The lowest query count should be at least zero, got '{self.bin_values[0]}' instead!"
        )
        self.num_bins = len(bins)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=bias),
            activation,
            nn.Linear(hidden_channels, hidden_channels, bias=bias),
            activation,
        )
        self._ordinal = ordinal
        if ordinal:
            self.out = nn.Linear(hidden_channels, self.num_bins - 1, bias=bias)
        else:
            self.out = nn.Linear(hidden_channels, self.num_bins, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            # No batch -> add extra dimension of size 1
            x = x.unsqueeze(0)
        # Mean pooling to shape (B, in_channels)
        x = x.mean(dim=1)
        logits = self.mlp(x)
        logits = self.out(logits)

        return logits

    def _predict_with_logits(self, logits: Tensor) -> Tensor:
        if self._ordinal:
            probs = torch.sigmoid(logits)
            pred_class = (probs > 0.5).sum(dim=1)
        else:
            probs = torch.softmax(logits, dim=-1)
            pred_class = probs.argmax(dim=1)

        return pred_class

    @torch.no_grad()
    def get_targets(
        self, gt_instance_labels: Tensor, non_instance_label: int = 0
    ) -> Tensor:
        if gt_instance_labels.dim() == 1:
            # No batch -> add extra dimension of size 1
            gt_instance_labels = gt_instance_labels.unsqueeze(0)
        targets = []
        for i in range(len(gt_instance_labels)):
            unq_instance_labels = torch.unique(gt_instance_labels[i])
            gt_instance_count = (
                len(unq_instance_labels[unq_instance_labels != non_instance_label])
            )
            if self._ordinal:
                target_i = (self.bin_thresholds[1:] <= gt_instance_count).float()
                targets.append(target_i)
            else:
                # Thresholds are sorted -> index of last threshold that is at least the gt instance
                # count corresponds to the density class
                target_i = torch.nonzero(self.bin_thresholds <= gt_instance_count)[-1]
                targets.append(target_i)

        if self._ordinal:
            targets = torch.vstack(targets)
        else:
            targets = torch.hstack(targets)

        return targets

    def get_query_counts(self, logits: Tensor) -> Tensor:
        pred_class = self._predict_with_logits(logits)
        pred_query_count = self.bin_values[pred_class]

        return pred_query_count

    def get_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        if self._ordinal:
            loss = F.binary_cross_entropy_with_logits(logits, targets)
        else:
            loss = F.cross_entropy(logits, targets)

        return loss