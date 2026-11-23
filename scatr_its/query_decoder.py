import torch
import torch.nn as nn

from torch import Tensor
from typing import List, Optional, Union, Tuple, Dict
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS


class CrossAttentionLayer(BaseModule):
    """Cross attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
        fix (bool): If True, apply dropout and layer normalization. Defaults to False.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float, fix: bool = False):
        super().__init__()
        self.fix = fix
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # todo: why BaseModule doesn't call it without us?
        self.init_weights()

    def init_weights(self):
        """Init weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        queries: List[Tensor],
        keys: List[Tensor],
        values: List[Tensor],
        attn_masks: Optional[List[Tensor]] = None,
    ) -> List[Tensor]:
        """Forward pass.

        Args:
            sources (List[Tensor]): of len batch_size, each of shape (n_points_i, d_model).
            queries (List[Tensor]): of len batch_size, each of shape(n_queries_i, d_model).
            attn_masks (Optional[List[Tensor]]): of len batch_size, each of shape (n_queries, n_points). Defaults to None.

        Return:
            List[Tensor]: Queries of len batch_size, each of shape(n_queries_i, d_model).
        """
        outputs = []
        for i in range(len(keys)):
            attn_mask = attn_masks[i] if attn_masks is not None else None
            output, _ = self.attn(queries[i], keys[i], values[i], attn_mask=attn_mask)
            if self.fix:
                output = self.dropout(output)
            output = output + queries[i]
            if self.fix:
                output = self.norm(output)
            outputs.append(output)

        return outputs


class SelfAttentionLayer(BaseModule):
    """Self attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: List[Tensor], values: List[Tensor]) -> List[Tensor]:
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size, each of shape(n_queries_i, d_model).

        Returns:
            List[Tensor]: Queries of len batch_size, each of shape(n_queries_i, d_model).
        """
        out = []
        for i in range(len(queries)):
            z, _ = self.attn(queries[i], queries[i], values[i])
            z = self.dropout(z) + values[i]
            z = self.norm(z)
            out.append(z)

        return out


class FFN(BaseModule):
    """Feed forward network.

    Args:
        d_model (int): Model dimension.
        hidden_dim (int): Hidden dimension.
        dropout (float): Dropout rate.
        activation_fn (str): 'relu' or 'gelu'.
    """

    _VALID_FNS = ["relu", "gelu"]

    def __init__(
        self, d_model: int, hidden_dim: int, dropout: float, activation_fn: str
    ):
        super().__init__()

        assert activation_fn in self._VALID_FNS, (
            f"'activation_fn' should be one of '{self._VALID_FNS}', got '{activation_fn}' instead!"
        )
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU() if activation_fn == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size, each of shape(n_queries_i, d_model).

        Returns:
            List[Tensor]: Queries of len batch_size, each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z = self.net(y)
            z = z + y
            z = self.norm(z)
            out.append(z)

        return out


@MODELS.register_module()
class QueryDecoder(BaseModule):
    """SCATR-ITS query decoder.

    Args:
        num_layers (int): Number of transformer layers.
        num_instance_queries (int): Number of instance queries.
        num_semantic_queries (int): Number of semantic queries.
        num_classes (int): Number of classes.
        in_channels (int): Number of input channels.
        d_model (int): Number of channels for model layers.
        num_heads (int): Number of head in attention layer.
        hidden_dim (int): Dimension of attention layer.
        dropout (float): Dropout rate for transformer layer.
        activation_fn (str): 'relu' of 'gelu'.
        iter_pred (bool): Whether to predict iteratively.
        attn_mask (bool): Whether to use mask attention.
        fix_attention (bool): If True, apply dropout and layer normalization in CrossAttentionLayer.
        objectness_flag (bool): If True, output instance scores.
        instance_only (bool, optional): If True, do not add learnable semantic queries and only use the provided non-parametric instance queries. Defaults to False.
    """
    def __init__(
        self,
        num_layers: int,
        num_semantic_queries: int,
        in_channels: int,
        d_model: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        activation_fn: str,
        iter_pred: bool,
        attn_mask: bool,
        fix_attention: bool,
        objectness_flag: bool,
        instance_only: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.objectness_flag = objectness_flag
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.LayerNorm(d_model), nn.ReLU()
        )
        self.query_proj = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.instance_only = instance_only
        if not instance_only:
            self.num_semantic_queries = num_semantic_queries
            self.semantic_queries = nn.Embedding(num_semantic_queries, in_channels)
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.cross_attn_layers.append(
                CrossAttentionLayer(d_model, num_heads, dropout, fix_attention)
            )
            self.self_attn_layers.append(
                SelfAttentionLayer(d_model, num_heads, dropout)
            )
            self.ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        if objectness_flag:
            self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
            )
        self.x_mask = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask

    def _get_queries(
        self,
        queries: Optional[List[Tensor]] = None,
        batch_size: Optional[int] = None,
    ) -> List[Tensor]:
        """Get query tensor.

        Args:
            queries (Optional[List[Tensor]]): of len batch_size, each of shape (n_queries_i, in_channels).
            batch_size (Optional[int]): batch size.

        Returns:
            List[Tensor]: of len batch_size, each of shape (n_queries_i, d_model).
        """
        if batch_size is None:
            batch_size = len(queries)

        result_queries = []

        for i in range(batch_size):
            result_query = []
            if hasattr(self, "query"):
                result_query.append(self.query.weight)
            if (queries is not None):
                if self.instance_only:
                    # No semantic queries
                    concat_queries = queries[i]
                else:
                    semantic_queries = self.semantic_queries.weight
                    # Concat queries[i] and semantic_queries
                    if len(queries[i]) == 0:
                        queries[i] = torch.empty(0, *semantic_queries.shape[1:]).to(
                            queries[i].device
                        )
                    concat_queries = torch.cat(
                        (queries[i], semantic_queries.to(queries[i].device)), dim=0
                    )
                result_query.append(self.query_proj(concat_queries))
            result_queries.append(torch.cat(result_query))

        return result_queries

    def _forward_head(
        self, queries: Union[List[Tensor], Tensor], mask_feats: List[Tensor]
    ) -> Tuple[List[Optional[Tensor]], List[Tensor], Optional[List[Tensor]]]:
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size, each of shape (n_queries_i, d_model). Or tensor of shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size, each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Confidence scores of len batch_size, each of shape (n_queries_i, 1) (filled with None if objectness_flag is not set).
                List[Tensor]: Predicted masks of len batch_size, each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size, each of shape (n_queries_i, n_points_i).
        """
        pred_scores, pred_masks, attn_masks = [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            pred_score = self.out_score(norm_query) if self.objectness_flag else None
            pred_scores.append(pred_score)
            pred_mask = torch.einsum("nd,md->nm", norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                mask = attn_mask.sum(-1) == attn_mask.shape[-1]
                # expand mask to match last dimension
                mask = mask.unsqueeze(-1).expand_as(attn_mask)
                attn_mask[mask] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None

        return (
            pred_scores,
            pred_masks,
            attn_masks,
        )

    def _forward_simple(
        self,
        feats: List[Tensor],
        queries: Optional[List[Tensor]] = None,
    ) -> Dict:
        """Simple forward pass.

        Args:
            feats (List[Tensor]): of len batch_size, each of shape (n_points_i, in_channels).
            queries (Optional[List[Tensor]]): of len batch_size, each of shape (n_points_i, in_channels). Defaults to None.

        Returns:
            Dict: with labels, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in feats]
        mask_feats = [self.x_mask(y) for y in feats]
        queries = self._get_queries(queries, len(feats))

        for i in range(len(self.cross_attn_layers)):
            inst_feats_orig = [f.clone() for f in inst_feats]
            queries = self.cross_attn_layers[i](queries, inst_feats, inst_feats_orig)
            queries_orig = [q.clone() for q in queries]
            queries = self.self_attn_layers[i](queries, queries_orig)
            queries = self.ffn_layers[i](queries)
        pred_scores, pred_masks, _ = self._forward_head(queries, mask_feats)

        return dict(
            masks=pred_masks,
            scores=pred_scores,
        )

    def _forward_iter_pred(
        self,
        feats: List[Tensor],
        queries: Optional[List[Tensor]] = None,
    ) -> Dict:
        """Iterative forward pass.

        Args:
            feats (List[Tensor]): of len batch_size, each of shape (n_points_i, in_channels).
            queries (Optional[List[Tensor]]): of len batch_size, each of shape (n_points_i, in_channels).

        Returns:
            Dict: with labels, masks, scores, and aux_outputs.
        """
        pred_scores, pred_masks = [], []
        inst_feats = [self.input_proj(y) for y in feats]
        mask_feats = [self.x_mask(y) for y in feats]
        queries = self._get_queries(queries, len(feats))
        pred_score, pred_mask, attn_mask = self._forward_head(queries, mask_feats)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](
                queries, inst_feats, inst_feats, attn_mask
            )
            queries = self.self_attn_layers[i](queries, queries)
            queries = self.ffn_layers[i](queries)
            pred_score, pred_mask, attn_mask = self._forward_head(queries, mask_feats)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            {"masks": masks, "scores": scores}
            for scores, masks in zip(pred_scores[:-1], pred_masks[:-1])
        ]

        return dict(
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs,
        )

    def forward(
        self,
        feats: List[Tensor],
        queries: Optional[List[Tensor]] = None,
    ) -> Dict:
        """Forward pass.

        Args:
            feats (List[Tensor]): of len batch_size, each of shape (n_points_i, in_channels).
            queries (Optional[List[Tensor]]): of len batch_size, each of shape (n_queries_i, in_channels).

        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        if self.iter_pred:
            return self._forward_iter_pred(feats, queries)
        else:
            return self._forward_simple(feats, queries)
