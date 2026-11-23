import os
import torch

import numpy as np
import spconv.pytorch as spconv
import MinkowskiEngine as ME
import torch.nn.functional as F

from tqdm import tqdm
from torch import Tensor
from spconv.pytorch import SparseConvTensor
from typing import List, Optional, Dict, Tuple
from scatr_its.tools import (
    Seq,
    MLP,
    generate_cylindrical_regions,
    generate_cylindrical_regions_max_stride,
)
from scatr_its.losses import discriminative_loss, lovasz_hinge_loss
from torch_graph_components import merge_components_by_contour_prior
from torch_cluster import knn_graph
from mmengine.config import Config
from mmdet3d.structures import PointData, Det3DDataSample
from mmengine.logging import MMLogger, print_log
from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS


@MODELS.register_module()
class SuperpointGenerator(Base3DDetector):
    """Superpoint generator for the ScaTR-ITS model.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): Number of output channels.
        voxel_size (float): Voxel size.
        num_semantic_classes (int): Number of classes.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        backbone (Optional[Config]): Config dict of the backbone.
        test_cfg (Config): Config dict of test hyper-parameters.
        fg_semantic_class (int): Semantic class for foreground points and class label for all instances. Defaults to 1.
        data_preprocessor (Optional[Config]): The pre-process config of :class:`BaseDataPreprocessor`.
        init_cfg (Optional[Config]): The config to control the initialization. Defaults to None.
        radius (float): Radius of cylinders sampled during trainingin (in meters). Defaults to 16.
        step_size_ratio (float): Step size for inference is radius * step_size_ratio. Defaults to 0.25.
        num_points (int): Maximum number of points after voxelization. Downsampled if above this. Defaults to 640000.
        score_th (float): Score threshold for instances. Defaults to 0.4.
        chunk_size (int): Chunk size. Defaults to 20_000.
        reg_lambda_fg (float): Regularization parameter for foreground superpoint generation. Defaults to 0.003.
        reg_lambda_bg (float): Regularization parameter for background superpoint generation. Defaults to 0.3.
        min_size (int): Minimum size for superpoints. Defaults to 1.
        graph_k (int): Number of neighbors used for generating the kNN graph representation of the point cloud. Defaults to 8.
        separate_bg_fg (bool): If True, process (predicted) foreground and background points separately when generating the superpoints. Defaults to True.
        spatial_weight (Optional[float]): If set, use as scalar weight for xyz coordinates when generating superpoint (concatenated with feature embedings). If None, only used feature emebedings. Defaults to None.
        edge_weight_mode (str): Method used for computing edge weights in the kNN graph. Defaults to "unit".
        max_iterations (Optional[int]): Maximum number of iterations in the superpoint generation algorithm. Defaults to None.
        force_rewrite (bool): If True, force generating superpoints even if they already exist for a given point cloud. When false, files with corresponding existing superpoints are skipped. Defaults to False.
        test_mode (bool): If True, run predicting in test mode (for final inference only). Defaults to False.
    """

    _EDGE_WEIGHT_MODES = [
        "unit",
        "inverse_distance",
        "exp_neg_distance",
        "exp_neg_latent_distance",
        "affinity_latent_distance",
        "affinity_latent_distance_exp_neg",
    ]
    _SP_MERGE_MODES = [
        "order",
        "size",
    ]
    _CYLINDER_STRIDE_MODES = [
        "fixed_ratio",
        "max_stride",
    ]
    _SEMANTIC_LOSSES = [
        "lovasz",
        "bce",
    ]

    def __init__(
        self,
        in_channels: int,
        num_channels: int,
        voxel_size: float,
        min_spatial_shape: int,
        backbone: Config,
        test_cfg: Config,
        fg_semantic_class: int = 1,
        data_preprocessor: Optional[Config] = None,
        init_cfg: Optional[Config] = None,
        radius: float = 16,
        step_size_ratio: Optional[float] = 0.25,
        num_points: int = 640_000,
        sp_merge_mode: str = "size",
        chunk_size: int = 20_000,
        reg_lambda_fg: float = 0.003,
        reg_lambda_bg: float = 0.3,
        min_size: int = 1,
        graph_k: int = 8,
        separate_bg_fg: bool = True,
        spatial_weight: Optional[float] = None,
        edge_weight_mode: str = "unit",
        cylinder_stride_mode: str = "fixed_ratio",
        semantic_loss: str = "lovasz",
        max_object_size: Optional[float] = 4,
        max_iterations: Optional[int] = None,
        force_rewrite: bool = False,
        test_mode: bool = False,
        **kwargs,
    ):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg
        )
        self.unet = MODELS.build(backbone)
        self.voxel_size = voxel_size
        self.min_spatial_shape = min_spatial_shape
        self.fg_semantic_class = fg_semantic_class
        self.test_cfg = test_cfg
        self._init_layers(in_channels, num_channels)
        self.embed = (
            Seq()
            .append(MLP([num_channels, num_channels], bias=False))
            .append(torch.nn.Linear(num_channels, 5))
        )
        self.radius = radius
        self.step_size_ratio = step_size_ratio
        self.num_points = num_points
        self.sp_merge_mode = sp_merge_mode
        assert sp_merge_mode in self._SP_MERGE_MODES, (
            f"'sp_merge_mode' should be one of '{self._SP_MERGE_MODES}', got '{sp_merge_mode}' instead!"
        )
        self.chunk_size = chunk_size
        self.bg_fg_semantic_head = (
            Seq()
            .append(MLP([num_channels, num_channels], bias=False))
            .append(torch.nn.Linear(num_channels, 1))
        )
        self.reg_lambda_fg = reg_lambda_fg
        self.reg_lambda_bg = reg_lambda_bg
        self.min_size = min_size
        self.graph_k = graph_k
        self.separate_bg_fg = separate_bg_fg
        self.spatial_weight = spatial_weight
        self.edge_weight_mode = edge_weight_mode
        assert edge_weight_mode in self._EDGE_WEIGHT_MODES, (
            f"'edge_weight_mode' should be one of '{self._EDGE_WEIGHT_MODES}', got '{edge_weight_mode}' instead!"
        )
        self.cylinder_stride_mode = cylinder_stride_mode
        assert cylinder_stride_mode in self._CYLINDER_STRIDE_MODES, (
            f"'cylinder_stride_mode' should be one of '{self._CYLINDER_STRIDE_MODES}', got '{cylinder_stride_mode}' instead!"
        )
        self.max_object_size = max_object_size
        if self.cylinder_stride_mode == "fixed_ratio":
            assert self.step_size_ratio is not None, (
                "'step_size_ratio' should not be None when 'cylinder_stride_mode' is set to 'fixed_ratio'!"
            )
        elif self.cylinder_stride_mode == "max_stride":
            assert self.max_object_size is not None, (
                "'max_object_size' should not be None when 'cylinder_stride_mode' is set to 'max_stride'!"
            )
        self.semantic_loss = semantic_loss
        assert semantic_loss in self._SEMANTIC_LOSSES, (
            f"'semantic_loss' should be one of '{self._SEMANTIC_LOSSES}', got '{semantic_loss}' instead!"
        )
        if max_iterations is None:
            self.max_iterations = -1
        else:
            self.max_iterations = max_iterations
        self._force_rewrite = force_rewrite
        self._test_mode = test_mode

    @property
    def test_mode(self) -> bool:
        return self._test_mode

    @test_mode.setter
    def test_mode(self, test_mode: bool):
        self._test_mode = test_mode

    @property
    def force_rewrite(self) -> bool:
        return self._force_rewrite

    @force_rewrite.setter
    def force_rewrite(self, force_rewrite: bool):
        self._force_rewrite = force_rewrite

    def _init_layers(self, in_channels: int, num_channels: int):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key="subm1",
            )
        )
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True),
        )

    def extract_feat(self, x: SparseConvTensor) -> List[Tensor]:
        """Extract features from sparse tensor.

        Args:
            x (SparseConvTensor): Input sparse tensor of shape (n_points, in_channels).

        Returns:
            List[Tensor]: of len batch_size, each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        out = []
        for i in x.indices[:, 0].unique():
            out.append(x.features[x.indices[:, 0] == i])

        return out

    def _collate(
        self, points: List[Tensor], elastic_points: Optional[List[Tensor]] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            elastic_points (Optional[List[Tensor]]): Batch of elastic points to use for quantization. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: coordinates, features and inverse map of a sparse tensor + grid size (per spatial dimension)
        """
        if elastic_points is None:
            # Features are [<any extra features>, <relative xyz-coordinates>]. If the data only contains xyz-coordinates
            # the resulting feature tensor only contains the relative xyz-coordinates
            coordinates, features = ME.utils.batch_sparse_collate(
                [
                    (
                        (p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                        torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))),
                    )
                    for p in points
                ]
            )
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [
                    (
                        (el_p - el_p.min(0)[0]),
                        torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))),
                    )
                    for el_p, p in zip(elastic_points, points)
                ]
            )

        spatial_shape = torch.clip(
            coordinates.max(0)[0][1:] + 1, self.min_spatial_shape
        )
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DSegmentor."""
        pass

    def loss(
        self,
        batch_inputs_dict: Dict,
        batch_data_samples: List[Det3DDataSample],
        **kwargs,
    ) -> Dict:
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (Dict): The model input dict which include `points` key.
            batch_data_samples (List[Det3DDataSample]): The Data Samples. It includes information such as `gt_instances_3d` and `gt_sem_seg_3d`.

        Returns:
            Dict: A dictionary of loss components.
        """
        coordinates, features, inverse_mapping, spatial_shape = self._collate(
            batch_inputs_dict["points"], batch_inputs_dict.get("elastic_coords", None)
        )
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples)
        )
        x = self.extract_feat(x)
        # Extract feature embedings and background-foreground logits based on the features extracted by U-Net
        embed_logits = [self.embed(y) for y in x]
        bg_fg_semantic_logits = [self.bg_fg_semantic_head(y) for y in x]
        # Initialize cumulative losses
        total_discriminative_loss = 0
        total_semantic_loss_bg_fg = 0
        batch_size = len(batch_data_samples)

        for i in range(batch_size):
            # Instance labels (per-point)
            instance_labels_i = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            inverse_mapping_i = inverse_mapping[coordinates[:, 0][inverse_mapping] == i]
            _, inverse_mapping_i = torch.unique(inverse_mapping_i, return_inverse=True)
            # Boolean mask for foreground points
            fg_mask_i = batch_data_samples[i].gt_pts_seg.instance_mask
            device = instance_labels_i.device
            fg_mask_i = torch.from_numpy(fg_mask_i).to(device)
            # Filter out background points using instance_mask
            embed_logits_i = embed_logits[i][inverse_mapping_i]
            fg_embed_logits = embed_logits_i[fg_mask_i]
            # Compute instance discriminative loss for embedings
            fg_instance_labels_i = instance_labels_i[fg_mask_i]
            batch = torch.full_like(fg_instance_labels_i, i)
            discriminative_losses = discriminative_loss(
                fg_embed_logits,
                fg_instance_labels_i,
                batch,
                5,
            )
            # Accumulate discriminative loss for this sample
            for loss_name, loss in discriminative_losses.items():
                total_discriminative_loss += loss if loss_name == "ins_loss" else 0
            # Use the precomputed bi_semantic_logits for the current sample
            bg_fg_semantic_logits_i = bg_fg_semantic_logits[i][
                inverse_mapping_i
            ].squeeze(1)
            # Calculate semantic loss over all voxels (background and foreground)
            if self.semantic_loss == "lovasz":
                semantic_loss_bg_fg = lovasz_hinge_loss(
                    bg_fg_semantic_logits_i.unsqueeze(0),
                    fg_mask_i.to(torch.int64).unsqueeze(0),
                )
            elif self.semantic_loss == "bce":
                semantic_loss_bg_fg = F.binary_cross_entropy_with_logits(
                    bg_fg_semantic_logits_i, fg_mask_i.float()
                )
            else:
                raise ValueError(
                    f"'semantic_loss' should be one of '{self._SEMANTIC_LOSSES}', got '{self.semantic_loss} instead!"
                )
            # Accumulate semantic loss for the batch
            total_semantic_loss_bg_fg += semantic_loss_bg_fg

        # Average the accumulated losses over the batch
        total_discriminative_loss /= batch_size
        total_semantic_loss_bg_fg /= batch_size

        # Add to total loss
        loss_final = {
            "discriminative_loss": total_discriminative_loss,
            "semantic_loss_bg_fg": total_semantic_loss_bg_fg,
        }

        return loss_final

    @staticmethod
    def _compute_edge_distances(x: torch.Tensor, edge_ind: torch.Tensor) -> Tensor:
        """Compute distance between all nodes connected by an edge. Code from: https://github.com/drprojects/superpoint_transformer

        Args:
            x (torch.Tensor): node features
            edge_ind (torch.Tensor): source and target index for all edges

        Returns:
            Tensor: distances
        """
        dist = torch.norm(x[edge_ind[0]] - x[edge_ind[1]], dim=-1, p=2)

        return dist

    def _compute_edge_weights(
        self, points: Tensor, embed_logits: Tensor, edge_ind: Tensor
    ) -> Tensor:
        """Compute weights for edges. Code from: https://github.com/drprojects/superpoint_transformer

        Args:
            points (Tensor): point xyz-coordinates
            embed_logits (Tensor): point feature embedings
            edge_ind (Tensor): source and target index for all edges

        Raises:
            ValueError: if edge weight mode is unsupported

        Returns:
            Tensor: weights for all edges
        """
        # Begin by computing distances for all edges (if necessary)
        if self.edge_weight_mode == "unit":
            pass
        elif self.edge_weight_mode in ["inverse_distance", "exp_neg_distance"]:
            dist = self._compute_edge_distances(points, edge_ind)
            d_0 = dist.mean()
        else:
            latent_dist = self._compute_edge_distances(embed_logits, edge_ind)
            d_0 = latent_dist.mean()
        if self.edge_weight_mode == "unit":
            edge_weights = torch.ones_like(edge_ind[0])
        elif self.edge_weight_mode == "inverse_distance":
            edge_weights = 1 / (1 + dist / d_0)
        elif self.edge_weight_mode == "exp_neg_distance":
            edge_weights = torch.exp(-dist / d_0)
        elif self.edge_weight_mode == "exp_neg_latent_distance":
            edge_weights = torch.exp(-latent_dist / d_0)
        elif self.edge_weight_mode == "affinity_latent_distance":
            d_neg_exp = torch.exp(-latent_dist / d_0)
            edge_weights = d_neg_exp / (1 - d_neg_exp + self.epsilon_edge_weight)
        else:
            raise ValueError(
                f"'edge_weight_mode' should be one of '{self._EDGE_WEIGHT_MODES}', got '{self.edge_weight_mode} instead!"
            )

        return edge_weights

    def _generate_superpoints_bg_fg(
        self,
        points: Tensor,
        embed_logits: Tensor,
        bg_fg_pred: Optional[Tensor],
        separate_bg_fg: bool = True,
    ) -> Tensor:
        if separate_bg_fg:
            assert bg_fg_pred is not None, (
                "'bg_fg_pred' should not be None when 'separate_bg_fg' is set to True!"
            )
            fg_mask = bg_fg_pred == self.fg_semantic_class
            if (not torch.any(fg_mask)) or (not torch.any(~fg_mask)):
                # All points are either foreground ro background
                superpoint_labels = self._generate_superpoints(points, embed_logits)
            else:
                fg_superpoint_labels = self._generate_superpoints(
                    points[fg_mask], embed_logits[fg_mask]
                )
                fg_superpoint_max_label = fg_superpoint_labels.max()
                # Just use xyz-coordinates as features for background since feature embedings are essentially meaningless
                bg_superpoint_labels = self._generate_superpoints(
                    points[~fg_mask],
                    points[~fg_mask],
                    bg=True,
                )
                bg_superpoint_labels += fg_superpoint_max_label
                # Combine background and foreground superpoint labels into one tensor
                superpoint_labels = torch.zeros(
                    (points.shape[0],),
                    dtype=fg_superpoint_labels.dtype,
                    device=points.device,
                )
                superpoint_labels[fg_mask] = fg_superpoint_labels
                superpoint_labels[~fg_mask] = bg_superpoint_labels
        else:
            superpoint_labels = self._generate_superpoints(points, embed_logits)
        # Ensure that the superpoint labels are continuous
        _, superpoint_labels = torch.unique(superpoint_labels, return_inverse=True)

        return superpoint_labels

    def _generate_superpoints(
        self, points: Tensor, embed_logits: Tensor, bg: bool = False
    ) -> Tensor:
        edge_ind = knn_graph(points, k=self.graph_k, batch=None)
        edge_weight = self._compute_edge_weights(points, embed_logits, edge_ind)
        # Concatenate feature embedings with xyz-coordinates if spatial weight is not None or 0
        if (self.spatial_weight is None) or (self.spatial_weight == 0):
            x = embed_logits
        else:
            x = torch.cat((embed_logits, points * self.spatial_weight), dim=1)
        # Node sizes are set to 1 (one voxel each)
        s = torch.ones(x.shape[0], device=x.device)
        if bg:
            reg_lambda = self.reg_lambda_bg
        else:
            reg_lambda = self.reg_lambda_fg
        superpoint_labels, _, _ = merge_components_by_contour_prior(
            x,
            s,
            edge_ind,
            edge_weight,
            reg_lambda,
            self.min_size,
            max_iterations=self.max_iterations,
        )

        return superpoint_labels

    def predict(
        self,
        batch_inputs_dict: Dict,
        batch_data_samples: List[Det3DDataSample],
        **kwargs,
    ) -> List[Det3DDataSample]:
        lidar_path = batch_data_samples[0].lidar_path
        current_filename = os.path.splitext(os.path.basename(lidar_path))[0]
        logger = MMLogger.get_current_instance()
        # Directory where generated superpoints are saved
        results_path = os.path.join(
            self.test_cfg.superpoint_data_dir,
            f"{current_filename}.npy",
        )

        if self.test_mode:
            if os.path.isfile(results_path) and not self.force_rewrite:
                # Skip superpoint generation and return empty result
                print_log(
                    f"==> Superpoints have already been generated for '{current_filename}'! Skipping.",
                    logger=logger,
                )
                for data_sample in batch_data_samples:
                    empty_pd = PointData()
                    empty_pd.pts_superpoint_labels = torch.empty(0, dtype=torch.long)
                    data_sample.pred_pts_seg = empty_pd

                return batch_data_samples

            print_log(
                f"==> Generating superpoints for '{current_filename}'", logger=logger
            )
            original_points = batch_inputs_dict["points"][0]
            n_points = original_points.shape[0]
            if self.cylinder_stride_mode == "fixed_ratio":
                step_size = self.radius * self.step_size_ratio
                regions = generate_cylindrical_regions(original_points, step_size)
            elif self.cylinder_stride_mode == "max_stride":
                regions = generate_cylindrical_regions_max_stride(
                    original_points, self.radius, self.voxel_size, self.max_object_size
                )
            else:
                raise ValueError(
                    f"'cylinder_stride_mode' should be one of '{self._CYLINDER_STRIDE_MODES}', got '{self.cylinder_stride_mode} instead!"
                )
            max_superpoint_label = 0
            # List of tuples corresponding to valid superpoints (superpoint point indices, superpoint label)
            cylinder_superpoint_labels_list = []
            cylinder_point_indices_list = []

            last_originids = None
            for region in tqdm(regions, desc="==> Processing cylindrical regions"):
                region_mask = (
                    (original_points[:, 0] - region[0]) ** 2
                    + (original_points[:, 1] - region[1]) ** 2
                ) <= self.radius**2
                cylinder_points = original_points[region_mask]
                if len(cylinder_points) == 0:
                    continue
                cylinder_point_indices = torch.where(region_mask)[0]
                cylinder_point_indices_list.append(cylinder_point_indices)
                # Voxelize/downsample input points
                cylinder_points_vox, cylinder_point_vox_indices, vox_inverse_map = (
                    self._grid_sample(
                        cylinder_points,
                        cylinder_point_indices,
                        self.voxel_size,
                        self.num_points,
                        self.chunk_size,
                    )
                )
                last_originids = cylinder_point_vox_indices.cpu().numpy()
                # Transform voxelized data into sparse tensor and extract features
                coordinates, features, collate_inverse_map, spatial_shape = (
                    self._collate([cylinder_points_vox])
                )
                x = spconv.SparseConvTensor(
                    features, coordinates, spatial_shape, len(batch_data_samples)
                )
                x = self.extract_feat(x)
                embed_logits = self.embed(x[0])
                embed_logits = embed_logits[collate_inverse_map]
                bg_fg_semantic_logits = self.bg_fg_semantic_head(x[0])
                bg_fg_semantic_pred = (bg_fg_semantic_logits > 0).long().squeeze(1)
                bg_fg_semantic_pred = bg_fg_semantic_pred[collate_inverse_map]
                cylinder_superpoint_labels = self._generate_superpoints_bg_fg(
                    cylinder_points_vox,
                    embed_logits,
                    bg_fg_semantic_pred,
                    separate_bg_fg=self.separate_bg_fg,
                )
                # Only invalidate superpoints close to cylinder edges if there is more than one cylinder in total
                if len(regions) > 1:
                    # Similarly to the instance predictions, remove superpoints close to cylinder edges
                    # Distance to cylinder edge threshold (2 * voxel size from the edge)
                    edge_threshold_sq = (self.radius - 2 * self.voxel_size) ** 2
                    # Compute distances to cylinder center for all points
                    center_dists_sq = (cylinder_points_vox[:, 0] - region[0]) ** 2 + (
                        cylinder_points_vox[:, 1] - region[1]
                    ) ** 2
                    invalid_point_mask = center_dists_sq > edge_threshold_sq
                    invalid_labels = cylinder_superpoint_labels[invalid_point_mask]
                    if invalid_labels.numel() > 0:
                        invalid_labels = torch.unique(invalid_labels)
                        valid_superpoints = ~torch.isin(
                            cylinder_superpoint_labels, invalid_labels
                        )
                    else:
                        valid_superpoints = torch.ones_like(
                            cylinder_superpoint_labels, dtype=torch.bool
                        )
                    # Set label of invalid superpoints to -1
                    cylinder_superpoint_labels[~valid_superpoints] = -1
                else:
                    valid_superpoints = torch.ones_like(
                        cylinder_superpoint_labels,
                        dtype=torch.bool,
                        device=cylinder_superpoint_labels.device,
                    )
                # Cast to continuous range
                _, inverse = torch.unique(
                    cylinder_superpoint_labels[valid_superpoints], return_inverse=True
                )
                cylinder_superpoint_labels[valid_superpoints] = (
                    inverse + max_superpoint_label
                )
                cylinder_superpoint_labels = cylinder_superpoint_labels[vox_inverse_map]
                cylinder_superpoint_labels_list.append(cylinder_superpoint_labels)
                max_superpoint_label = int(cylinder_superpoint_labels.max().item()) + 1
                torch.cuda.empty_cache()

            # Merge cylinder-level superpoint labels to global superpoint labels
            final_superpoint_labels = self._merge_superpoints(
                cylinder_point_indices_list,
                cylinder_superpoint_labels_list,
                n_points,
            )
            # Save superpoint labels in .npy file
            self._save_superpoint_labels(
                results_path, final_superpoint_labels.cpu().numpy()
            )
            # Store results in return value for accuracy computations
            for _, data_sample in enumerate(batch_data_samples):
                data_sample.pred_pts_seg = PointData(
                    pts_superpoint_labels=[final_superpoint_labels.cpu().numpy()]
                )
                data_sample.pred_pts_seg["originids"] = last_originids
                data_sample.pred_pts_seg["filename"] = current_filename

            return batch_data_samples
        else:
            # Generate superpoints for validation (only generated for a single input cylinder)
            coordinates, features, inverse_mapping, spatial_shape = self._collate(
                batch_inputs_dict["points"]
            )
            x = spconv.SparseConvTensor(
                features, coordinates, spatial_shape, len(batch_data_samples)
            )
            x = self.extract_feat(x)
            embed_logits = self.embed(x[0])
            bg_fg_semantic_logits = self.bg_fg_semantic_head(x[0])
            bg_fg_semantic_pred = (bg_fg_semantic_logits > 0).long().squeeze(1)
            superpoint_labels = self._generate_superpoints_bg_fg(
                coordinates[coordinates[:, 0] == 0, 1:].float(),
                embed_logits,
                bg_fg_semantic_pred,
                separate_bg_fg=self.separate_bg_fg,
            )
            inverse_mapping_0 = inverse_mapping[coordinates[:, 0][inverse_mapping] == 0]
            superpoint_labels = superpoint_labels[inverse_mapping_0]
            for _, data_sample in enumerate(batch_data_samples):
                data_sample.pred_pts_seg = PointData(
                    pts_superpoint_labels=[superpoint_labels.cpu().numpy()]
                )

            return batch_data_samples

    def _merge_superpoints(
        self,
        cylinder_point_indices_list: List[Tensor],
        cylinder_superpoint_labels_list: List[Tensor],
        n_points: int,
    ) -> Tensor:
        """Merge superpoints from cylinders. Assumes that superpoint labels from each cylinder are globally unique.
        if sp_merge_mode is "order":
            * greedily assign each point the superpoint label that is processed first
        if sp_merge_mode is "size":
            * assign each point a superpoint label in a largest-superpoint-first order

        Args:
            cylinder_point_indices_list (List[Tensor]): list of global point cloud point indices for each cylinder
            cylinder_superpoint_labels_list (List[Tensor]): list of superpoint labels from each cylinder
            n_points (int): number of points in the full point cloud

        Raises:
            ValueError: if unsupported value for self.sp_merge_mode

        Returns:
            Tensor: merged superpoint labels mapped to a continuous range
        """
        if self.sp_merge_mode == "order":
            final_superpoint_labels = self._merge_superpoints_order_first(
                cylinder_point_indices_list, cylinder_superpoint_labels_list, n_points
            )
        elif self.sp_merge_mode == "size":
            final_superpoint_labels = self._merge_superpoints_largest_first(
                cylinder_point_indices_list, cylinder_superpoint_labels_list, n_points
            )
        else:
            raise ValueError(
                f"'sp_merge_mode' should be one of '{self._SP_MERGE_MODES}', got '{self.sp_merge_mode} instead!"
            )
        # Map superpoint labels to a continuous range (this assumes that there should be no invalid (-1) superpoints left after merging)
        _, final_superpoint_labels = torch.unique(
            final_superpoint_labels, return_inverse=True
        )

        return final_superpoint_labels

    @staticmethod
    def _grid_sample(
        points: torch.Tensor,
        indices: torch.Tensor,
        voxel_size: float,
        num_points: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Voxel-downsample point cloud by averaging points in each voxel.

        Args:
            points (Tensor): (N, 3) or (N, F) point coordinates/features.
            indices (Tensor): (N,) original point indices (int32/int64).
            voxel_size (float): size of each voxel in meters.
            num_points (Optional[int]): If set and voxelized point count > num_points, randomly subsample to num_points. Defaults to None.
            chunk_size (Optional[int]): Chunk size to control memory usage when computing nearest voxel for voxels dropped during downsampling. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                * (M, F) voxelized (and possibly downsampled) point cloud.
                * (M,) one representative original index per voxel.
                * (N,) for each original point, index into the voxelized/downsampled points.
        """
        voxels = torch.floor(points[:, :3] / voxel_size).to(torch.int32)
        unique_voxels, inverse = torch.unique(voxels, return_inverse=True, dim=0)
        M = unique_voxels.size(0)
        # Compute voxel-level mean
        ones = torch.ones_like(inverse, dtype=points.dtype)
        sum_xyz = torch.zeros((M, points.size(1)), device=points.device, dtype=points.dtype)
        count_xyz = torch.zeros(M, device=points.device, dtype=points.dtype)
        sum_xyz.index_add_(0, inverse, points)
        count_xyz.index_add_(0, inverse, ones)
        voxel_points = sum_xyz / count_xyz.unsqueeze(1)
        # Pick one representative original index per voxel
        voxel_indices = torch.full((M,), -1, device=indices.device, dtype=indices.dtype)
        voxel_indices.index_copy_(0, inverse, indices)
        # Randomly downsample voxels, if necessary
        if num_points is not None and M > num_points:
            # Randomly downsample voxels
            choices = np.random.choice(M, num_points, replace=False)
            choices = torch.as_tensor(choices, device=points.device)
            kept_voxels = unique_voxels[choices]
            voxel_points = voxel_points[choices]
            voxel_indices = voxel_indices[choices]
            # Build map from all voxels to the nearest kept voxel
            nn_kept_ind = torch.empty(M, dtype=torch.long, device=points.device)
            # Process in chunks to keep memory usage reasonable
            for start in range(0, M, chunk_size):
                end = min(start + chunk_size, M)
                dists = torch.cdist(unique_voxels[start:end].float(), kept_voxels.float())
                nn_kept_ind[start:end] = torch.argmin(dists, dim=1)
            # Map each original point to its nearest kept voxel
            inverse_map = nn_kept_ind[inverse]
        else:
            # No downsampling required
            inverse_map = inverse

        return voxel_points, voxel_indices, inverse_map

    @staticmethod
    def _merge_superpoints_order_first(
        cylinder_point_indices_list: List[Tensor],
        cylinder_superpoint_labels_list: List[Tensor],
        n_points: int,
    ) -> Tensor:
        final_superpoint_labels = torch.full(
            (n_points,),
            -1,
            dtype=cylinder_superpoint_labels_list[0].dtype,
            device=cylinder_superpoint_labels_list[0].device,
        )
        # Iterate over cylinders and assign superpoint labels to points that do not yet have one (greedy approach, first superpoint label always wins)
        for point_inds, superpoint_labels in zip(
            cylinder_point_indices_list,
            cylinder_superpoint_labels_list,
        ):
            valid_mask = superpoint_labels >= 0
            point_inds = point_inds[valid_mask]
            superpoint_labels = superpoint_labels[valid_mask]
            unassigned_mask = final_superpoint_labels[point_inds] == -1
            final_superpoint_labels[point_inds[unassigned_mask]] = superpoint_labels[
                unassigned_mask
            ]

        return final_superpoint_labels

    @staticmethod
    def _merge_superpoints_largest_first(
        cylinder_point_indices_list: List[Tensor],
        cylinder_superpoint_labels_list: List[Tensor],
        n_points: int,
    ) -> Tensor:
        global_point_inds = torch.cat(cylinder_point_indices_list)
        global_superpoint_labels = torch.cat(cylinder_superpoint_labels_list)
        valid_mask = global_superpoint_labels >= 0
        global_point_inds = global_point_inds[valid_mask]
        global_superpoint_labels = global_superpoint_labels[valid_mask]

        if global_superpoint_labels.numel() == 0:
            return torch.full(
                (n_points,),
                -1,
                device=global_superpoint_labels.device,
                dtype=global_superpoint_labels.device,
            )
        # Sort points by superpoint label
        sort_order = torch.argsort(global_superpoint_labels)
        global_superpoint_labels = global_superpoint_labels[sort_order]
        global_point_inds = global_point_inds[sort_order]
        # Find unique superpoints (since labels are sorted, unique_consecutive gives unique superpoint labels)
        unq_superpoint_labels, counts = torch.unique_consecutive(
            global_superpoint_labels, return_counts=True
        )
        # Compute superpoint start indices
        segment_starts = torch.cumsum(
            torch.cat(
                [torch.tensor([0], device=global_superpoint_labels.device), counts[:-1]]
            ),
            dim=0,
        )
        # Sort superpoints by size
        size_order = torch.argsort(counts, descending=True)
        # Assign superpoints with largest-first ordering
        final_superpoint_labels = torch.full(
            (n_points,),
            -1,
            device=global_superpoint_labels.device,
            dtype=global_superpoint_labels.device,
        )
        for i in size_order:
            start = segment_starts[i]
            count = counts[i]
            sp_point_inds = global_point_inds[start : start + count]
            unassigned_mask = final_superpoint_labels[sp_point_inds] == -1
            if unassigned_mask.any():
                final_superpoint_labels[sp_point_inds[unassigned_mask]] = (
                    unq_superpoint_labels[i]
                )

        return final_superpoint_labels

    @staticmethod
    def _save_superpoint_labels(save_path: str, superpoint_labels: np.ndarray) -> None:
        np.save(save_path, superpoint_labels)
