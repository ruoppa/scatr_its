import os
import torch

import numpy as np
import spconv.pytorch as spconv
import MinkowskiEngine as ME
import torch.nn.functional as F

from tqdm import tqdm
from torch import Tensor
from torch.cuda.amp import autocast
from spconv.pytorch import SparseConvTensor
from typing import List, Optional, Dict, Tuple, Union
from scatr_its.tools import (
    Seq,
    MLP,
    mask_matrix_nms,
    generate_cylindrical_regions,
    generate_cylindrical_regions_max_stride,
)
from scatr_its.losses import discriminative_loss, lovasz_hinge_loss
from util import write_pc
from torch_cluster import fps
from torch_scatter import scatter_mean, scatter_add, scatter_max
from mmengine.config import Config
from mmdet3d.structures import PointData, Det3DDataSample
from mmengine.logging import MMLogger, print_log
from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS


@MODELS.register_module()
class SCATR_ITS(Base3DDetector):
    """SCATR-ITS model for individual tree segmentation from LiDAR point clouds.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): Number of output channels.
        voxel_size (float): Voxel size.
        num_semantic_classes (int): Number of classes.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        backbone (Optional[Config]): Config dict of the backbone.
        decoder (Optional[Config]): Config dict of the decoder.
        criterion (Config): Config dict of the criterion.
        train_cfg (Configl): Config dict of training hyper-parameters.
        test_cfg (Config): Config dict of test hyper-parameters.
        fg_semantic_class (int): Semantic class for foreground points and class label for all instances. Defaults to 1.
        bg_semantic_class (int): Semantic class of points considered background (e.g. "ground" or "other"). Defaults to 0.
        query_point_num (int): Number of query points. Defaults to 200.
        data_preprocessor (Optional[Config]): The pre-process config of :class:`BaseDataPreprocessor`.
        init_cfg (Optional[Config]): The config to control the initialization. Defaults to None.
        prepare_epochs (Optional[int]): Number of pretraining epochs (only train backbone network). Defaults to None.
        radius (float): Radius of cylinders sampled during trainingin (in meters). Defaults to 16.
        step_size_ratio (float): Step size for inference is radius * step_size_ratio. Defaults to 0.25.
        num_points (int): Maximum number of points after voxelization. Downsampled if above this. Defaults to 640000.
        score_th (float): Score threshold for instances. Defaults to 0.4.
        overlap_th (float): Overlap percentage threshold for removing overlapping instances. Defaults to 0.3.
        chunk_size (int): Chunk size. Defaults to 20_000.
        cylinder_stride_mode (str): Mode for cylinder stride during inference. See _CYLINDER_STRIDE_MODE for supported modes. Defaults to "fixed_ratio".
        max_object_size (Optional[float]): Maximum object width. Only used if cylinder_stride mode is set to "max_stride". Defaults to 4.
        test_mode (bool): If True, run predicting in test mode (for final inference only). Defaults to False.
    """

    _CYLINDER_STRIDE_MODES = [
        "fixed_ratio",
        "max_stride",
    ]
    _SEMANTIC_LOSSES = [
        "lovasz",
        "bce",
    ]
    # Max integer value for int 32 (for processing large tensors in chunks to avoid overflow errors)
    _MAX_INT = 1_800_000_000

    def __init__(
        self,
        in_channels: int,
        num_channels: int,
        voxel_size: float,
        num_semantic_classes: int,
        min_spatial_shape: int,
        backbone: Config,
        decoder: Config,
        criterion: Config,
        train_cfg: Config,
        test_cfg: Config,
        fg_semantic_class: int = 1,
        bg_semantic_class: int = 0,
        bg_instance_class: int = 0,
        query_point_num: int = 300,
        data_preprocessor: Optional[Config] = None,
        init_cfg: Optional[Config] = None,
        prepare_epochs: Optional[int] = None,
        radius: float = 16,
        step_size_ratio: float = 0.25,
        num_points: int = 640_000,
        score_th: float = 0.3,
        overlap_th: float = 0.3,
        chunk_size: int = 20_000,
        cylinder_stride_mode: str = "fixed_ratio",
        semantic_loss: str = "bce",
        max_object_size: Optional[float] = 6,
        sp_processing: bool = True,
        sp_size_aware: bool = True,
        dynamic_query_count: bool = True,
        dynamic_query_module: Optional[Config] = None,
        test_mode: bool = False,
        **kwargs,
    ):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg
        )
        self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_semantic_classes = num_semantic_classes
        self.min_spatial_shape = min_spatial_shape
        self.fg_semantic_class = fg_semantic_class
        self.bg_semantic_class = bg_semantic_class
        self.bg_instance_class = bg_instance_class
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if prepare_epochs is not None:
            self.prepare_epochs = prepare_epochs
        else:
            # No pretraining epochs
            self.prepare_epochs = 0
        self._init_layers(in_channels, num_channels)
        self.embed = (
            Seq()
            .append(MLP([num_channels, num_channels], bias=False))
            .append(torch.nn.Linear(num_channels, 5))
        )
        self.query_point_num = query_point_num
        self.radius = radius
        self.step_size_ratio = step_size_ratio
        self.num_points = num_points
        self.score_th = score_th
        self.overlap_th = overlap_th
        self.chunk_size = chunk_size
        self.bg_fg_semantic_head = (
            Seq()
            .append(MLP([num_channels, num_channels], bias=False))
            .append(torch.nn.Linear(num_channels, 1))
        )
        self.sp_processing = sp_processing
        self.sp_size_aware = sp_size_aware
        self.dynamic_query_count = dynamic_query_count
        if dynamic_query_count:
            assert dynamic_query_module is not None, (
                "'dynamic_query_module' should not be 'None' when 'dynamic_query_count' is set to True!"
            )
            self.dynamic_query_head = MODELS.build(dynamic_query_module)
        self.cylinder_stride_mode = cylinder_stride_mode
        assert cylinder_stride_mode in self._CYLINDER_STRIDE_MODES, (
            f"'cylinder_stride_mode' should be one of '{self._CYLINDER_STRIDE_MODES}', got '{cylinder_stride_mode}' instead!"
        )
        self.semantic_loss = semantic_loss
        assert semantic_loss in self._SEMANTIC_LOSSES, (
            f"'semantic_loss' should be one of '{self._SEMANTIC_LOSSES}', got '{semantic_loss}' instead!"
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
        self._test_mode = test_mode
        # Generator object for randomness
        self._rng = torch.Generator(device="cuda").manual_seed(33)

    @property
    def test_mode(self) -> bool:
        return self._test_mode

    @test_mode.setter
    def test_mode(self, test_mode: bool):
        self._test_mode = test_mode

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
        """Implement abstract method of Base3DDetector."""
        pass

    def _get_isa_guided_queries(
        self,
        x_sp: Tensor,
        bg_fg_semantic_logits_sp: Tensor,
        embed_logits_sp: Tensor,
        instance_labels_sp: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        # Extract bg/fg predictions from logits
        bg_fg_predictions = (bg_fg_semantic_logits_sp > 0).long().squeeze(1)
        # Extract indices of voxels that were predicted to be part of the foreground (i.e. an instance)
        # -> these points are considered tree points
        tree_indices = torch.where(bg_fg_predictions == self.fg_semantic_class)[0]
        if tree_indices.numel() <= 0:
            return None, None, None 
        # FPS from all tree points
        tree_point_embeddings = embed_logits_sp[tree_indices]
        n_tree_points = tree_point_embeddings.size(0)
        batch_indices = torch.zeros(
            n_tree_points, dtype=torch.long, device=embed_logits_sp.device
        )
        with autocast(enabled=False):
            if self.dynamic_query_count:
                density_logits = self.dynamic_query_head(x_sp)
                pred_query_point_num = self.dynamic_query_head.get_query_counts(
                    density_logits
                ).item()
                ratio = min(pred_query_point_num / n_tree_points, 1.0)
            else:
                ratio = min(self.query_point_num / n_tree_points, 1.0)
            if self.test_mode:
                # Quantize for more deterministic results in test mode
                tree_point_embeddings = torch.round(tree_point_embeddings * 1e4) / 1e4
            topk_indices = fps(
                tree_point_embeddings.float(),
                batch_indices,
                ratio=ratio,
                random_start=False,
            )
        query_point_indices = tree_indices[topk_indices]
        query_point_feats = x_sp[query_point_indices]
        query_instance_labels = None
        if instance_labels_sp is not None:
            query_instance_labels = instance_labels_sp[query_point_indices]

        return (
            query_point_feats,
            query_point_indices,
            query_instance_labels,
        )

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
        coordinates, features, collate_inverse_mapping, spatial_shape = self._collate(
            batch_inputs_dict["points"]
        )
        batch_size = len(batch_data_samples)
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples)
        )

        x = self.extract_feat(x)
        # Extract feature embedings and background-foreground logits based on the features extracted by U-Net
        embed_logits = [self.embed(y) for y in x]
        bg_fg_semantic_logits = [self.bg_fg_semantic_head(y) for y in x]
        # Fetch inverse maps that cast features back to their pre-collate dimensionality (number of points)
        inverse_mappings = self._get_inverse_mappings(
            batch_size, collate_inverse_mapping, coordinates
        )
        # Construct superpoint-level representations of features (if necessary)
        if kwargs["epoch"] > self.prepare_epochs or self.dynamic_query_count:
            superpoint_labels = []
            instance_labels = []
            for i in range(batch_size):
                superpoint_labels.append(
                    batch_data_samples[i].gt_pts_seg.pts_superpoint_labels
                )
                instance_labels.append(
                    batch_data_samples[i].gt_pts_seg.pts_instance_mask
                )
            (
                x_sp,
                bg_fg_semantic_logits_sp,
                embed_logits_sp,
                _,
                instance_labels_sp,
            ) = self._superpoint_mean_pooling(
                superpoint_labels,
                inverse_mappings,
                x,
                bg_fg_semantic_logits,
                embed_logits,
                None,
                instance_labels,
            )

        # Initialize cumulative losses
        total_discriminative_loss = 0
        total_semantic_loss_bg_fg = 0
        if self.dynamic_query_count:
            total_density_loss = 0

        for i in range(batch_size):
            # Instance labels (per-point)
            instance_labels_i = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            inverse_mapping_i = inverse_mappings[i]
            # Boolean mask for foreground points
            fg_mask_i = batch_data_samples[i].gt_pts_seg.instance_mask
            device = instance_labels_i.device
            fg_mask_i = torch.from_numpy(fg_mask_i).to(device)
            # Filter out background points using instance_mask
            embed_logits_i = embed_logits[i][inverse_mapping_i]
            fg_embed_logits_i = embed_logits_i[fg_mask_i]
            # Compute instance discriminative loss for embedings
            fg_instance_labels_i = instance_labels_i[fg_mask_i]
            batch = torch.full_like(fg_instance_labels_i, i)
            discriminative_losses = discriminative_loss(
                fg_embed_logits_i,
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
                    f"'semantic_loss' should be one of '{self._SEMANTIC_LOSSES}', got '{self.semantic_loss}' instead!"
                )
            # Accumulate semantic loss for the batch
            total_semantic_loss_bg_fg += semantic_loss_bg_fg
            # Calculate dynamic query count loss
            if self.dynamic_query_count:
                density_logits = self.dynamic_query_head(x_sp[i])
                density_targets = self.dynamic_query_head.get_targets(
                    instance_labels_sp[i]
                )
                density_loss = self.dynamic_query_head.get_loss(
                    density_logits, density_targets
                )
                total_density_loss += density_loss

        # Average the accumulated losses over the batch
        total_discriminative_loss /= batch_size
        total_semantic_loss_bg_fg /= batch_size
        # Add to total loss
        loss_final = {
            "discriminative_loss": total_discriminative_loss,
            "semantic_loss_bg_fg": total_semantic_loss_bg_fg,
        }
        if self.dynamic_query_count:
            total_density_loss /= batch_size
            loss_final["density_loss"] = total_density_loss

        query_point_feats = []
        query_point_indices = []
        query_instance_labels = []
        # Generate instance queries if not pretraining
        if kwargs["epoch"] > self.prepare_epochs:
            for i in range(batch_size):
                with torch.no_grad():
                    fg_mask_i = batch_data_samples[i].gt_pts_seg.instance_mask
                    superpoint_labels_i = superpoint_labels[i]
                    valid_superpoint_labels_i = torch.unique(
                        superpoint_labels_i[fg_mask_i]
                    )
                    if valid_superpoint_labels_i.numel() < 10:
                        # Too few valid superpoints, skip query generation
                        query_point_feats.append([])
                        query_point_indices.append([])
                        query_instance_labels.append([])
                        continue
                    else:
                        (
                            query_point_feats_i,
                            query_point_indices_i,
                            query_instance_labels_i,
                        ) = self._get_isa_guided_queries(
                            x_sp[i],
                            bg_fg_semantic_logits_sp[i],
                            embed_logits_sp[i],
                            instance_labels_sp[i],
                        )
                        if query_point_feats_i is not None:
                            query_point_feats.append(query_point_feats_i)
                            query_point_indices.append(query_point_indices_i)
                            query_instance_labels.append(query_instance_labels_i)
                        else:
                            # No queries due to no predicted tree points
                            query_point_feats.append([])
                            query_point_indices.append([])
                            query_instance_labels.append([])

            if all(len(q) == 0 for q in query_point_feats):
                # No instance query points for any scene in batch -> no need to compute instance losses
                return loss_final
            if any(len(q) == 0 for q in query_point_feats):
                # Filter out empty queries and store original batch indices
                filtered_results = []
                for i in range(len(query_point_feats)):
                    if len(query_point_feats[i]) > 0:
                        filtered_results.append(
                            (
                                batch_data_samples[i],
                                x_sp[i],
                                xyz_sp[i],
                                query_point_feats[i],
                                query_point_indices[i],
                                query_instance_labels[i],
                                bg_fg_semantic_logits_sp[i],
                            )
                        )
                # Unpack filtered results into separate lists
                (
                    batch_data_samples,
                    x_sp,
                    xyz_sp,
                    query_point_feats,
                    query_point_indices,
                    query_instance_labels,
                    bg_fg_semantic_logits_sp,
                ) = zip(*filtered_results)
                # Convert the zipped result back to list format
                batch_data_samples = list(batch_data_samples)
                x_sp = list(x_sp)
                xyz_sp = list(xyz_sp)
                query_point_feats = list(query_point_feats)
                query_point_indices = list(query_point_indices)
                query_point_xyz = list(query_point_xyz)
                query_instance_labels = list(query_instance_labels)
                bg_fg_semantic_logits_sp = list(bg_fg_semantic_logits_sp)

            # Decode queries
            x = self.decoder(
                x_sp,
                query_point_feats,
            )
            # Compute full semantic and instance loss for predictions
            sp_gt_instances = []
            for i in range(len(batch_data_samples)):
                superpoint_labels_i = batch_data_samples[
                    i
                ].gt_pts_seg.pts_superpoint_labels
                instance_labels_i = batch_data_samples[i].gt_pts_seg.pts_instance_mask
                semantic_labels_i = batch_data_samples[
                    i
                ].gt_pts_seg.pts_semantic_mask
                # Generate superpoint level gt semantic masks and store them
                batch_data_samples[
                    i
                ].gt_instances_3d.sp_sem_masks = self._get_gt_semantic_masks(
                    semantic_labels_i,
                    superpoint_labels_i,
                    self.num_semantic_classes,
                )
                # Generate superpoint level gt instance masks and store them
                batch_data_samples[
                    i
                ].gt_instances_3d.sp_inst_masks = self._get_gt_instance_masks(
                    instance_labels_i, superpoint_labels_i
                )
                # Also store superpoint sizes
                superpoint_sizes = torch.from_numpy(
                    batch_data_samples[i].gt_pts_seg.superpoint_sizes
                ).float()
                superpoint_sizes = superpoint_sizes.to(
                    batch_data_samples[i].gt_instances_3d.labels_3d.device
                )
                batch_data_samples[
                    i
                ].gt_instances_3d.superpoint_sizes = superpoint_sizes
                (
                    batch_data_samples[i].gt_instances_3d.labels_3d,
                    batch_data_samples[i].gt_instances_3d.sp_inst_masks,
                    batch_data_samples[i].gt_instances_3d.ratio_inspoint,
                ) = self._filter_bg_masks(
                    batch_data_samples[i].gt_instances_3d,
                    [self.bg_instance_class],
                    batch_data_samples[i].gt_pts_seg.ratio_inspoint,
                )
                batch_data_samples[
                    i
                ].gt_instances_3d.query_inslabel = query_instance_labels[i]
                sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)
            # Compute instance losses
            inst_loss_epoch = kwargs["epoch"] - self.prepare_epochs
            loss = self.criterion(x, sp_gt_instances, inst_loss_epoch)
            loss_final.update(loss)

        return loss_final

    def predict(
        self,
        batch_inputs_dict: Dict,
        batch_data_samples: List[Det3DDataSample],
        **kwargs,
    ) -> List[Det3DDataSample]:
        lidar_path = batch_data_samples[0].lidar_path
        offsets = np.array(batch_data_samples[0].offsets)
        scales = np.array(batch_data_samples[0].scales)
        current_filename = os.path.splitext(os.path.basename(lidar_path))[0]
        logger = MMLogger.get_current_instance()

        if self.test_mode:
            print_log(
                f"==> Performing inference for '{current_filename}'", logger=logger
            )
            pts_semantic_gt = batch_data_samples[0].eval_ann_info["pts_semantic_mask"]
            pts_instance_gt = batch_data_samples[0].eval_ann_info["pts_instance_mask"]
            pts_superpoint_labels = batch_data_samples[0].eval_ann_info[
                "pts_superpoint_labels"
            ]
            original_points = batch_inputs_dict["points"][0]
            # Move superpoint labels to gpu
            pts_superpoint_labels = torch.from_numpy(pts_superpoint_labels)
            pts_superpoint_labels = pts_superpoint_labels.to(original_points.device)
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
            if self.sp_processing:
                final_semantic_labels, final_instance_labels, final_instance_scores = (
                    self._predict_sp(
                        original_points,
                        pts_superpoint_labels,
                        regions,
                        len(batch_data_samples),
                    )
                )
            else:
                final_semantic_labels, final_instance_labels, final_instance_scores = (
                    self._predict_full(
                        original_points,
                        pts_superpoint_labels,
                        regions,
                        len(batch_data_samples),
                    )
                )
            # Save the final combined results
            results_path = os.path.join(
                self.test_cfg.output_data_dir,
                f"{current_filename}.{self.test_cfg.file_extension}",
            )
            write_pc(
                results_path,
                self.test_cfg.file_extension,
                original_points.cpu().numpy(),
                final_semantic_labels,
                self.test_cfg.semantic_pred_field,
                final_instance_labels,
                self.test_cfg.instance_pred_field,
                final_instance_scores,
                self.test_cfg.score_pred_field,
                self.test_cfg.extra_features,
                offsets,
                scales,
                pts_semantic_gt,
                self.test_cfg.semantic_gt_field,
                pts_instance_gt,
                self.test_cfg.instance_gt_field,
            )

            # Return an empty pointdata object that only contains the name of the prediction file.
            # The accuracy metrics are then computed using the full file.
            for data_sample in batch_data_samples:
                empty_pd = PointData()
                empty_pd.pts_instance_mask = torch.empty(0, dtype=torch.long)
                empty_pd.scores = torch.empty(0, dtype=torch.float32)
                data_sample.pred_pts_seg = empty_pd
                data_sample.pred_pts_seg["filename"] = current_filename

            return batch_data_samples
        else:
            coordinates, features, collate_inverse_mapping, spatial_shape = (
                self._collate(batch_inputs_dict["points"])
            )
            batch_size = len(batch_data_samples)
            x = spconv.SparseConvTensor(
                features, coordinates, spatial_shape, batch_size
            )
            x = self.extract_feat(x)
            inverse_mappings = self._get_inverse_mappings(
                batch_size, collate_inverse_mapping, coordinates
            )
            result_data = []
            for i in range(batch_size):
                embed_logits = self.embed(x[i])
                bg_fg_semantic_logits = self.bg_fg_semantic_head(x[i])
                # Ensure superpoint labels are continuous
                superpoint_labels = batch_data_samples[i].eval_ann_info[
                    "pts_superpoint_labels"
                ]
                superpoint_labels = torch.from_numpy(superpoint_labels)
                superpoint_labels = superpoint_labels.to(embed_logits.device)
                _, superpoint_labels = torch.unique(
                    superpoint_labels,
                    return_inverse=True,
                    sorted=True,
                )
                # Superpoint pooling
                (
                    x_sp,
                    bg_fg_semantic_logits_sp,
                    embed_logits_sp,
                    _,
                    _,
                ) = self._superpoint_mean_pooling(
                    superpoint_labels=superpoint_labels,
                    inverse_mappings=inverse_mappings[i],
                    x=x[i],
                    bg_fg_semantic_logits=bg_fg_semantic_logits,
                    embed_logits=embed_logits,
                    xyz=None,
                    instance_labels=None,
                )
                # ISA-guided queries
                query_point_feats, _, _ = self._get_isa_guided_queries(
                    x_sp[0],
                    bg_fg_semantic_logits_sp[0],
                    embed_logits_sp[0],
                )
                if query_point_feats is None:
                    # No queries -> no predicted instances
                    semantic_pred = (
                        (bg_fg_semantic_logits[inverse_mappings[i]] > 0)
                        .long()
                        .squeeze(1)
                    )
                    instance_pred = torch.zeros_like(semantic_pred).unsqueeze(0)
                    result_data_i = PointData(
                        pts_semantic_mask=[semantic_pred.cpu().numpy()],
                        pts_instance_mask=[instance_pred.cpu().numpy()],
                        instance_labels=np.array([1]),
                        instance_scores=np.array([0.0]),
                    )
                else:
                    query_point_feats = [query_point_feats]
                    x = self.decoder(x_sp, query_point_feats)
                    if self.sp_processing:
                        # NOTE: sp_size_aware is an inference-only option
                        result_data_i = self._collect_pred_masks(decoder_out=x)[0]
                        # Cast back to full size from superpoints
                        superpoint_labels_numpy = superpoint_labels.cpu().numpy()
                        result_data_i.pts_semantic_mask[0] = (
                            result_data_i.pts_semantic_mask[0][superpoint_labels_numpy]
                        )
                        result_data_i.pts_instance_mask[0] = (
                            result_data_i.pts_instance_mask[0][
                                :, superpoint_labels_numpy
                            ]
                        )
                    else:
                        x["masks"][0] = x["masks"][0][:, superpoint_labels]
                        result_data_i = self._collect_pred_masks(decoder_out=x)[0]
                result_data.append(result_data_i)

            for i, data_sample in enumerate(batch_data_samples):
                data_sample.pred_pts_seg = result_data[i]

            return batch_data_samples

    def _predict_full(
        self,
        original_points: Tensor,
        pts_superpoint_labels: Tensor,
        regions: List[Tuple[float, float]],
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        multi_cylinder = len(regions) > 1
        votes_counter = torch.zeros(
            (original_points.shape[0], self.num_semantic_classes),
            device=original_points.device,
            dtype=torch.int16,
        )
        all_preliminary_instances = torch.full(
            (original_points.shape[0],),
            -1,
            device=original_points.device,
        )
        max_instance_id = 0
        global_instance_scores = torch.zeros(
            (original_points.shape[0],),
            device=original_points.device,
            dtype=torch.float32,
        )
        # List of tuples corresponding to best instances (instance point indices, instance id, instance score)
        best_instances = []
        for region in tqdm(regions, desc="==> Processing cylindrical regions"):
            region_mask = (
                (original_points[:, 0] - region[0]) ** 2
                + (original_points[:, 1] - region[1]) ** 2
            ) <= self.radius**2
            cylinder_points = original_points[region_mask].clone()
            cylinder_point_indices = torch.where(region_mask)[0]
            cylinder_superpoint_labels = pts_superpoint_labels[region_mask]
            # Make labels continuous
            _, cylinder_superpoint_labels = torch.unique(
                cylinder_superpoint_labels,
                return_inverse=True,
                sorted=True,
            )
            if len(cylinder_points) == 0:
                continue
            # Voxelize/downsample input points
            rng = self._get_rng(cylinder_points.device)
            (
                cylinder_points_vox,
                cylinder_superpoint_labels_vox,
                _,
                vox_inverse_map,
            ) = self._grid_sample(
                points=cylinder_points,
                superpoint_labels=cylinder_superpoint_labels,
                indices=cylinder_point_indices,
                voxel_size=self.voxel_size,
                rng=rng,
                max_int=self._MAX_INT,
                num_points=self.num_points,
                return_inverse=True,
            )
            # Transform voxelized data into sparse tensor and extract features
            coordinates, features, collate_inverse_map, spatial_shape = self._collate(
                [cylinder_points_vox]
            )
            x = spconv.SparseConvTensor(features, coordinates, spatial_shape, n_samples)
            x = self.extract_feat(x)
            embed_logits = self.embed(x[0])
            bg_fg_semantic_logits = self.bg_fg_semantic_head(x[0])
            # Superpoint pooling
            _, cylinder_superpoint_labels_vox = torch.unique(
                cylinder_superpoint_labels_vox, return_inverse=True, sorted=True
            )
            (
                x_sp,
                bg_fg_semantic_logits_sp,
                embed_logits_sp,
                _,
                _,
            ) = self._superpoint_mean_pooling(
                superpoint_labels=cylinder_superpoint_labels_vox,
                inverse_mappings=collate_inverse_map,
                x=x,
                bg_fg_semantic_logits=bg_fg_semantic_logits,
                embed_logits=embed_logits,
                xyz=None,
                instance_labels=None,
            )
            query_point_feats, _, _  = self._get_isa_guided_queries(
                x_sp[0],
                bg_fg_semantic_logits_sp[0],
                embed_logits_sp[0],
            )
            if query_point_feats is not None:
                x = self.decoder(
                    feats=x_sp,
                    queries=[query_point_feats],
                )
                # Cast predictions back to voxel-level represnetation
                x["masks"][0] = x["masks"][0][:, cylinder_superpoint_labels_vox]
                result_data = self._collect_pred_masks(
                    decoder_out=x,
                    test_mode=True,
                    coordinates=cylinder_points_vox[:, :3],
                )
                # Collect masks and their scores
                instance_mask_preds = result_data["pts_instance_mask"]
                instance_score_preds = result_data["instance_scores"]
                valid_scores = instance_score_preds > self.score_th
                # Only invalidate superpoints close to cylinder edges if there is more than one cylinder in total
                if multi_cylinder:
                    # Distance to cylinder edge threshold (2 * voxel size from the edge)
                    edge_threshold_sq = (self.radius - 2 * self.voxel_size) ** 2
                    # Compute distances to cylinder center for all points
                    center_dists_sq = (cylinder_points_vox[:, 0] - region[0]) ** 2 + (
                        cylinder_points_vox[:, 1] - region[1]
                    ) ** 2
                    center_dists_sq = center_dists_sq.unsqueeze(0)
                    edge_th_violation_mask = (
                        center_dists_sq > edge_threshold_sq
                    ) & instance_mask_preds
                    edge_th_violation_mask = edge_th_violation_mask.any(dim=1)
                    valid_scores_2 = ~edge_th_violation_mask
                else:
                    valid_scores_2 = torch.ones_like(
                        valid_scores,
                        dtype=torch.bool,
                        device=valid_scores.device,
                    )
                # Binary mask for instances that are retained
                kept_instances = torch.where(valid_scores & valid_scores_2)[0]
                if kept_instances.numel() > 0:
                    instance_mask_preds = instance_mask_preds[kept_instances]
                    instance_score_preds = instance_score_preds[kept_instances]
                    # Number of points in the original input
                    n_points = cylinder_points.shape[0]
                    # Cast masks from the voxelized input back to the original point cloud
                    rows_list = []
                    cols_list = []
                    # Dynamically determine chunk size
                    max_chunk = (
                        self._MAX_INT // instance_mask_preds.shape[0]
                    ) - 1_000_000
                    max_chunk = max(1, max_chunk)
                    chunk_size = min(n_points, max_chunk)
                    # Process data in chunks to avoid overflow errors for large point clouds
                    for start in range(0, n_points, chunk_size):
                        end = min(start + chunk_size, n_points)
                        mask_preds_chunk = instance_mask_preds[
                            :, vox_inverse_map[start:end]
                        ]
                        rows_chunk, cols_chunk = mask_preds_chunk.nonzero(as_tuple=True)
                        cols_chunk = cols_chunk + start
                        rows_list.append(rows_chunk)
                        cols_list.append(cols_chunk)
                    # Concatenate chunks to single tensors
                    rows = torch.cat(rows_list, dim=0)
                    cols = torch.cat(cols_list, dim=0)
                    # Single score value for each instance
                    score_per_instance = instance_score_preds[rows]
                    # Tensor for storing max instance score for each point (original cloud)
                    best_score = torch.full(
                        (n_points,), -1.0, device=cylinder_points_vox.device
                    )
                    # Tensor for storing the row index (i.e. id) of the instance that yields the best score (for each point)
                    best_score_id = torch.full(
                        (n_points,),
                        -1,
                        dtype=torch.long,
                        device=cylinder_points_vox.device,
                    )
                    # Fetch maximum score for each point (across all instances that contain the point)
                    best_score.index_reduce_(0, cols, score_per_instance, reduce="amax")
                    # Mask indicating whether a hit corresponds to the best score for some point
                    best_score_mask = score_per_instance == best_score[cols]
                    # Fetch ids of instance with the highest score for each point
                    best_score_id.index_put_(
                        (cols[best_score_mask],),
                        rows[best_score_mask],
                        accumulate=False,
                    )
                    # Check whether points within the cylinder improved global score improved
                    global_score_improved_mask = (
                        best_score > global_instance_scores[cylinder_point_indices]
                    )
                    if global_score_improved_mask.any():
                        # Update instance scores and ids for points where global scores improved
                        global_instance_scores[
                            cylinder_point_indices[global_score_improved_mask]
                        ] = best_score[global_score_improved_mask]
                        all_preliminary_instances[
                            cylinder_point_indices[global_score_improved_mask]
                        ] = max_instance_id + best_score_id[global_score_improved_mask]
                        # Store instance point indices, ids and scores
                        best_score_id_unq = torch.unique(
                            best_score_id[global_score_improved_mask]
                        )
                        for id in best_score_id_unq:
                            instance_mask = cols[rows == id]
                            best_instances.append(
                                (
                                    cylinder_point_indices[instance_mask],
                                    max_instance_id + int(id),
                                    float(instance_score_preds[int(id)]),
                                )
                            )
                    max_instance_id += int(instance_mask_preds.size(0))
                # Extract semantic predictions
                semantic_preds_vox = result_data["pts_semantic_mask"]
                semantic_preds = semantic_preds_vox[vox_inverse_map]
                # Accumulate votes semantic predictions (final predictions chosen based on majority vote)
                votes_counter.index_put_(
                    (cylinder_point_indices, semantic_preds),
                    torch.ones_like(cylinder_point_indices, dtype=votes_counter.dtype),
                    accumulate=True,
                )
            else:
                # No tree points --> extract semantic predictions from bg/fg semantic predictions
                bg_fg_semantic_logits = bg_fg_semantic_logits[collate_inverse_map]
                semantic_preds_vox = (bg_fg_semantic_logits > 0).long().squeeze(1)
                semantic_preds = semantic_preds_vox[vox_inverse_map]
                # Accumulate votes semantic predictions (final predictions chosen based on majority vote)
                votes_counter.index_put_(
                    (cylinder_point_indices, semantic_preds),
                    torch.ones_like(cylinder_point_indices, dtype=votes_counter.dtype),
                    accumulate=True,
                )
            # NOTE: emptying cache is necessary, otherwise memory usage grows with each iteration (up to a certain point)
            torch.cuda.empty_cache()
        # Post-process results
        # Form final point-level semantic predictions based on majority vote
        final_semantic_labels = torch.argmax(votes_counter, dim=1)
        final_semantic_labels[votes_counter.sum(1) == 0] = -1
        # Remove points classified as background from instances
        bg_mask = final_semantic_labels == self.bg_semantic_class
        all_preliminary_instances[bg_mask] = -1
        # Remove replaced old instances
        unq_best_instances = []
        for instance_indices, instance_id, score in best_instances:
            if torch.any(all_preliminary_instances[instance_indices] == instance_id):
                unq_best_instances.append((instance_indices, instance_id, score))
        # Merge overlapping instances
        final_instance_labels, final_instance_scores = (
            self._merge_overlapping_instances(
                n_points=all_preliminary_instances.shape[0],
                best_instances=unq_best_instances,
                device=final_semantic_labels.device,
                overlap_threshold=self.overlap_th,
                allow_overwrite=False,
            )
        )
        # Re-label instances to ensure continuous labeling
        non_bg_mask = final_instance_labels >= 0
        _, instance_labels_remapped = torch.unique(
            final_instance_labels[non_bg_mask], return_inverse=True
        )
        # Set background to 0 instead of -1 in remapped labels. Instance labels start from 1
        final_instance_labels = torch.full_like(final_instance_labels, fill_value=0)
        final_instance_labels[non_bg_mask] = instance_labels_remapped + 1
        # Set negative scores to 0
        final_instance_scores[final_instance_scores < 0] = 0
        # Move the predictions to the cpu
        final_instance_labels = final_instance_labels.cpu().numpy()
        final_instance_scores = final_instance_scores.cpu().numpy()
        final_semantic_labels = final_semantic_labels.cpu().numpy()

        return final_semantic_labels, final_instance_labels, final_instance_scores

    def _predict_sp(
        self,
        original_points: Tensor,
        pts_superpoint_labels: Tensor,
        regions: List[Tuple[float, float]],
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        multi_cylinder = len(regions) > 1
        # Ensure the superpoint labels are continuous. Also compute sizes (superpoint-level point-counts)
        # if size-aware processing is enabled
        if self.sp_size_aware:
            _, pts_superpoint_labels, superpoint_sizes = torch.unique(
                pts_superpoint_labels,
                return_inverse=True,
                return_counts=True,
                sorted=True,
            )
        else:
            _, pts_superpoint_labels = torch.unique(
                pts_superpoint_labels,
                return_inverse=True,
                sorted=True,
            )
            superpoint_sizes = None
        n_superpoints = torch.unique(pts_superpoint_labels).shape[0]
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
        votes_counter = torch.zeros(
            (n_superpoints, self.num_semantic_classes),
            device=original_points.device,
            dtype=torch.int16,
        )
        all_preliminary_instances = torch.full(
            (n_superpoints,),
            -1,
            device=original_points.device,
        )
        max_instance_id = 0
        global_instance_scores = torch.zeros(
            (n_superpoints,),
            device=original_points.device,
            dtype=torch.float32,
        )
        # List of tuples corresponding to best instances (instance point indices, instance id, instance score)
        best_instances = []
        for region in tqdm(regions, desc="==> Processing cylindrical regions"):
            region_mask = (
                (original_points[:, 0] - region[0]) ** 2
                + (original_points[:, 1] - region[1]) ** 2
            ) <= self.radius**2
            cylinder_points = original_points[region_mask]
            cylinder_point_indices = torch.where(region_mask)[0]
            cylinder_superpoint_labels = pts_superpoint_labels[region_mask]
            # Make labels continuous
            unq_cylinder_superpoint_labels, cylinder_superpoint_labels = torch.unique(
                cylinder_superpoint_labels,
                return_inverse=True,
                sorted=True,
            )
            # Skip if no points within region
            if len(cylinder_points) == 0:
                continue
            # Voxelize/downsample input points
            rng = self._get_rng(cylinder_points.device)
            (
                cylinder_points_vox,
                cylinder_superpoint_labels_vox,
                _,
            ) = self._grid_sample(
                points=cylinder_points,
                superpoint_labels=cylinder_superpoint_labels,
                indices=cylinder_point_indices,
                voxel_size=self.voxel_size,
                rng=rng,
                max_int=self._MAX_INT,
                num_points=self.num_points,
                return_inverse=False,
            )
            # Transform voxelized data into sparse tensor and extract features
            coordinates, features, collate_inverse_map, spatial_shape = self._collate(
                [cylinder_points_vox]
            )
            x = spconv.SparseConvTensor(features, coordinates, spatial_shape, n_samples)
            x = self.extract_feat(x)
            embed_logits = self.embed(x[0])
            bg_fg_semantic_logits = self.bg_fg_semantic_head(x[0])
            # Superpoint pooling
            (
                unq_cylinder_superpoint_labels_vox,
                cylinder_superpoint_labels_vox,
            ) = torch.unique(
                cylinder_superpoint_labels_vox,
                return_inverse=True,
                sorted=True,
            )
            x_sp, bg_fg_semantic_logits_sp, embed_logits_sp, xyz_sp, _  = (
                self._superpoint_mean_pooling(
                    superpoint_labels=cylinder_superpoint_labels_vox,
                    inverse_mappings=collate_inverse_map,
                    x=x,
                    bg_fg_semantic_logits=bg_fg_semantic_logits,
                    embed_logits=embed_logits,
                    xyz=cylinder_points_vox,
                    instance_labels=None,
                )
            )
            xyz_sp = xyz_sp[0]
            query_point_feats, _, _ = self._get_isa_guided_queries(
                x_sp[0],
                bg_fg_semantic_logits_sp[0],
                embed_logits_sp[0],
            )
            # Global (original) superpoint labels for each superpoint within the cylinder ->
            # Since superpoints are continuous and in the range [0, M], the labels also correspond
            # to the global indiced of the superpoint
            global_superpoint_labels = unq_cylinder_superpoint_labels[
                unq_cylinder_superpoint_labels_vox
            ]
            if self.sp_size_aware:
                cylinder_superpoint_sizes = superpoint_sizes[global_superpoint_labels]
            else:
                cylinder_superpoint_sizes = None
            if query_point_feats is not None:
                x = self.decoder(
                    feats=x_sp,
                    queries=[query_point_feats],
                )
                result_data = self._collect_pred_masks(
                    decoder_out=x,
                    superpoint_sizes=cylinder_superpoint_sizes,
                    test_mode=True,
                    coordinates=xyz_sp,
                )
                # Collect masks and their scores
                instance_mask_preds = result_data["pts_instance_mask"]
                instance_score_preds = result_data["instance_scores"]
                valid_scores = instance_score_preds > self.score_th
                # Only invalidate superpoints close to cylinder edges if there is more than one cylinder in total
                if multi_cylinder:
                    # Distance to cylinder edge threshold (2 * voxel size from the edge)
                    edge_threshold_sq = (self.radius - 2 * self.voxel_size) ** 2
                    # Compute distances to cylinder center for all points within the cylinder.
                    # Distances are computed for all (voxelized) points and the aggregated at the
                    # superpoint level by taking the maximum distance within the superpoint
                    center_dists_sq = (cylinder_points_vox[:, 0] - region[0]) ** 2 + (
                        cylinder_points_vox[:, 1] - region[1]
                    ) ** 2
                    center_dists_sq_sp, _ = scatter_max(
                        center_dists_sq, cylinder_superpoint_labels_vox, dim=0
                    )
                    center_dists_sq_sp = center_dists_sq_sp.unsqueeze(0)
                    edge_th_violation_mask = (
                        center_dists_sq_sp > edge_threshold_sq
                    ) & instance_mask_preds
                    edge_th_violation_mask = edge_th_violation_mask.any(dim=1)
                    valid_scores_2 = ~edge_th_violation_mask
                else:
                    valid_scores_2 = torch.ones_like(
                        valid_scores,
                        dtype=torch.bool,
                        device=valid_scores.device,
                    )
                # Binary mask for instances that are retained
                kept_instances = torch.where(valid_scores & valid_scores_2)[0]
                if kept_instances.numel() > 0:
                    instance_mask_preds = instance_mask_preds[kept_instances]
                    instance_score_preds = instance_score_preds[kept_instances]
                    # Number of superpoints in the current cylinder (voxelized)
                    n_superpoints = xyz_sp.shape[0]
                    rows_list = []
                    cols_list = []
                    # Dynamically determine chunk size
                    max_chunk = (
                        self._MAX_INT // instance_mask_preds.shape[0]
                    ) - 1_000_000
                    max_chunk = max(1, max_chunk)
                    chunk_size = min(n_superpoints, max_chunk)
                    # Process data in chunks to avoid overflow errors for very large point clouds
                    for start in range(0, n_superpoints, chunk_size):
                        end = min(start + chunk_size, n_superpoints)
                        mask_preds_chunk = instance_mask_preds[:, start:end]
                        rows_chunk, cols_chunk = mask_preds_chunk.nonzero(as_tuple=True)
                        cols_chunk = cols_chunk + start
                        rows_list.append(rows_chunk)
                        cols_list.append(cols_chunk)
                    # Concatenate chunks to single tensors
                    rows = torch.cat(rows_list, dim=0)
                    cols = torch.cat(cols_list, dim=0)
                    # Single score value for each instance
                    score_per_instance = instance_score_preds[rows]
                    # Tensor for storing max instance score for each point
                    device = xyz_sp.device
                    best_score = torch.full((n_superpoints,), -1.0, device=device)
                    # Tensor for storing the row index (i.e. id) of the instance that yields the best score (for each superpoint)
                    best_score_id = torch.full(
                        (n_superpoints,),
                        -1,
                        dtype=torch.long,
                        device=device,
                    )
                    # Fetch maximum score for each superpoint (across all instances that contain the superpoint)
                    best_score.index_reduce_(0, cols, score_per_instance, reduce="amax")
                    # Mask indicating whether a hit corresponds to the best score for some superpoint
                    best_score_mask = score_per_instance == best_score[cols]
                    # Fetch ids of instance with the highest score for each superpoint
                    best_score_id.index_put_(
                        (cols[best_score_mask],),
                        rows[best_score_mask],
                        accumulate=False,
                    )
                    # Check whether global score improved for any superpoint
                    global_score_improved_mask = (
                        best_score > global_instance_scores[global_superpoint_labels]
                    )
                    if global_score_improved_mask.any():
                        # Update instance scores and ids for superpoints where global scores improved
                        global_instance_scores[
                            global_superpoint_labels[global_score_improved_mask]
                        ] = best_score[global_score_improved_mask]
                        all_preliminary_instances[
                            global_superpoint_labels[global_score_improved_mask]
                        ] = max_instance_id + best_score_id[global_score_improved_mask]
                        # Store instance point indices, ids and scores
                        best_score_id_unq = torch.unique(
                            best_score_id[global_score_improved_mask]
                        )
                        for id in best_score_id_unq:
                            instance_mask = cols[rows == id]
                            best_instances.append(
                                (
                                    global_superpoint_labels[instance_mask],
                                    max_instance_id + int(id),
                                    float(instance_score_preds[int(id)]),
                                )
                            )
                    max_instance_id += int(instance_mask_preds.size(0))
                # Extract semantic predictions
                semantic_preds = result_data["pts_semantic_mask"]
                # Accumulate votes semantic predictions (final predictions chosen based on majority vote)
                votes_counter.index_put_(
                    (global_superpoint_labels, semantic_preds),
                    torch.ones_like(
                        global_superpoint_labels, dtype=votes_counter.dtype
                    ),
                    accumulate=True,
                )
            else:
                # No tree points --> extract semantic predictions from bg/fg semantic predictions
                semantic_preds = (bg_fg_semantic_logits_sp[0] > 0).long().squeeze(1)
                # Accumulate votes semantic predictions (final predictions chosen based on majority vote)
                votes_counter.index_put_(
                    (global_superpoint_labels, semantic_preds),
                    torch.ones_like(
                        global_superpoint_labels, dtype=votes_counter.dtype
                    ),
                    accumulate=True,
                )
            # NOTE: emptying cache is necessary, otherwise memory usage grows with each iteration (up to a certain point)
            torch.cuda.empty_cache()
        # Post-process results
        # Form final superpoint-level semantic predictions based on majority vote
        final_semantic_labels = torch.argmax(votes_counter, dim=1)
        final_semantic_labels[votes_counter.sum(1) == 0] = -1
        # Remove points classified as background from instances
        bg_mask = final_semantic_labels == self.bg_semantic_class
        all_preliminary_instances[bg_mask] = -1
        # Remove replaced old instances
        unq_best_instances = []
        for instance_indices, instance_id, score in best_instances:
            if torch.any(all_preliminary_instances[instance_indices] == instance_id):
                unq_best_instances.append((instance_indices, instance_id, score))
        # Merge overlapping instances
        final_instance_labels, final_instance_scores = (
            self._merge_overlapping_instances(
                n_points=all_preliminary_instances.shape[0],
                best_instances=unq_best_instances,
                device=final_semantic_labels.device,
                superpoint_sizes=superpoint_sizes,
                overlap_threshold=self.overlap_th,
                allow_overwrite=False,
            )
        )
        # Re-label instances to ensure continuous labeling
        non_bg_mask = final_instance_labels >= 0
        _, instance_labels_remapped = torch.unique(
            final_instance_labels[non_bg_mask], return_inverse=True
        )
        # Set background to 0 instead of -1 in remapped labels. Instance labels start from 1
        final_instance_labels = torch.full_like(final_instance_labels, fill_value=0)
        final_instance_labels[non_bg_mask] = instance_labels_remapped + 1
        # Set negative scores to 0
        final_instance_scores[final_instance_scores < 0] = 0
        # Move the predictions to the cpu
        final_instance_labels = final_instance_labels.cpu().numpy()
        final_instance_scores = final_instance_scores.cpu().numpy()
        final_semantic_labels = final_semantic_labels.cpu().numpy()
        # Cast from superpoint-level representation back to full point cloud
        pts_superpoint_labels = pts_superpoint_labels.cpu().numpy()
        final_instance_labels = final_instance_labels[pts_superpoint_labels]
        final_instance_scores = final_instance_scores[pts_superpoint_labels]
        final_semantic_labels = final_semantic_labels[pts_superpoint_labels]

        return final_semantic_labels, final_instance_labels, final_instance_scores

    def _collect_pred_masks(
        self,
        decoder_out: Dict,
        superpoint_sizes: Optional[Tensor] = None,
        test_mode: bool = False,
        coordinates: Optional[Tensor] = None,
    ) -> Union[Dict[str, Tensor], List[PointData]]:
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            decoder_out (Dict): Decoder output, each value is List of len 1. Keys:
                * `masks` of shape (n_queries, n_points),
                * `scores` of shape (n_queris, 1) or None.
            superpoint_sizes (Tensor): superpoint sizes, if data is at the superpoint level. Of shape (n_points,). Defaults to None.
            test_mode (bool, optional): if True, 

        Returns:
            Union[Dict[str, Tensor], List[PointData]]: of len 1 with `pts_semantic_mask`, `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        pred_masks = decoder_out["masks"][0]
        pred_scores = decoder_out["scores"][0]
        pred_masks_sem = pred_masks[-self.num_semantic_classes :, :]
        pred_masks_inst = pred_masks[: -self.num_semantic_classes, :]
        pred_scores_inst = pred_scores[: -self.num_semantic_classes, :]
        # Fetch semantic predictions
        semantic_mask_preds = self._collect_pred_sem_masks(pred_masks=pred_masks_sem)
        bg_z_max = None
        if test_mode:
            # Calculate maximum z coordinate of points classified as background
            bg_points = coordinates[semantic_mask_preds == self.bg_semantic_class]
            bg_z_max = (
                bg_points[:, 2].max().item() if bg_points.size(0) > 0 else float("inf")
            )
        # Fetch instance predictions
        instance_mask_preds, instance_labels, instance_scores = (
            self._collect_pred_inst_masks(
                pred_masks=pred_masks_inst,
                pred_scores=pred_scores_inst,
                semantic_preds=semantic_mask_preds,
                superpoint_sizes=superpoint_sizes,
                coordinates=coordinates,
                bg_z_max=bg_z_max,
            )
        )

        if test_mode:
            # Keep results on the GPU in test mode
            result_data = {
                "pts_semantic_mask": semantic_mask_preds,
                "pts_instance_mask": instance_mask_preds,
                "instance_labels": instance_labels,
                "instance_scores": instance_scores,
            }
        else:
            pts_semantic_mask = [semantic_mask_preds.cpu().numpy()]
            pts_instance_mask = [instance_mask_preds.cpu().bool().numpy()]
            result_data = [
                PointData(
                    pts_semantic_mask=pts_semantic_mask,
                    pts_instance_mask=pts_instance_mask,
                    instance_labels=instance_labels.cpu().numpy(),
                    instance_scores=instance_scores.cpu().numpy(),
                )
            ]

        return result_data

    def _collect_pred_inst_masks(
        self,
        pred_masks: Tensor,
        pred_scores: Tensor,
        semantic_preds: Tensor,
        superpoint_sizes: Optional[Tensor] = None,
        coordinates: Optional[Tensor] = None,
        bg_z_max: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Predict instance masks for a single scene. Discard instance for which >50% points in the instance
        were predicted to be in a non-instance semantic class

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queries, 1).
            semantic_preds (Tensor): predicted semantic labels of shape (n_points, 1)
            superpoint_sizes (Optional[Tensor]): size of each superpoint (n_points, 1). Defaults to None.
            coordinates (Optional[Tensor]): data xyz-coordinates. Defaults to None.
            bg_z_max (Optional[int]): maximum z-coordinate in predicted background points. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                * Tensor: mask_preds of shape (n_preds, n_raw_points),
                * Tensor: labels of shape (n_preds,),
                * Tensor: scores of shape (n_preds,).
        """
        assert (coordinates is None) == (bg_z_max is None), (
            "'coordinates' and 'bg_z_max' should either both have a value or both be 'None'!"
        )
        sp_size_aware = superpoint_sizes is not None
        if sp_size_aware:
            superpoint_sizes = superpoint_sizes.unsqueeze(0)

        # Only one instance class in data (trees)
        n_preds = pred_masks.shape[0]
        labels = torch.full(
            (n_preds,), self.fg_semantic_class, device=pred_scores.device
        )
        # Fetch k highest scoring instances
        pred_scores, topk_ind = pred_scores.flatten(0, 1).topk(
            min(self.test_cfg.topk_insts, n_preds), sorted=False
        )
        labels = labels[topk_ind]
        pred_masks = pred_masks[topk_ind]
        pred_masks_sigmoid = pred_masks.sigmoid()

        if self.test_cfg.get("obj_normalization", None):
            # Normalize instance scores above a user-defined treshold
            mask_pred_thr = pred_masks_sigmoid > self.test_cfg.obj_normalization_thr
            if sp_size_aware:
                mask_scores = (
                    pred_masks_sigmoid * mask_pred_thr * superpoint_sizes
                ).sum(1) / ((mask_pred_thr * superpoint_sizes).sum(1) + 1e-6)
            else:
                mask_scores = (pred_masks_sigmoid * mask_pred_thr).sum(1) / (
                    mask_pred_thr.sum(1) + 1e-6
                )
            pred_scores = pred_scores * mask_scores

        if self.test_cfg.get("nms", None):
            # Apply non-maximum supression on instances
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                pred_masks_sigmoid,
                labels,
                pred_scores,
                kernel=kernel,
                superpoint_sizes=superpoint_sizes,
            )
        # Final boolean masks are constructed by setting all points for which the probability is above  sp_score_thr to true
        pred_masks = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        bg_classes_tensor = torch.tensor(
            [self.bg_semantic_class], device=semantic_preds.device
        )
        # Set scores to 0 when the majority of points within a mask are predicted to be background
        is_bg = torch.isin(semantic_preds, bg_classes_tensor).float()
        if sp_size_aware:
            num_bg_points_in_mask = (pred_masks * is_bg * superpoint_sizes).sum(
                dim=1
            )
            num_points_in_mask = (pred_masks * superpoint_sizes).sum(dim=1)
        else:
            num_bg_points_in_mask = (pred_masks * is_bg).sum(dim=1)
            num_points_in_mask = pred_masks.sum(dim=1)
        scores[num_bg_points_in_mask > (num_points_in_mask / 2)] = 0
        # Filter instances based on z-coordinate values (if coordinate data available)
        if coordinates is not None:
            z_values = coordinates[:, 2]
            pred_z_values = torch.where(pred_masks, z_values.unsqueeze(0), torch.inf)
            pred_min_z_values = pred_z_values.min(dim=1).values
            reject_mask = pred_min_z_values > (bg_z_max + self.test_cfg.z_diff_thr)
            scores[reject_mask] = 0
        # Filter out small instances
        if sp_size_aware:
            num_points_in_mask = (pred_masks * superpoint_sizes).sum(1)
        else:
            num_points_in_mask = pred_masks.sum(1)
        size_accept_mask = num_points_in_mask > self.test_cfg.npoint_thr
        scores = scores[size_accept_mask]
        labels = labels[size_accept_mask]
        pred_masks = pred_masks[size_accept_mask]

        return pred_masks, labels, scores

    def _collect_pred_sem_masks(self, pred_masks: Tensor) -> Tensor:
        """Predict semantic masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_points, n_semantic_classes).

        Returns:
            Tensor: semantic preds of shape (n_raw_points, 1).
        """
        mask_pred = pred_masks.sigmoid()
        semantic_preds = mask_pred.argmax(0)

        return semantic_preds

    def _get_rng(self, device: torch.device) -> torch.Generator:
        """Fetch an rng object for the given device

        Args:
            device (torch.device): device to fetch generator for

        Returns:
            torch.Generator: rng object
        """
        if self._rng.device != device:
            new_rng = torch.Generator(device=device)
            new_rng.set_state(self._rng.get_state())
            self._rng = new_rng

        return self._rng

    def _superpoint_mean_pooling(
        self,
        superpoint_labels: Union[List[Tensor], Tensor],
        inverse_mappings: Optional[Union[List[Tensor], Tensor]],
        x: Union[List[Tensor], Tensor],
        bg_fg_semantic_logits: Optional[Union[List[Tensor], Tensor]] = None,
        embed_logits: Optional[Union[List[Tensor], Tensor]] = None,
        xyz: Optional[Union[List[Tensor], Tensor]] = None,
        instance_labels: Optional[Union[List[Tensor], Tensor]] = None,
    ) -> Tuple[
        List[Tensor],
        Optional[List[Tensor]],
        Optional[List[Tensor]],
        Optional[List[Tensor]],
        Optional[List[Tensor]],
    ]:
        """Perform mean pooling for tensors or lists of tensors. Before pooling, cast back to non-voxelized coordinates using the inverse mapping, if available.

        Args:
            superpoint_labels (Union[List[Tensor], Tensor]): superpoint labels (n_raw_points) or (n_points) is no inverse mappings available.
            inverse_mappings (Optional[Union[List[Tensor], Tensor]]): inverse map to non-voxelized format (n_raw_points)
            x (Union[List[Tensor], Tensor]): features, voxelized (n_points)
            bg_fg_semantic_logits (Optional[Union[List[Tensor], Tensor]], optional): background/foregroind logits, voxelized (n_points). Defaults to None.
            embed_logits (Optional[Union[List[Tensor], Tensor]], optional): instance discriminative features, voxelized (n_points). Defaults to None.
            xyz (Optional[Union[List[Tensor], Tensor]], optional): xyz-coordinates. Defaults to None.
            instance_labels (Optional[Union[List[Tensor], Tensor]], optional): instance labels, unvoxelized. Defaults to None.

        Returns:
            Tuple[ List[Tensor], Optional[List[Tensor]], Optional[List[Tensor]], Optional[List[Tensor]], Optional[List[Tensor]], ]: _description_
        """
        superpoint_labels = self._init_for_sp_pooling(superpoint_labels)
        n_scenes = len(superpoint_labels)
        inverse_mappings = self._init_for_sp_pooling(
            inverse_mappings, none_flag=True, n_scenes=n_scenes
        )
        x = self._init_for_sp_pooling(x)
        bg_fg_semantic_logits = self._init_for_sp_pooling(
            bg_fg_semantic_logits, none_flag=True, n_scenes=n_scenes
        )
        embed_logits = self._init_for_sp_pooling(
            embed_logits, none_flag=True, n_scenes=n_scenes
        )
        xyz = self._init_for_sp_pooling(xyz, none_flag=True, n_scenes=n_scenes)
        instance_labels = self._init_for_sp_pooling(
            instance_labels, none_flag=True, n_scenes=n_scenes
        )
        x_sp = []
        bg_fg_semantic_logits_sp = []
        embed_logits_sp = []
        xyz_sp = []
        instance_labels_sp = []

        for i in range(n_scenes):
            inverse_mapping_i = inverse_mappings[i]
            x_i = x[i]
            bg_fg_semantic_logits_i = bg_fg_semantic_logits[i]
            embed_logits_i = embed_logits[i]
            xyz_i = xyz[i]
            instance_labels_i = instance_labels[i]
            # Map features back to their unvoxelized state such that the number of points matches the number of superpoint labels
            if inverse_mapping_i is not None:
                x_i = x_i[inverse_mapping_i]
                if bg_fg_semantic_logits_i is not None:
                    bg_fg_semantic_logits_i = bg_fg_semantic_logits_i[inverse_mapping_i]
                if embed_logits_i is not None:
                    embed_logits_i = embed_logits_i[inverse_mapping_i]
                if xyz_i is not None:
                    xyz_i = xyz_i[inverse_mapping_i]
            # Perform mean pooling
            x_sp.append(scatter_mean(x_i, superpoint_labels[i], dim=0))
            if bg_fg_semantic_logits_i is not None:
                bg_fg_semantic_logits_sp.append(
                    scatter_mean(
                        bg_fg_semantic_logits_i,
                        superpoint_labels[i],
                        dim=0,
                    )
                )
            if embed_logits_i is not None:
                embed_logits_sp.append(
                    scatter_mean(embed_logits_i, superpoint_labels[i], dim=0)
                )
            if xyz_i is not None:
                xyz_sp.append(scatter_mean(xyz_i, superpoint_labels[i], dim=0))
            if instance_labels_i is not None:
                instance_labels_sp.append(
                    self._get_superpoint_instance_labels(
                        instance_labels_i,
                        superpoint_labels[i],
                        remap_to_contiguous=False,
                    )
                )
        if len(bg_fg_semantic_logits_sp) <= 0:
            bg_fg_semantic_logits_sp = None
        if len(embed_logits_sp) <= 0:
            embed_logits_sp = None
        if len(xyz_sp) <= 0:
            xyz_sp = None
        if len(instance_labels_sp) <= 0:
            instance_labels_sp = None

        return (
            x_sp,
            bg_fg_semantic_logits_sp,
            embed_logits_sp,
            xyz_sp,
            instance_labels_sp,
        )

    @staticmethod
    def _init_for_sp_pooling(
        input: Optional[Union[List[Tensor], Tensor]],
        none_flag: bool = False,
        n_scenes: int = 0,
    ) -> List[Optional[Tensor]]:
        if isinstance(input, Tensor):
            input = [input]
        if none_flag and (input is None):
            input = [None] * n_scenes

        return input

    @staticmethod
    def _get_inverse_mappings(
        batch_size: int, collate_inverse_mapping: Tensor, coordinates: Tensor
    ) -> List[Tensor]:
        inverse_mappings = []
        for i in range(batch_size):
            inverse_mapping_i = collate_inverse_mapping[
                coordinates[:, 0][collate_inverse_mapping] == i
            ]
            _, inverse_mapping_i = torch.unique(inverse_mapping_i, return_inverse=True)
            inverse_mappings.append(inverse_mapping_i)

        return inverse_mappings

    @staticmethod
    def _get_superpoint_instance_labels(
        pts_instance_mask: Tensor,
        superpoint_labels: Tensor,
        remap_to_contiguous: bool = False,
    ) -> Tensor:
        """Aggregrate instance labels at the superpoint level by majority vote

        Args:
            pts_instance_mask (Tensor): original instance mask (n_points)
            superpoint_labels (Tensor): superpoint_labels (n_points)
            remap_to_contiguous (bool, optional): If True, remap instance labels to range [0, C]. Defaults to False.

        Returns:
            Tensor: _description_
        """
        _, superpoint_ids = torch.unique(
            superpoint_labels,
            return_inverse=True,
            sorted=True,
        )
        # Handle background (label -1)
        pts_instance_mask = pts_instance_mask.clone()
        shift = torch.any(pts_instance_mask == -1)
        if shift:
            # Shift to make background 0
            pts_instance_mask += 1
        # Remap instance ids to contiguous
        unq_vals, pts_instance_mask = torch.unique(
            pts_instance_mask,
            return_inverse=True,
            sorted=True,
        )
        # Build composite key so each (superpoint, instance) pair becomes unique.
        n_instances = int(pts_instance_mask.max().item()) + 1
        composite = superpoint_ids * n_instances + pts_instance_mask
        # Count occurrences of each pair
        unq, counts = torch.unique(composite, return_counts=True)
        # Recover superpoint + instance
        sp_of_pair = unq // n_instances
        instance_of_pair = unq % n_instances
        score = counts * (n_instances + 1) - instance_of_pair
        _, argmax_ind = scatter_max(score, sp_of_pair, dim=0)
        superpoint_instance_labels = unq_vals[instance_of_pair[argmax_ind]]
        if shift:
            superpoint_instance_labels -= 1

        valid_mask = superpoint_instance_labels != -1
        # Remap labels to a continuous range
        if remap_to_contiguous and valid_mask.any():
            _, remapped = torch.unique(
                superpoint_instance_labels[valid_mask], return_inverse=True, sorted=True
            )
            compact = torch.full_like(superpoint_instance_labels, -1)
            compact[valid_mask] = remapped
            superpoint_instance_labels = compact

        return superpoint_instance_labels

    @staticmethod
    def _get_gt_semantic_masks(
        semantic_mask: Tensor, superpoint_labels: Tensor, num_classes: int
    ) -> Tensor:
        """Create ground truth semantic masks.

        Args:
            pts_semantic_mask (Tensor): semantic labels of shape (n_raw_points, 1).
            superpoint_labels (Tensor): of shape (n_raw_points, 1).
            num_classes (Int): number of semantic classes.

        Returns:
            sp_semantic_masks (Tensor): semantic mask of shape (num_classes, n_points).
        """
        mask = F.one_hot(semantic_mask, num_classes=num_classes).float()
        sp_semantic_masks = scatter_add(mask, superpoint_labels, dim=0)
        sp_semantic_masks = sp_semantic_masks.argmax(dim=-1)
        sp_semantic_masks = F.one_hot(
            sp_semantic_masks, num_classes=num_classes
        ).float()
        sp_semantic_masks = sp_semantic_masks.T
        assert sp_semantic_masks.shape == (
            num_classes,
            superpoint_labels.max().item() + 1,
        )

        return sp_semantic_masks

    @staticmethod
    def _get_gt_instance_masks(
        instance_mask: Tensor, superpoint_labels: Tensor
    ) -> Tensor:
        """Create ground truth instance masks.

        Args:
            instance_mask (Tensor): of shape (n_raw_points, 1).
            superpoint_labels (Tensor): of shape (n_raw_points, 1).

        Returns:
            sp_instance_masks (Tensor): semantic mask of shape (n_points, num_inst_obj).
        """
        mask = instance_mask.clone()

        if torch.sum(mask == -1) != 0:
            mask[mask == -1] = torch.max(mask) + 1
            mask = F.one_hot(mask)[:, :-1]
        else:
            mask = F.one_hot(mask)
        mask = mask.T
        sp_instance_masks = scatter_mean(mask, superpoint_labels, dim=-1)
        sp_instance_masks = sp_instance_masks > 0.5

        return sp_instance_masks

    @staticmethod
    def _filter_bg_masks(
        batch_data_samples_i: Tensor,
        bg_classes: List[int],
        ratio_inspoint: Dict,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Filter out instances in background classes

        Args:
            batch_data_samples_i (Tensor): Batch data.
            bg_classes (List[int]): List of background class labels.
            ratio_inspoint (Dict): ratio of points in each instance that fall within the input boundaries.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Input tensors with instances in background classes filtered out.
        """
        labels_3d = batch_data_samples_i.labels_3d
        sp_instance_masks = batch_data_samples_i.sp_inst_masks
        bg_classes_tensor = torch.tensor(bg_classes, device=labels_3d.device)
        mask = torch.isin(labels_3d, bg_classes_tensor)
        indices_to_keep = ~mask
        filtered_labels_3d = labels_3d[indices_to_keep]
        filtered_sp_instance_masks = sp_instance_masks[indices_to_keep]
        ratio_tensor = torch.zeros(len(labels_3d), device=labels_3d.device)
        for i in range(len(labels_3d)):
            ratio_tensor[i] = ratio_inspoint[i]
        ratio_subset = ratio_tensor[indices_to_keep]

        return filtered_labels_3d, filtered_sp_instance_masks, ratio_subset

    @staticmethod
    def _grid_sample(
        points: torch.Tensor,
        superpoint_labels: torch.Tensor,
        indices: torch.Tensor,
        voxel_size: float,
        rng: torch.Generator,
        max_int: int,
        num_points: Optional[int] = None,
        return_inverse: bool = True,
    ) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """
        Voxel-downsample point cloud by averaging points in each voxel.

        Args:
            points (Tensor): (N, 3) or (N, F) point coordinates/features.
            superpoint_labels: (N,) superpoint label of each point.
            indices (Tensor): (N,) original point indices (int32/int64).
            voxel_size (float): size of each voxel in meters.
            rng (torch.Generator): rng object used for random downsampling when it is necessary.
            num_points (Optional[int]): If set and voxelized point count > num_points, randomly subsample to num_points. Defaults to None.
            chunk_size (Optional[int]): Chunk size to control memory usage when computing nearest voxel for voxels dropped during downsampling. Defaults to None.
            return_inverse (bool, optional): If True, return an inverse map from voxels to original coordinates. Defaults to False.

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                * (M, F) voxelized (and possibly downsampled) point cloud.
                * (M,) superpoint label of each voxel.
                * (M,) one representative original index per voxel.
                * (N,) for each original point, index into the voxelized/downsampled points (if return_inverse is True).
        """
        voxels = torch.floor(points[:, :3] / voxel_size).to(torch.int32)
        unique_voxels, inverse = torch.unique(voxels, return_inverse=True, dim=0)
        M = unique_voxels.size(0)
        # Compute voxel-level mean
        ones = torch.ones_like(inverse, dtype=points.dtype)
        sum_xyz = torch.zeros(
            (M, points.size(1)), device=points.device, dtype=points.dtype
        )
        count_xyz = torch.zeros(M, device=points.device, dtype=points.dtype)
        sum_xyz.index_add_(0, inverse, points)
        count_xyz.index_add_(0, inverse, ones)
        voxel_points = sum_xyz / count_xyz.unsqueeze(1)
        # Pick one representative original index per voxel
        voxel_indices = torch.full((M,), -1, device=indices.device, dtype=indices.dtype)
        voxel_indices.index_copy_(0, inverse, indices)
        # Voxelize superpoint labels
        sp = superpoint_labels.long()
        # Remap superpoints to contiguous
        unique_sp, sp_remap = torch.unique(sp, return_inverse=True, sorted=True)
        n_sp = unique_sp.numel()
        # Composite key: (voxel, superpoint)
        composite = inverse * n_sp + sp_remap
        uniq_pairs, counts = torch.unique(composite, return_counts=True)
        voxel_of_pair = uniq_pairs // n_sp
        sp_of_pair = uniq_pairs % n_sp
        # Tie-break toward smaller superpoint id for determinism
        score = counts * (n_sp + 1) - sp_of_pair
        _, argmax_ind = scatter_max(score, voxel_of_pair, dim=0)
        voxel_superpoints = unique_sp[sp_of_pair[argmax_ind]]
        # Randomly downsample voxels, if necessary
        if num_points is not None and M > num_points:
            # Randomly downsample voxels
            choices = torch.randperm(M, generator=rng, device=points.device)[
                :num_points
            ]
            kept_voxels = unique_voxels[choices]
            kept_voxels_f = kept_voxels.float()
            voxel_points = voxel_points[choices]
            voxel_indices = voxel_indices[choices]
            voxel_superpoints = voxel_superpoints[choices]
            # Computing the nearest neighbors for downsampled points can be quite slow, so it's only
            # done if necessary (return_inverse set to True)
            if return_inverse:
                # Build map from all voxels to the nearest kept voxel
                nn_kept_ind = torch.empty(M, dtype=torch.long, device=points.device)
                # Dynamically determine maximum chunk size
                max_chunk = max_int // kept_voxels_f.shape[0]
                max_chunk = max(1, max_chunk)
                chunk_size = min(unique_voxels.shape[0], max_chunk)
                # Process in chunks to keep memory usage reasonable
                for start in range(0, M, chunk_size):
                    end = min(start + chunk_size, M)
                    if end <= start:
                        continue
                    dists = torch.cdist(
                        unique_voxels[start:end].float(), kept_voxels.float()
                    )
                    nn_kept_ind[start:end] = torch.argmin(dists, dim=1)
                # Map each original point to its nearest kept voxel
                inverse_map = nn_kept_ind[inverse]
        else:
            # No downsampling required
            inverse_map = inverse
        if return_inverse:
            return voxel_points, voxel_superpoints, voxel_indices, inverse_map
        else:
            return voxel_points, voxel_superpoints, voxel_indices

    @staticmethod
    def _merge_overlapping_instances(
        n_points: int,
        best_instances: List[Tuple],
        device: torch.device,
        superpoint_sizes: Optional[Tensor] = None,
        overlap_threshold: float = 0.3,
        allow_overwrite: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Greedily remove overlapping instances. The instances are processed in descending order by score,
        and instances that overlap more than x% with some higher score instance are removed.

        Args:
            n_points (int): Number of points in the original input point cloud.
            best_instances (List[Tuple]): List of Tuples containing instance points, ids and scores.
            device (torch.device): Device the predictions in best_instances are on.
            superpoint_sizes (Optional[Tensor]): size of each superpoint in the data (if at the superpoint level). Defaults to None.
            overlap_threshold (float, optional): Overlap percetange threshold for removing instances. Defaults to 0.3.
            allow_overwrite (bool, optional): If True, overwrite overlapping points with the id of lower scoring instance (for some reason this is default behavior in the official FF3D source code). Defaults to False.

        Returns:
            Tuple[Tensor, Tensor]: instance ids and scores for all points.
        """
        merged_instance_labels = torch.full((n_points,), -1, dtype=int, device=device)
        merged_instance_scores = torch.full(
            (n_points,), -1.0, dtype=float, device=device
        )
        sp_size_aware = superpoint_sizes is not None

        if not best_instances:
            return merged_instance_labels, merged_instance_scores
        # Sort instances by score
        best_instances = sorted(best_instances, key=lambda x: -x[2])
        # Binary mask of points already assigned to some instance
        taken_flag = torch.zeros((n_points,), dtype=bool, device=device)

        for point_indices, instance_id, score in best_instances:
            if sp_size_aware:
                # Get sizes for the superpoints in this instance
                instance_sp_sizes = superpoint_sizes[point_indices]
                overlap_mask = taken_flag[point_indices]
                # Compute weighted overlap
                overlap_size = instance_sp_sizes[overlap_mask].sum()
                total_size = instance_sp_sizes.sum()
                overlap_precentage = (
                    (overlap_size / total_size).item() if total_size > 0 else 0.0
                )
            else:
                overlap_precentage = taken_flag[point_indices].float().mean().item()
            if overlap_precentage > overlap_threshold:
                # Skip instance if percentage of points overlapping with some higher score instance is above threshold
                continue
            if not allow_overwrite:
                # Only assign new instance ids to points that were not previously assigned to some other instance
                # Default behavior (same as original source code) is to not do this
                point_indices = point_indices[~taken_flag[point_indices]]
                if point_indices.numel() == 0:
                    # No free points in instance
                    continue

            merged_instance_labels[point_indices] = instance_id
            merged_instance_scores[point_indices] = score
            taken_flag[point_indices] = True

        return merged_instance_labels, merged_instance_scores
