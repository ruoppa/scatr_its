import os
import mmengine

import numpy as np

from concurrent import futures
from typing import Dict, List, Tuple, Set
from argparse import ArgumentParser, Namespace
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.runner import Runner
from mmengine.registry import RUNNERS


_VALID_FILE_EXTENSIONS = ["las", "laz", "ply"]
_VALID_DATA_SPLITS = ["training", "validation", "test"]
_PREPROCESS_REQUIRED_FIELDS = [
    "raw_data_dir",
    "preprocessed_data_dir",
    "input_data_dir",
    "training_dir",
    "validation_dir",
    "test_dir",
    "feat_to_load",
    "max_n_points",
    "file_extension",
    "instance_field",
    "semantic_field",
    "ignore_semantic_classes",
    "non_instance_id",
    "semantic_class_mapping",
]


def validate_data_config(config: Config) -> None:
    if config.preprocess.file_extension not in _VALID_FILE_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{config.preprocess.file_extension}'. Should be one of '{_VALID_FILE_EXTENSIONS}'!"
        )
    for field in _PREPROCESS_REQUIRED_FIELDS:
        if field not in config.preprocess:
            raise AttributeError(f"Missing required preprocessing argument '{field}'")
    for split in config.data_split.keys():
        if split not in _VALID_DATA_SPLITS:
            raise ValueError(
                f"Unsupported data split '{split}'. Should be one of '{_VALID_DATA_SPLITS}'!"
            )
    n_instance_classes = len(config.general.instance_class_names)
    assert n_instance_classes == 1, (
        f"Forest dataset should have only one instance class, found '{n_instance_classes}' instead!"
    )


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_config",
        required=True,
        help="Dataset config file path.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Superpoint generator config file path. Required in superpoint mode, ignored otherwise.",
    )
    parser.add_argument(
        "--pipeline_config", default="configs/pipelines", help="Pipeline template config file path."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to trained superpoint generator weights. Required in superpoint mode, ignored otherwise.",
    )
    parser.add_argument(
        "--unlabeled_mode",
        "-u",
        action="store_true",
        help="If set, skip ground truth labels during preprocessing.",
    )
    parser.add_argument(
        "--force_rewrite",
        "-f",
        action="store_true",
        help="If set, force rewrite even if file has already been preprocessed before.",
    )
    parser.add_argument(
        "--superpoint_mode",
        "-sp",
        action="store_true",
        help="If set, generate superpoints during preprocessing.",
    )
    parser.add_argument(
        "--deterministic_alg",
        "-d",
        action="store_true",
        help="set 'torch.use_deterministic_algorithms' to True for reproducibility. "
        "Note that this may make the model slightly slower and that not all randomness "
        "can be removed even if this argument is set.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        "be overwritten is a list, it should be like key='[a,b]' or key=a,b "
        "It also allows nested list/tuple values, e.g. key='[(a,b),(c,d)]' "
        "Note that the quotation marks are necessary and that no whitespace "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="Job launcher.",
    )

    return parser


def move_all_to_test(config: Config) -> ConfigDict:
    original_split = config.data_split.copy()
    new_test_split = []
    new_test_split.extend(config.data_split.training)
    new_test_split.extend(config.data_split.validation)
    new_test_split.extend(config.data_split.test)
    config.data_split.training = []
    config.data_split.validation = []
    config.data_split.test = new_test_split

    return original_split


def generate_superpoints(args: Namespace, config: Config) -> None:
    config.launcher = args.launcher
    if args.cfg_options is not None:
        config.merge_from_dict(args.cfg_options)
    if args.superpoint_mode:
        config.work_dir = config.model.test_cfg.superpoint_data_dir
    else:
        config.work_dir = config.model.test_cfg.output_data_dir
    config.load_from = args.checkpoint
    # Always skip accuracy computation for superpoints during preprocessing (to save time)
    config.test_evaluator.unlabeled_mode = True
    if "runner_type" not in config:
        # Build the default runner
        runner = Runner.from_cfg(config)
    else:
        runner = RUNNERS.build(config)
    # Run inference to generate superpoints
    runner.test()


class ForestInputDataGenerator:
    def __init__(
        self,
        in_path: str,
        out_path: str,
        data_split: str,
        split_config: Config,
        instance_class_names: Dict[int, str],
        data_dim: int,
        superpoint_mode: bool,
    ):
        self.in_path = in_path
        self.out_path = out_path
        self._data_split = data_split
        self._split_config = split_config
        self.instance_class_names = instance_class_names
        self.name_to_instance_class = {
            value: key for key, value in instance_class_names.items()
        }
        self.data_dim = data_dim
        self.superpoint_mode = superpoint_mode

    @property
    def data_split(self):
        return self._data_split

    @data_split.setter
    def data_split(self, data_split: str):
        assert data_split in _VALID_DATA_SPLITS, (
            f"'data_split' should be one of '{_VALID_DATA_SPLITS}'. Got '{data_split}' instead!"
        )
        self._data_split = data_split

    def _get_split_filenames(self) -> List[str]:
        return self._split_config[self._data_split]

    def _get_npy_data(self, filename: str, ctg: str) -> np.ndarray:
        npy_data_path = os.path.join(
            self.in_path, f"{filename}_{ctg}.npy"
        )
        mmengine.check_file_exist(npy_data_path)

        return np.load(npy_data_path)

    def _init_instance(self) -> Dict:
        empty_instance = dict(
            bbox=None,
            bbox_label=None,
            bbox_3d=None,
            bbox_3d_isvalid=None,
            bbox_label_3d=None,
            depth=None,
            center_2d=None,
            attr_label=None,
            num_lidar_pts=None,
            num_radar_pts=None,
            difficulty=None,
            unaligned_bbox_3d=None,
            instance_center=None,
        )

        return empty_instance

    def _init_lidar_points(self) -> Dict:
        empty_lidar_points = dict(
            num_pts_feats=None,
            lidar_path=None,
            offsets=None,
            scales=None,
            lidar2ego=None,
        )

        return empty_lidar_points

    def _init_radar_points(self) -> Dict:
        empty_radar_points = dict(
            num_pts_feats=None,
            radar_path=None,
            radar2ego=None,
        )

        return empty_radar_points

    def _init_img_info(self) -> Dict:
        empty_img_info = dict(
            img_path=None,
            height=None,
            width=None,
            depth_map=None,
            cam2img=None,
            lidar2img=None,
            cam2ego=None,
        )

        return empty_img_info

    def _init_single_image_sweep(self, camera_types: List[str]) -> Dict:
        empty_single_image_sweep = dict(timestamp=None, ego2global=None)
        images = dict()
        for cam_type in camera_types:
            images[cam_type] = self._init_img_info()
        empty_single_image_sweep["images"] = images

        return empty_single_image_sweep

    def _init_standard_data_info(
        self, camera_types: List[str] = ["CAM0", "CAM1", "CAM2", "CAM3", "CAM4"]
    ) -> Dict:
        empty_data_info = dict(
            sample_idx=None,
            token=None,
            **self._init_single_image_sweep(camera_types),
            lidar_points=self._init_lidar_points(),
            radar_points=self._init_radar_points(),
            image_sweeps=[],
            lidar_sweeps=[],
            instances=[],
            instances_ignore=[],
            pts_semantic_mask_path=None,
            pts_instance_mask_path=None,
            offsets=[],
            instance_centers=None,
        )

        return empty_data_info

    def _clear_instance_unused_keys(self, instance: Dict):
        keys = list(instance.keys())
        for k in keys:
            if instance[k] is None:
                del instance[k]

        return instance

    def _clear_data_info_unused_keys(self, data_info: Dict) -> Tuple[Dict, bool]:
        keys = list(data_info.keys())
        empty_flag = True
        for key in keys:
            if key in ["instances", "cam_sync_instances", "cam_instances"]:
                empty_flag = False
                continue
            if isinstance(data_info[key], list):
                if len(data_info[key]) == 0:
                    del data_info[key]
                else:
                    empty_flag = False
            elif data_info[key] is None:
                del data_info[key]
            elif isinstance(data_info[key], dict):
                _, sub_empty_flag = self._clear_data_info_unused_keys(data_info[key])
                if sub_empty_flag is False:
                    empty_flag = False
                else:
                    del data_info[key]
            else:
                empty_flag = False

        return data_info, empty_flag

    def _process_single_file(self, filename: str) -> Tuple[Dict, Set]:
        """Process a single file from preprocessed npy files to the standard format in OpenMMLab V2.0

        Args:
            filename (str): name of the file to process
        """
        # Load points
        file_info = self._init_standard_data_info()
        file_info["lidar_points"]["num_pts_feats"] = self.data_dim
        points = self._get_npy_data(filename, "points")
        points_path = os.path.join(self.out_path, "points")
        mmengine.mkdir_or_exist(points_path)
        points.tofile(os.path.join(points_path, f"{filename}.bin"))
        file_info["lidar_points"]["lidar_path"] = f"{filename}.bin"
        instance_labels_path = os.path.join(
            self.out_path, "instance_mask"
        )
        semantic_labels_path = os.path.join(
            self.out_path, "semantic_mask"
        )
        superpoint_labels_path = os.path.join(
            self.out_path, "superpoint_labels"
        )
        # Load and process annotations
        if not self.superpoint_mode:
            # In superpoint mode it is assumed that the .bin files exist already, so they do not need to
            # be generated again
            instance_labels = self._get_npy_data(filename, "instance_labels")
            semantic_labels = self._get_npy_data(filename, "semantic_labels")
            mmengine.mkdir_or_exist(instance_labels_path)
            mmengine.mkdir_or_exist(semantic_labels_path)
            instance_labels.tofile(
                os.path.join(instance_labels_path, f"{filename}.bin")
            )
            semantic_labels.tofile(
                os.path.join(semantic_labels_path, f"{filename}.bin")
            )
        else:
            superpoint_labels = self._get_npy_data(filename, "superpoint_labels")
            mmengine.mkdir_or_exist(superpoint_labels_path)
            superpoint_labels.tofile(
                os.path.join(superpoint_labels_path, f"{filename}.bin")
            )

        file_info["pts_instance_mask_path"] = f"{filename}.bin"
        file_info["pts_semantic_mask_path"] = f"{filename}.bin"
        file_info["pts_superpoint_labels_path"] = f"{filename}.bin"
        # Process bounding boxes
        ignored_classes = set()
        aligned_bboxes = self._get_npy_data(filename, "aligned_bboxes")
        # We assume that instance centers occur in the same order as bboxes (generated at the same time so they should)
        instance_centers = self._get_npy_data(filename, "instance_centers")
        n_gt = aligned_bboxes.shape[0]
        n_centers = instance_centers.shape[0]
        assert n_gt == n_centers, (
            f"Number of boundig boxes and instance centers does not match! ('{n_gt}' and '{n_centers}')"
        )
        if n_gt > 0:
            axis_align_matrix = self._get_npy_data(filename, "axis_align_matrix")
            file_info["axis_align_matrix"] = axis_align_matrix
            instance_list = []
            for i in range(n_gt):
                empty_instance = self._init_instance()
                empty_instance["bbox_3d"] = aligned_bboxes[i, :-1].tolist()
                empty_instance["instance_center"] = instance_centers[i, :].tolist()
                instance_label = aligned_bboxes[i, -1]
                instance_name = self.instance_class_names.get(instance_label, None)
                if instance_name is not None:
                    empty_instance["bbox_label_3d"] = instance_label
                else:
                    ignored_classes.add(instance_label)
                    empty_instance["bbox_label_3d"] = -1
                empty_instance = self._clear_instance_unused_keys(empty_instance)
                instance_list.append(empty_instance)

            file_info["instances"] = instance_list
        # Load offset and scale and add to file info
        offsets = self._get_npy_data(filename, "offsets")
        scales = self._get_npy_data(filename, "scales")
        file_info["lidar_points"]["offsets"] = offsets.tolist()
        file_info["lidar_points"]["scales"] = scales.tolist()

        file_info["instace_centers"] = instance_centers
        # Remove unused keys
        file_info, _ = self._clear_data_info_unused_keys(file_info)

        return file_info, ignored_classes

    def process(self, num_workers: int = 4) -> List[Dict]:
        split_filenames = self._get_split_filenames()
        with futures.ThreadPoolExecutor(num_workers) as executor:
            outputs = executor.map(self._process_single_file, split_filenames)

        file_infos = [info for info, _ in outputs]
        ignored_classes = [c for _, c in outputs]
        ignored_classes = list(set().union(*ignored_classes))

        return file_infos, ignored_classes
