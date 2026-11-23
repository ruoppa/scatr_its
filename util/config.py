import os

from typing import Optional, Dict, Tuple
from omegaconf import OmegaConf, DictConfig
from mmengine.config import Config


_FIELDS_WITH_PATHS = [
    "preprocess",
    "output",
    "accuracy",
]
_PATHS_TO_FORMAT = [
    "preprocessed_data_dir",
    "input_data_dir",
    "training_dir",
    "validation_dir",
    "test_dir",
    "output_data_dir",
    "superpoint_data_dir",
]
# Directories for the different data splits in the raw data (handled differently)
_PATHS_DATA_SPLIT = [
    "training_dir",
    "validation_dir",
    "test_dir",
]
# Directories where the value None is skipped (an error is raised otherwise, except for split directories)
_SKIP_IF_NONE = []
# Config resolver for integer addition
OmegaConf.register_new_resolver(
    "add",
    lambda x, y: int(x) + int(y),
)
# Config resolver to dynamically create the num_planes list from the number of channels in the backbone
OmegaConf.register_new_resolver(
    "channel_count",
    lambda dim_list: len(dim_list)
)
# Config resolver to dynamically create the num_planes list from the number of channels in the backbone
OmegaConf.register_new_resolver(
    "channel_list",
    lambda num_channels, n: [
        num_channels * (i + 1) for i in range(int(n))
    ],
)

def _format_all_paths(config: DictConfig) -> None:
    """Format given directory names in the config into full paths (in-place)

    Args:
        config (OmegaConf): The main dataset config
    """
    for field in _FIELDS_WITH_PATHS:
        for key, val in config.dataset[field].items():
            if key in _PATHS_TO_FORMAT:
                formatted_path = _format_path(config, val, key)
                if (formatted_path is None) and (formatted_path not in _SKIP_IF_NONE):
                    raise ValueError(f"Value of '{key}' in the configuration file should not be 'None'!")
                config.dataset[field][key] = formatted_path
    # Metadata file path formatted differently (if exists)
    config.dataset.accuracy.metadata_file = _format_metadata_path(config)
    # Add formatted paths to model test_cfg
    if "output_data_dir" in _PATHS_TO_FORMAT:
        if "test_cfg" in config.model:
            config.model.test_cfg.output_data_dir = config.dataset.output.output_data_dir
        else:
            config.model.test_cfg = {
                "ouput_data_dir": config.dataset.output.output_data_dir
            }
    if "superpoint_data_dir" in _PATHS_TO_FORMAT:
        if "test_cfg" in config.superpoint_model:
            config.superpoint_model.test_cfg.superpoint_data_dir = config.dataset.output.superpoint_data_dir
        else:
            config.superpoint_model.test_cfg = {
                "superpoint_data_dir": config.dataset.output.superpoint_data_dir
            }


def _format_path(config: DictConfig, value: Optional[str], key: str) -> Optional[str]:
    formatted_path = None
    if key in _PATHS_DATA_SPLIT:
        if value is None:
            formatted_path = os.path.join(
                config.dataset.data_root,
                config.dataset.preprocess.raw_data_dir,
                config.dataset.dataset_name,
            )
        else:
            formatted_path = os.path.join(
                config.dataset.data_root,
                config.dataset.preprocess.raw_data_dir,
                config.dataset.dataset_name,
                value
            )
    elif value is not None:
        formatted_path = os.path.join(
            config.dataset.data_root,
            value,
            config.dataset.dataset_name,
        )
    
    return formatted_path


def _format_metadata_path(config: DictConfig) -> Optional[str]:
    formatted_path = None
    if config.dataset.accuracy.metadata_file is not None:
        formatted_path = os.path.join(
            config.dataset.data_root,
            config.dataset.preprocess.raw_data_dir,
            config.dataset.dataset_name,
            config.dataset.accuracy.metadata_file
        )
    
    return formatted_path


def _update_pipelines(config: Config) -> None:
    """Update dataloader pipelines to fully formatted values and update test evaluator to validation evaluator

    Args:
        config (Config): model config file
    """
    config.train_dataloader.dataset.pipeline = config.train_pipeline
    config.val_dataloader.dataset.pipeline = config.val_pipeline
    config.test_dataloader.dataset.pipeline = config.test_pipeline
    config.test_evaluator = config.val_evaluator


def _format_extra_features(config: Config, data_config: Config) -> None:
    if (len(data_config.general.use_dim) > 3) and (data_config.preprocess.feat_to_load is not None):
        extra_features = {}
        # Index of extra feature in the dataloader point array (differs from index in full input if only some features are loaded)
        extra_feat_ind = 3
        for i, feat_name in enumerate(data_config.preprocess.feat_to_load):
            if (i + 3) in data_config.general.use_dim:
                # Add _f postfix for internal processing to prevent errors when saving in las/laz format
                extra_features[extra_feat_ind] = f"{feat_name}_f"
                extra_feat_ind += 1
        config.model.test_cfg["extra_features"] = extra_features
    else:
        config.model.test_cfg["extra_features"] = None


def _merge_configs(
    base: Dict,
    overrides: Dict,
    path: str = "",
    override_existing: bool = True,
) -> Dict:
    if isinstance(base, dict) and isinstance(overrides, dict):
        for key, value in overrides.items():
            current_path = f"{path}{key}"
            if key not in base:
                raise KeyError(f"Unknown config parameter: {current_path}")
            base[key] = _merge_configs(
                base[key],
                value,
                path=current_path + ".",
                override_existing=override_existing,
            )

        return base
    # Data pipelines are stored as lists of dicts rather than dict objects
    elif isinstance(base, list) and isinstance(overrides, dict):
        # Match pipeline operations by their type
        base_by_type = {item["type"]: item for item in base}
        for pipeline_type, params in overrides.items():
            if pipeline_type not in base_by_type:
                raise KeyError(
                    f"Unknown data pipeline operation: {pipeline_type}"
                )
            _merge_configs(
                base_by_type[pipeline_type],
                params,
                path=f"{path}{pipeline_type}.",
                override_existing=override_existing,
            )

        return base

    elif (base is None) or override_existing:
        return overrides
    else:
        if base != overrides:
            raise ValueError(
                f"Unexpected override at {path[:-1]}: '{base}' vs '{overrides}'"
            )

        return base


def load_config(
    data_config_path: str,
    model_config_path: str,
    pipeline_template_path: str,
    test_mode: bool = False,
    superpoint_mode: bool = False,
    force_rewrite: bool = False,
) -> Tuple[Config, Config]:
    data_config = OmegaConf.load(data_config_path)
    _format_all_paths(data_config)
    if superpoint_mode:
        pipeline_config_name = "superpoint_pipeline.yaml"
        model_key = "superpoint_model"
    else:
        pipeline_config_name = "model_pipeline.yaml"
        model_key = "model"
    pipeline_config = OmegaConf.load(
        os.path.join(pipeline_template_path, pipeline_config_name)
    )
    data_config = OmegaConf.merge(
        data_config,
        OmegaConf.create({
            f"{model_key}_training": pipeline_config
        })
    )
    # Resolve internal references
    data_config = OmegaConf.to_container(data_config, resolve=True)
    # Extract relevant parts for merging
    override_config = OmegaConf.create({
        "model": data_config[model_key]
    })
    override_config = OmegaConf.merge(
        override_config, data_config[f"{model_key}_training"]
    )
    override_config = OmegaConf.to_container(override_config)
    data_config = data_config["dataset"]
    data_config = Config(data_config)
    # Load the python configuration file and merge information from yaml configuration
    model_config = Config.fromfile(model_config_path).to_dict()
    config = _merge_configs(
        model_config,
        override_config,
        override_existing=True,
    )
    config = Config(config)
    _update_pipelines(config)
    _format_extra_features(config, data_config)
    # Set modes
    config.model.test_mode = test_mode
    config.test_evaluator.test_mode = test_mode
    if superpoint_mode:
        config.model.force_rewrite = force_rewrite
    
    return config, data_config