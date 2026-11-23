import os
import time
import mmengine

from mmengine.config import Config
from mmengine.logging import MMLogger, print_log
from typing import Optional
from preprocess.tools import ForestInputDataGenerator


def generate_inputs(
    data_config: Config,
    superpoint_mode: bool = False,
    logger: Optional[MMLogger] = None,
) -> None:
    """Generate input bin files for points and annotations, and info pkl files

    Args:
        data_config (Config): data configuration
        instance_class_names (Dict[int, str]): dictionary with (instance id, class name) pairs
    """
    in_path = data_config.preprocess.preprocessed_data_dir
    out_path = data_config.preprocess.input_data_dir
    # Number of dimensions in data = 3 + number of extra features
    data_dim = 3
    if data_config.preprocess.feat_to_load is not None:
        data_dim += len(data_config.preprocess.feat_to_load)
    assert os.path.exists(in_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # Generate inputs for all splits in the dataset
    for data_split in data_config.data_split.keys():
        split_start_time = time.time()
        print_log(f"==> Generating inputs for '{data_split}' split", logger=logger)
        filename = os.path.join(out_path, f"{data_split}_infos.pkl")
        generator = ForestInputDataGenerator(
            in_path,
            out_path,
            data_split,
            data_config.data_split,
            data_config.general.instance_class_names,
            data_dim,
            superpoint_mode,
        )
        infos, ignored_classes = generator.process()
        # Generate dataset metainfo
        name_to_instance_label = dict()
        for key, value in data_config.general.instance_class_names.items():
            if key not in ignored_classes:
                name_to_instance_label[value] = key
            else:
                name_to_instance_label[value] = -1
        metainfo = {
            "categories": name_to_instance_label,
            "dataset": data_config.dataset_name,
            "info_version": "1.1",
        }
        data_info = dict(metainfo=metainfo, data_list=infos)
        mmengine.dump(data_info, filename, "pkl")
        split_end_time = time.time()
        status = f"==> Finish generating inputs for '{data_split}' split in {(split_end_time - split_start_time):.3f} s!"
        if len(ignored_classes) > 0:
            status += f"Ignored classes: {ignored_classes}"
        print_log(status, logger=logger)
