import time
import argparse

from typing import List
from util import load_config
from mmengine.config import Config, DictAction
from mmdet3d.registry import METRICS


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate accuracy of model predictions"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Model config file path"
    )
    parser.add_argument(
        "--data_config",
        required=True,
        help="Dataset config file path"
    )
    parser.add_argument(
        "--pipeline_config", default="configs/pipelines", help="Pipeline template config file path."
    )
    parser.add_argument(
        "--superpoint_mode",
        "-sp",
        action="store_true",
        help="If set, compute oracle accuracy of generated superpoints",
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
    args = parser.parse_args()

    return args


def _construct_filename_list(data_config: Config) -> List:
    filename_list = []
    for filename in data_config.data_split.test:
        filename_list.append((None, {"filename": filename}))

    return filename_list


def main():
    start_time = time.time()
    args = _parse_args()
    # Load config
    config, data_config = load_config(
        data_config_path=args.data_config,
        model_config_path=args.config,
        pipeline_template_path=args.pipeline_config,
        test_mode=True,
        superpoint_mode=args.superpoint_mode,
    )

    if args.cfg_options is not None:
        config.merge_from_dict(args.cfg_options)
    # Build metrics evaluator
    forest_metrics = METRICS.build(config.test_evaluator)
    # Construct a list of test file names that follows the format expected by the evaluator
    filename_list = _construct_filename_list(data_config)
    # Compute accuracy
    forest_metrics.compute_metrics(results=filename_list)
    end_time = time.time()
    print(f"==> Finished computing acuracy in {(end_time - start_time):.3f} s!")


if __name__ == "__main__":
    main()
