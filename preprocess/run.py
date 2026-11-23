import time
import argparse

from typing import Optional
from preprocess.load_data import export_scans
from preprocess.generate_input import generate_inputs
from preprocess.tools import (
    validate_data_config,
    get_parser,
    move_all_to_test,
    generate_superpoints,
)
from util import set_seed, load_config
from mmengine.logging import MMLogger, print_log


def preprocess_data(
    args: Optional[argparse.Namespace] = None,
    logger: Optional[MMLogger] = None,
) -> None:
    preprocess_start_time = time.time()
    if args is None:
        parser = get_parser()
        args = parser.parse_args()
    if args.superpoint_mode:
        assert args.checkpoint is not None, "'--checkpoint' must be set when '--superpoint_mode' is set!"
    model_config, data_config = load_config(
        data_config_path=args.data_config,
        model_config_path=args.config,
        pipeline_template_path=args.pipeline_config,
        test_mode=True,
        superpoint_mode=True,
        force_rewrite=args.force_rewrite,
    )
    validate_data_config(data_config)
    set_seed(33, args.deterministic_alg)
    # Preliminary preprocessing from raw point clouds to npy files
    export_scans(
        data_config,
        args.unlabeled_mode,
        args.force_rewrite,
        logger=logger,
    )
    # If superpoint mode, move all files to test split to enable predicting superpoints for them
    if args.superpoint_mode:
        # The function modifies the config file in place and outputs the original split to enable restoring it
        original_data_split = move_all_to_test(data_config)
    # Generate model inputs from preprocessed npy files
    generate_inputs(
        data_config,
        logger=logger,
    )
    # In superpoint mode, generate superpoints using results of the initial preprocessing and
    # then preprocess again to include superpoint labels in the data
    if args.superpoint_mode:
        # Generate superpoints for all files using a pretrained model
        print_log("==> Loading superpoint generator", logger=logger)
        superpoint_start_time = time.time()
        generate_superpoints(args, model_config)
        superpoint_end_time = time.time()
        print_log(
            f"==> Finish generating superpoints for all splits in {(superpoint_end_time - superpoint_start_time):.3f} s!",
            logger=logger,
        )
        # Revert back to original data split
        data_config.data_split = original_data_split
        # Preprocess data again to include superpoints
        export_scans(
            data_config,
            args.unlabeled_mode,
            args.force_rewrite,
            superpoint_mode=True,
            logger=logger,
        )
        generate_inputs(
            data_config,
            superpoint_mode=True,
            logger=logger,
        )
    preprocess_end_time = time.time()
    print_log(
        f"==> Finish preprocessing '{data_config.dataset_name}' in {(preprocess_end_time - preprocess_start_time):.3f} s!",
        logger=logger,
    )


if __name__ == "__main__":
    preprocess_data()
