import os
import time
import argparse

from util import load_config, cast_all, set_seed, VRAMMonitor
from mmengine.runner import Runner
from mmengine.config import DictAction
from mmengine.logging import MMLogger, print_log
from mmengine.registry import RUNNERS


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Perform inference for a trained model."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Model config file path.",
    )
    parser.add_argument(
        "--data_config",
        required=True,
        help="Dataset config file path.",
    )
    parser.add_argument(
        "--pipeline_config",
        default="configs/pipelines",
        help="Pipeline template config file path.",
    )
    parser.add_argument(
        "--superpoint_mode",
        "-sp",
        action="store_true",
        help="If set, generate superpoints",
    )
    parser.add_argument(
        "--force_rewrite",
        "-f",
        action="store_true",
        help="If set, force rewrite even if file has already been preprocessed before. Ignored if '-sp' is not set.",
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file path.")
    parser.add_argument(
        "--unlabeled_mode",
        "-u",
        action="store_true",
        help="Set if data has no ground truth labels. Skips accuracy computation.",
    )
    parser.add_argument(
        "--cast_to_original",
        "-c",
        action="store_true",
        help="Cast predictions back to original input point clouds.",
    )
    parser.add_argument(
        "--prec",
        type=int,
        default=3,
        help="Decimal precision used when casting predictions.",
    )
    parser.add_argument(
        "--force_nn",
        action="store_true",
        help="Force using nearest neighbor search for casting. Can be extremely slow/infeasible for large point clouds.",
    )
    parser.add_argument(
        "--monitor_vram",
        "-m",
        action="store_true",
        help="If set, monitor VRAM usage during training and output the peak value once the script finishes.",
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
    parser.add_argument(
        "--deterministic_alg",
        "-d",
        action="store_true",
        help="set 'torch.use_deterministic_algorithms' to True for reproducibility. "
        "Note that this will make the model slightly slower and that not all randomness "
        "can be removed even if this argument is set.",
    )
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/test.py` instead
    # of `--local_rank`.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    args.prec = 10 ** (-args.prec)
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():
    start_time = time.time()
    args = _parse_args()
    if args.monitor_vram:
        monitor = VRAMMonitor()
        monitor.start()
    set_seed(33, args.deterministic_alg)
    # Load config
    config, data_config = load_config(
        data_config_path=args.data_config,
        model_config_path=args.config,
        pipeline_template_path=args.pipeline_config,
        test_mode=True,
        superpoint_mode=args.superpoint_mode,
        force_rewrite=args.force_rewrite,
    )
    # Set command line options
    config.launcher = args.launcher
    if args.cfg_options is not None:
        config.merge_from_dict(args.cfg_options)
    if args.superpoint_mode:
        config.work_dir = config.model.test_cfg.superpoint_data_dir
    else:
        config.work_dir = config.model.test_cfg.output_data_dir
    config.load_from = args.checkpoint
    # Set flag to skip accuracy computation for unlabeled data
    config.test_evaluator.unlabeled_mode = args.unlabeled_mode
    if "runner_type" not in config:
        # Build the default runner
        runner = Runner.from_cfg(config)
    else:
        # Build custom runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(config)
    # Run inference once
    runner.test()
    if args.cast_to_original:
        # Cast predictions to original input
        logger = MMLogger.get_current_instance()
        cast_all(
            config=None,
            data_config=data_config,
            unlabeled_mode=args.unlabeled_mode,
            force_nn=args.force_nn,
            prec=args.prec,
            logger=logger,
        )
    end_time = time.time()
    logger = MMLogger.get_current_instance()
    print_log(
        f"==> Finished inference in {(end_time - start_time):.3f} s!", logger=logger
    )
    if args.monitor_vram:
        monitor.stop()
        print_log(
            f"==> Peak VRAM usage during inference (nvidia-smi): {monitor.get_peak_gib():.3f} GiB",
            logger=logger,
        )


if __name__ == "__main__":
    main()
