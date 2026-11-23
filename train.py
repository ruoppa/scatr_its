import os
import time
import argparse
import logging

from util import load_config, VRAMMonitor
from mmengine.logging import print_log, MMLogger
from mmdet3d.utils import replace_ceph_backend
from pathlib import Path
from mmengine.runner import Runner
from mmengine.config import DictAction
from mmengine.registry import RUNNERS


def _parse_args():
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--config", required=True, help="Model config file path.")
    parser.add_argument(
        "--data_config", required=True, help="Dataset config file path."
    )
    parser.add_argument(
        "--pipeline_config", default="configs/pipelines", help="Pipeline template config file path."
    )
    parser.add_argument(
        "--superpoint_mode",
        "-sp",
        action="store_true",
        help="If set, train superpoint generator backbone",
    )
    parser.add_argument(
        "--experiment_name",
        default=None,
        help="Name of training experiment (used for saving checkpoints).",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Enable automatic-mixed-precision training",
    )
    parser.add_argument(
        "--auto-scale-lr", action="store_true", help="Enable automatically scaling LR."
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        type=str,
        const="auto",
        help="If checkpoint path is specified, resume from it. If not, try to auto "
        "resume from the latest checkpoint in the work directory.",
    )
    parser.add_argument(
        "--monitor_vram",
        "-m",
        action="store_true",
        help="If set, monitor VRAM usage during training and output the peak value once the script finishes.",
    )
    parser.add_argument(
        "--ceph", action="store_true", help="Use ceph as data storage backend."
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
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():
    start_time = time.time()
    args = _parse_args()
    if args.monitor_vram:
        monitor = VRAMMonitor()
        monitor.start()
    # Load config
    config, data_config = load_config(
        data_config_path=args.data_config,
        model_config_path=args.config,
        pipeline_template_path=args.pipeline_config,
        test_mode=False,
        superpoint_mode=args.superpoint_mode,
    )

    if args.ceph:
        config = replace_ceph_backend(config)
    config.launcher = args.launcher
    if args.cfg_options is not None:
        config.merge_from_dict(args.cfg_options)
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.experiment_name is not None:
        # Update configs according to CLI args if args.work_dir is not None
        work_dir = os.path.join(
            data_config.data_root,
            "experiments",
            data_config.dataset_name,
            args.experiment_name,
        )
    else:
        # Use config filename as default work_dir if config.work_dir is None
        filename = Path(args.data_config).name
        exp_name = filename.split(".")[0]
        work_dir = os.path.join(
            data_config.data_root, "experiments", data_config.dataset_name, exp_name
        )
    config.work_dir = work_dir
    # Enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = config.optim_wrapper.type
        if optim_wrapper == "AmpOptimWrapper":
            print_log(
                "AMP training is already enabled in your config.",
                logger="current",
                level=logging.WARNING,
            )
        else:
            assert optim_wrapper == "OptimWrapper", (
                "`--amp` is only supported when the optimizer wrapper type is "
                f"`OptimWrapper` but got {optim_wrapper}."
            )
            config.optim_wrapper.type = "AmpOptimWrapper"
            config.optim_wrapper.loss_scale = "dynamic"
    # Enable automatically scaling LR
    if args.auto_scale_lr:
        if (
            "auto_scale_lr" in config
            and "enable" in config.auto_scale_lr
            and "base_batch_size" in config.auto_scale_lr
        ):
            config.auto_scale_lr.enable = True
        else:
            raise RuntimeError(
                "Can not find `auto_scale_lr`, `auto_scale_lr.enable` or"
                " `auto_scale_lr.base_batch_size` in your configuration file."
            )
    # Resume is determined in this priority: resume from > auto_resume
    if args.resume == "auto":
        config.resume = True
        config.load_from = None
    elif args.resume is not None:
        config.resume = True
        config.load_from = args.resume
    # Build the runner from config
    if "runner_type" not in config:
        # Build the default runner
        runner = Runner.from_cfg(config)
    else:
        # Build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(config)
    # Start training
    runner.train()
    end_time = time.time()
    logger = MMLogger.get_current_instance()
    print_log(
        f"==> Finished training in {(end_time - start_time):.3f} s!", logger=logger
    )
    if args.monitor_vram:
        monitor.stop()
        print_log(
            f"==> Peak VRAM usage during training (nvidia-smi): {monitor.get_peak_gib():.3f} GiB",
            logger=logger,
        )


if __name__ == "__main__":
    main()
