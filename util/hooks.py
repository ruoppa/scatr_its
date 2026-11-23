import os
import torch

from typing import Optional, List, Union
from mmengine.runner import Runner
from mmengine.registry import HOOKS
from mmengine.hooks import CheckpointHook, RuntimeInfoHook
from mmengine.dist import is_main_process


@HOOKS.register_module()
class CheckpointHook_(CheckpointHook):
    def __init__(self, *args, keep_every_n_epochs: Optional[int] = None, **kwargs):
        """Same as generic CheckpointHook with two key differences:

        * Enable keeping every n epochs. By default only the x latest checkpoints are kept. This functionality
        enables also preserving every nth checkpoint
        * Fix checkpoints by default, since spconv weights are saved in the wrong order

        Args:
            keep_every_n_epochs (Optional[int], optional): If set, preserve every nth checkpoint at all times. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.keep_every_n_epochs = keep_every_n_epochs

    def _save_checkpoint(self, runner: Runner) -> None:
        if self.by_epoch:
            ckpt_step = runner.epoch + 1
        else:
            ckpt_step = runner.iter + 1
        ckpt_filename = self.filename_tmpl.format(ckpt_step)

        # Save checkpoint
        runner.save_checkpoint(
            self.out_dir,
            ckpt_filename,
            self.file_client_args,
            save_optimizer=self.save_optimizer,
            save_param_scheduler=self.save_param_scheduler,
            by_epoch=self.by_epoch,
            backend_args=self.backend_args,
            **self.args,
        )
        # Fix bugged weights in spconv layers
        self._fix_spconv_checkpoint(os.path.join(self.out_dir, ckpt_filename))

        if not is_main_process():
            return

        # update last_ckpt
        runner.message_hub.update_info(
            "last_ckpt", self.file_backend.join_path(self.out_dir, ckpt_filename)
        )

        if self.max_keep_ckpts > 0:
            redundant_ckpts = []
            if self.by_epoch:
                current_ckpt = runner.epoch + 1
            else:
                current_ckpt = runner.iter + 1

            # Compute redundant checkpoints
            redundant_ckpts = range(
                current_ckpt - self.max_keep_ckpts * self.interval, 0, -self.interval
            )

            for _step in redundant_ckpts:
                if self.keep_every_n_epochs is not None:
                    # keep every nth checkpoint
                    if _step % self.keep_every_n_epochs == 0:
                        continue
                ckpt_path = self.file_backend.join_path(
                    self.out_dir, self.filename_tmpl.format(_step)
                )
                if self.file_backend.isfile(ckpt_path):
                    self.file_backend.remove(ckpt_path)
                elif self.file_backend.isdir(ckpt_path):
                    self.file_backend.rmtree(ckpt_path)
                else:
                    break

        save_file = os.path.join(runner.work_dir, "last_checkpoint")
        filepath = self.file_backend.join_path(self.out_dir, ckpt_filename)
        with open(save_file, "w") as f:
            f.write(filepath)

    @staticmethod
    def _fix_spconv_checkpoint(
        ckpt_path: str,
        weight_key: str = "state_dict",
        spconv_modules: List[str] = ["unet", "input_conv"],
    ) -> None:
        """Fix checkpoint for spconv layers (currently bugged)

        Args:
            ckpt_path (str): Path to checkpoint. Fixed checkpoint overwrites this.
            weight_key (str, optional): Key corresponding to model weights in checkpoit. Defaults to "state_dict".
            spconv_modules (List[str], optional): Names of modules that use spconv. Defaults to ["unet", "input_conv"].
        """
        checkpoint = torch.load(ckpt_path)
        # Iterate over all layers in the checkpoint and permute weights in spconv layers
        for layer in checkpoint[weight_key]:
            if (
                any([layer.startswith(module) for module in spconv_modules])
                and layer.endswith("weight")
                and (len(checkpoint[weight_key][layer].shape) == 5)
            ):
                checkpoint[weight_key][layer] = checkpoint[weight_key][layer].permute(
                    1, 2, 3, 4, 0
                )
        # Save fixed checkpoint
        torch.save(checkpoint, ckpt_path)


@HOOKS.register_module()
class RuntimeInfoHook_(RuntimeInfoHook):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def after_val_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: Optional[Union[dict, tuple, list]] = None,
                         outputs: Optional[dict] = None) -> None:
        """Update ``log_vars`` in model outputs every iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs (dict, optional): Outputs from model. Defaults to None.
        """
        if outputs is not None:
            for key, value in outputs.items():
                runner.message_hub.update_scalar(f"val/{key}", value)