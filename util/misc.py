import os
import time
import random
import torch
import subprocess
import threading

import numpy as np

def set_seed(seed: int, deterministic_alg: bool):
    """Set seed for all randomness (if possible)

    Args:
        seed (int): seed for randomness.
        deterministic_alg (bool): if True, set torch.use_deterministic_algorithms to True.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    if deterministic_alg:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class VRAMMonitor:
    def __init__(self, interval=0.05):
        """Class for monitoring VRAM usage during training/inference. 
        Internally the class uses the nvidia-smi command line tool to monitor VRAM.
        Keeps track of the peak VRAM usage between start() and stop() calls.

        Args:
            interval (float, optional): Interval for calling nvidia-smi (in seconds). Defaults to 0.05.
        """
        self.interval = interval
        self.max_mem = 0
        self._running = False
        self.pid = os.getpid()

    def _monitor(self) -> None:
        while self._running:
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=0.5,
                )

                output = result.stdout.decode("utf-8")

                for line in output.strip().split("\n"):
                    if not line:
                        continue
                    pid_str, mem_str = line.split(",")
                    pid = int(pid_str.strip())
                    mem = int(mem_str.strip())

                    if pid == self.pid:
                        self.max_mem = max(self.max_mem, mem)

            except subprocess.TimeoutExpired:
                continue
            except Exception:
                pass

            time.sleep(self.interval)

    def start(self) -> None:
        self._running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._running = False
        self.thread.join(timeout=1.0)

    def get_peak_gib(self) -> float:
        """Get peak VRAM usage (in GiB) between the start() and stop() calls.

        Returns:
            float: peak VRAM usage in GiB.
        """
        return self.max_mem / 1024