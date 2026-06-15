"""
DataLoader helpers — macOS worker cleanup and platform-safe defaults.

PyTorch DataLoaders with num_workers > 0 spawn ``torch_shm_manager`` processes
for shared-memory IPC.  On macOS these frequently fail to terminate after the
script exits, causing memory pressure until manually killed with
``pkill -f torch_shm_manager``.

Fix: use num_workers=0 on Darwin (MPS/CPU Mac) and explicitly shut down any
worker pools before the process exits.
"""

from __future__ import annotations

import gc
import sys
from typing import Optional

from torch.utils.data import DataLoader

_warned = False


def resolve_num_workers(num_workers: int) -> int:
    """
    Return a safe worker count for the current platform.

    On macOS (darwin) always returns 0 — multiprocessing DataLoader workers
    are unreliable and spawn lingering torch_shm_manager processes.
    On Linux/CUDA the requested value is used unchanged.
    """
    global _warned
    if sys.platform == "darwin" and num_workers > 0:
        if not _warned:
            print(
                "  [info] num_workers forced to 0 on macOS "
                "(avoids lingering torch_shm_manager processes)"
            )
            _warned = True
        return 0
    return num_workers


def shutdown_dataloaders(*loaders: Optional[DataLoader]) -> None:
    """Explicitly terminate DataLoader worker processes and release resources."""
    for loader in loaders:
        if loader is None:
            continue
        # Shut down an active iterator's worker pool if one exists
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    pass
            loader._iterator = None
        # Drop reference so workers can be reaped
        del loader

    gc.collect()
