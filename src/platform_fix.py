"""
macOS OpenMP workaround — import before torch.

On Apple Silicon, conda PyTorch and other libs (numpy, scipy) can each link
libomp; loading both aborts with OMP Error #15. Setting this env var is the
standard workaround when you cannot deduplicate the runtime at link time.
"""

from __future__ import annotations

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
