#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import warnings

warnings.warn(
    "The executorch.backends.apple.mps package is deprecated and will be "
    "removed in ExecuTorch 1.5. Use executorch.backends.apple.coreml instead.",
    FutureWarning,
    stacklevel=2,
)

from .mps_preprocess import MPSBackend

__all__ = [
    MPSBackend,
]
