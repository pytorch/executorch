# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
"""Shared TOSA lowering implementation for the Arm backend.

Only symbols listed in the Arm public API manifest are public API. Other
modules under ``executorch.backends.arm.tosa`` are implementation details and
may change without notice.

"""

from .specification import TosaSpecification

__all__ = ["TosaSpecification"]
