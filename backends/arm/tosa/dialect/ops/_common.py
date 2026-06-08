# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.arm.tosa.dialect.lib import TosaValueError

_VALID_NAN_MODES = {"PROPAGATE", "IGNORE"}


def validate_nan_mode(nan_mode: str, op: str) -> None:
    if nan_mode not in _VALID_NAN_MODES:
        raise TosaValueError(
            f"Unsupported nan_mode {nan_mode}. Expected one of {_VALID_NAN_MODES}",
            op=op,
        )
