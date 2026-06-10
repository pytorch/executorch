# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Configuration for PropagateDevicePass.

This is intentionally kept in a lightweight module (no heavy imports such as
the et_copy op registry) so that ``ExecutorchBackendConfig`` -- which is
imported throughout the codebase -- can reference ``PropagateDeviceConfig``
without pulling in the device-copy op registration as an import-time side
effect.
"""

from dataclasses import dataclass
from typing import Dict, Union

from torch.fx._compatibility import compatibility


@compatibility(is_backward_compatible=False)
@dataclass
class PropagateDeviceConfig:
    # When True, method-level input tensors that feed directly into a device
    # delegate are NOT wrapped with _h2d_copy. The user must provide tensors
    # already on the target device. Useful for pipelines where inputs are
    # pre-staged on GPU.
    # A dict can be used to set per-method values, keyed by method name.
    skip_h2d_for_method_inputs: Union[bool, Dict[str, bool]] = False

    # When True, device delegate outputs that are directly method outputs
    # are NOT wrapped with _d2h_copy. The method outputs stay on device.
    # Useful for cross-method GPU pipelines where the next method consumes
    # GPU tensors directly.
    # A dict can be used to set per-method values, keyed by method name.
    skip_d2h_for_method_outputs: Union[bool, Dict[str, bool]] = False

    def __hash__(self) -> int:
        return hash(
            (
                str(self.skip_h2d_for_method_inputs),
                str(self.skip_d2h_for_method_outputs),
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PropagateDeviceConfig):
            return False
        return (
            self.skip_h2d_for_method_inputs == other.skip_h2d_for_method_inputs
            and self.skip_d2h_for_method_outputs == other.skip_d2h_for_method_outputs
        )
