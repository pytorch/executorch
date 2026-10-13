# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re
from dataclasses import dataclass
from typing import Iterable

import eiq_neutron_sdk
import torch

from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from torch import memory_format
from torchao.quantization.pt2e.quantizer import Quantizer


@dataclass
class ModelInputSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype = torch.float32
    dim_order: memory_format = torch.contiguous_format


def get_default_quantizer(target_spec: NeutronTargetSpec, use_qat: bool) -> Quantizer:
    return NeutronQuantizer(target_spec, is_qat=use_qat)


def handle_kernel_selection(model_name: str = ""):
    # Collect all kernel selection files in the current working directory
    kernel_selection_files = [
        fname
        for fname in os.listdir(".")
        if re.search("_kernel_selection.*\\.c$", fname)
    ]

    if not kernel_selection_files:
        raise RuntimeError(
            "No kernel_selection files found in the current directory."
        )  # Should never happen.

    output_mask = model_name + "_kernel_selection.c"
    eiq_neutron_sdk.merge_kernel_selection_files(kernel_selection_files, output_mask)

    if not logging.root.isEnabledFor(logging.DEBUG):
        for fname in kernel_selection_files:
            os.remove(fname)
    else:
        logging.debug(
            f"Debug mode enabled, keeping intermediate kernel_selection.c files: {kernel_selection_files}"
        )


def to_model_input_spec(
    input_spec: Iterable[ModelInputSpec] | tuple[int, ...] | list[tuple[int, ...]],
) -> tuple[ModelInputSpec, ...]:
    match input_spec:
        case _ if isinstance(input_spec, Iterable) and all(
            isinstance(spec, ModelInputSpec) for spec in input_spec
        ):
            return tuple(input_spec)
        case tuple() if all(isinstance(spec, int) for spec in input_spec):
            return (ModelInputSpec(input_spec),)
        case list() if all(
            isinstance(input_shape, tuple) for input_shape in input_spec
        ):
            return tuple(ModelInputSpec(spec) for spec in input_spec)
        case _:
            raise TypeError(f"Unsupported type {type(input_spec)}")
