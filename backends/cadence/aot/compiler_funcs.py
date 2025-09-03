# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from typing import Optional

import torch
from torch._inductor.decomposition import remove_decompositions
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, prepare_qat_pt2e
from torchao.quantization.pt2e.quantizer import Quantizer


@torch.no_grad()
def trace(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    is_qat: bool = False,
    strict: bool = False,
    ops_to_keep: Optional[list[torch._ops.OpOverload]] = None,
) -> torch.export.ExportedProgram:
    if is_qat:
        model.train()
    else:
        model.eval()

    decomp_table = torch.export.default_decompositions()
    # pyre-fixme[6]: For 1st argument expected `Dict[typing.Callable[..., typing.Any
    remove_decompositions(decomp_table, ops_to_keep)  # type: ignore[arg-type]
    program = torch.export.export(model, inputs, strict=strict).run_decompositions(
        decomp_table
    )

    return program


def prepare(
    traced_program: torch.export.ExportedProgram,
    quantizer: Quantizer,
    is_qat: bool = False,
) -> torch.fx.GraphModule:
    traced_model = traced_program.module()
    assert isinstance(traced_model, torch.fx.GraphModule)

    if is_qat:
        prepared_model = prepare_qat_pt2e(traced_model, quantizer)
    else:
        prepared_model = prepare_pt2e(traced_model, quantizer)

    return prepared_model
