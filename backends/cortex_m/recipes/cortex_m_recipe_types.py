# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from executorch.export import RecipeType


CORTEX_M_BACKEND: str = "cortex_m"


class CortexMRecipeType(RecipeType):
    """Cortex-M recipe types.

    Cortex-M lowers by rewriting edge operators into ``cortex_m::`` CMSIS-NN
    kernels rather than by delegating a subgraph, so a recipe carries no
    partitioner.

    CMSIS-NN needs NHWC, and the quantizer rejects a convolution whose operands
    are not channels_last, so trace from channels_last inputs::

        inputs = (torch.randn(1, 3, 96, 96).to(memory_format=torch.channels_last),)
        session = export(model=model, example_inputs=[inputs], export_recipe=recipe)

    Only the first entry of ``example_inputs`` is traced, and the layout it
    carries is what the quantizer annotates against. Get it wrong and the
    convolutions stay on the portable float kernels, which the runner's
    operator list then has to cover.

    Accepted kwargs:
        target (str): ``cortex-m<variant>`` CPU to compile for. Defaults to
            ``"cortex-m55"``.
        isa (cmsis_nn.Backend): Override the CMSIS-NN backend ``target`` would
            imply, for cores whose ISA extensions are optional (an M55 built
            without MVE).
    """

    INT8 = "cortex_m_int8"

    @classmethod
    def get_backend_name(cls) -> str:
        return CORTEX_M_BACKEND
