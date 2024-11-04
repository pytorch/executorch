# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.examples.models.llama3_2_vision.preprocess.model import (
    CLIPImageTransformModel,
    PreprocessConfig,
)
from executorch.exir import EdgeCompileConfig, to_edge


def main():
    # Eager model.
    model = CLIPImageTransformModel(PreprocessConfig())

    # Export.
    ep = torch.export.export(
        model.get_eager_model(),
        model.get_example_inputs(),
        dynamic_shapes=model.get_dynamic_shapes(),
        strict=False,
    )

    # AOTInductor. Note: export AOTI before ExecuTorch, as
    # ExecuTorch will modify the ExportedProgram.
    torch._inductor.aot_compile(
        ep.module(),
        model.get_example_inputs(),
        options={"aot_inductor.output_path": "preprocess_aoti.so"},
    )

    # Executorch.
    edge_program = to_edge(
        ep, compile_config=EdgeCompileConfig(_check_ir_validity=False)
    )
    et_program = edge_program.to_executorch()
    with open("preprocess_et.pte", "wb") as file:
        et_program.write_to_file(file)


if __name__ == "__main__":
    main()
