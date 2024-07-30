# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge
from preprocess import PreprocessModel


def main():
    preprocess_model = PreprocessModel()
    preprocess = preprocess_model.get_eager_model()

    ep = torch.export.export(
        preprocess,
        preprocess_model.get_example_inputs(),
        dynamic_shapes=preprocess_model.get_dynamic_shapes(),
        strict=False,
    )

    edge_program = to_edge(
        ep, compile_config=EdgeCompileConfig(_check_ir_validity=False)
    )

    et_program = edge_program.to_executorch(ExecutorchBackendConfig())
    with open("preprocess.pte", "wb") as file:
        file.write(et_program.buffer)


if __name__ == "__main__":
    main()
