# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

try:
    from pathlib import Path

    quantized_ops_aot_lib = list(Path(__file__).parent.resolve().glob("**/libquantized_ops_aot_lib.*"))
    # embedding_ops_aot_lib = list(Path(__file__).parent.resolve().glob("**/libembedding_ops_aot_lib.*"))
    del Path
    assert len(quantized_ops_aot_lib) == 1, f"Expected 1 library but got {len(quantized_ops_aot_lib)}"
    # assert len(embedding_ops_aot_lib) == 1, f"Expected 1 library but got {len(embedding_ops_aot_lib)}"
    import torch as _torch

    _torch.ops.load_library(quantized_ops_aot_lib[0])
    op = torch.ops.quantized.add_out
    assert op is not None
    # _torch.ops.load_library(embedding_ops_aot_lib[0])

    del _torch
except:
    import logging

    logging.info("libquantized_ops_aot_lib and/or libembedding_ops_aot_lib is not loaded")
    del logging
