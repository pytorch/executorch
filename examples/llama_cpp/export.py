# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import argparse
import logging

from ..models import MODEL_NAME_TO_MODEL
from ..models.model_factory import EagerModelFactory
from ..portable.utils import export_to_edge, save_pte_program
from .permute_mm_fusion_pass import PermuteMMFusionPass
from torch._export import capture_pre_autograd_graph

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


if __name__ == "__main__":

    model, example_inputs = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL["llama2"]
    )
    m = model.eval()
    # pre-autograd export. eventually this will become torch.export
    m = capture_pre_autograd_graph(m, example_inputs)

    edge_ir = export_to_edge(m, example_inputs).transform([PermuteMMFusionPass(_fix_node_meta_val=True)])
    print(f"Exported graph:\n{edge_ir.exported_program().graph}")

    prog = edge_ir.to_executorch()

    save_pte_program(prog.buffer, "llama2_fused")
