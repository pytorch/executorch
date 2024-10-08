# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManagerAdaptor

from executorch.backends.qualcomm.utils.utils import (
    canonicalize_program,
    generate_qnn_executorch_option,
)
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass


def get_encoding(
    path_to_shard: str,
    compiler_specs: str,
    get_input: bool,
    get_output: bool,
    num_input: int,
    num_output: int,
):
    encoding_list = []
    with open(path_to_shard, "rb") as f:
        ctx_bin = f.read()
        qnn_mgr = PyQnnManagerAdaptor.QnnManager(
            generate_qnn_executorch_option(compiler_specs), ctx_bin
        )
        assert qnn_mgr.Init().value == 0, "failed to load context binary"
        qnn_mgr.AllocateTensor()
        if get_input:
            encoding_input = {"scale": [], "offset": []}
            for i in range(num_input):
                inputs = qnn_mgr.GetGraphInputs()[i]
                encoding = inputs.GetEncodings()
                encoding_input["scale"].append(encoding.data["scale"].item())
                encoding_input["offset"].append(encoding.data["offset"].item())
            encoding_list.append(encoding_input)
        if get_output:
            encoding_output = {"scale": [], "offset": []}
            for i in range(num_output):
                outputs = qnn_mgr.GetGraphOutputs()[i]
                encoding = outputs.GetEncodings()
                encoding_output["scale"].append(encoding.data["scale"].item())
                encoding_output["offset"].append(encoding.data["offset"].item())
            encoding_list.append(encoding_output)
        qnn_mgr.Destroy()
    return encoding_list


def gen_pte_from_ctx_bin(
    artifact, pte_names, compiler_specs, bundle_programs, custom_spill_fill=None
):

    # Lower with QnnBackend
    lowered_modules = [
        to_backend("QnnBackend", prog["edge_program"], compiler_specs)
        for prog in bundle_programs
    ]
    # Setup spill-fill buffer for relieving runtime memory usage
    canonicalize_program(lowered_modules, custom_buffer_size=custom_spill_fill)
    # export pte files
    pte_files = []
    for pte_name in pte_names:
        print(f"{pte_name} generating...")
        memory_planning_pass = MemoryPlanningPass(
            alloc_graph_input=False,
            alloc_graph_output=False,
        )
        pte_files.append(f"{artifact}/{pte_name}.pte")
        with open(pte_files[-1], "wb") as file:
            file.write(
                lowered_modules[0].buffer(
                    extract_delegate_segments=True, memory_planning=memory_planning_pass
                )
            )
        # GC for reducing host memory consuming
        bundle_programs.pop(0)
        lowered_modules.pop(0)
        gc.collect()

    return pte_files
