# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManagerAdaptor

from executorch.backends.qualcomm.utils.utils import (
    generate_qnn_executorch_option,
    update_spill_fill_size,
)


def preprocess_binary(ctx_bin, compiler_specs):
    qnn_mgr = PyQnnManagerAdaptor.QnnManager(
        generate_qnn_executorch_option(compiler_specs),
    )
    return bytes(qnn_mgr.MakeBinaryInfo(ctx_bin))


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
        ctx_bin = preprocess_binary(f.read(), compiler_specs)
        qnn_mgr = PyQnnManagerAdaptor.QnnManager(
            generate_qnn_executorch_option(compiler_specs), ctx_bin
        )
        assert qnn_mgr.Init().value == 0, "failed to load context binary"
        graph_name = qnn_mgr.GetGraphNames()[0]
        qnn_mgr.AllocateTensor(graph_name)
        if get_input:
            encoding_input = {"scale": [], "offset": []}
            for i in range(num_input):
                inputs = qnn_mgr.GetGraphInputs(graph_name)[i]
                encoding = inputs.GetEncodings()
                encoding_input["scale"].append(encoding.data["scale"].item())
                encoding_input["offset"].append(encoding.data["offset"].item())
            encoding_list.append(encoding_input)
        if get_output:
            encoding_output = {"scale": [], "offset": []}
            for i in range(num_output):
                outputs = qnn_mgr.GetGraphOutputs(graph_name)[i]
                encoding = outputs.GetEncodings()
                encoding_output["scale"].append(encoding.data["scale"].item())
                encoding_output["offset"].append(encoding.data["offset"].item())
            encoding_list.append(encoding_output)
        qnn_mgr.Destroy()
    return encoding_list


def gen_pte_from_ctx_bin(artifact, pte_names, bundle_programs, backend_config):
    edge_prog_mgrs = [prog["edge_program_manager"] for prog in bundle_programs]
    # Setup spill-fill buffer for relieving runtime memory usage
    update_spill_fill_size(
        [
            prog_mgr._edge_programs[list(prog_mgr.methods)[0]]
            for prog_mgr in edge_prog_mgrs
        ]
    )
    # Export pte files
    pte_files = []
    for pte_name in pte_names:
        print(f"{pte_name} generating...")
        pte_files.append(f"{artifact}/{pte_name}.pte")
        with open(pte_files[-1], "wb") as f:
            edge_prog_mgrs[0].to_executorch(config=backend_config).write_to_file(f)
        # GC for reducing host memory consuming
        bundle_programs.pop(0)
        edge_prog_mgrs.pop(0)
        gc.collect()

    return pte_files
