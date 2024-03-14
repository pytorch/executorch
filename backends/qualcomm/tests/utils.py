# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import collections
import os
import tempfile
import unittest
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np
import torch

from executorch import exir
from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
from executorch.backends.qualcomm.qnn_preprocess import QnnBackend
from executorch.backends.qualcomm.quantizer.quantizer import (
    get_default_16bit_qnn_ptq_config,
    QnnQuantizer,
)
from executorch.backends.qualcomm.serialization.qnn_compile_spec_schema import (
    QcomChipset,
)
from executorch.backends.qualcomm.utils.utils import capture_program
from executorch.examples.qualcomm.scripts.utils import SimpleADB

from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e


class TestQNN(unittest.TestCase):
    rtol: float = 0
    atol: float = 0
    host: Literal = ""
    device: Literal = ""
    build_folder: Literal = ""
    model: QcomChipset = None
    compiler_specs: List[CompileSpec] = None
    arch_table = {
        "SM8650": QcomChipset.SM8650,
        "SM8550": QcomChipset.SM8550,
        "SM8475": QcomChipset.SM8475,
        "SM8450": QcomChipset.SM8450,
    }
    error_only = False
    ip = "localhost"
    port = 8080
    executorch_root: Literal = ""
    artifact_dir: Literal = ""
    image_dataset: Literal = ""
    pretrained_weight: Literal = ""
    online_prepare: bool = False

    def _assert_outputs_equal(self, model_output, ref_output):
        self.assertTrue(len(ref_output) == len(model_output))
        for i in range(len(ref_output)):
            self.assertTrue(
                torch.allclose(
                    model_output[i], ref_output[i], atol=self.atol, rtol=self.rtol
                ),
                msg=f"ref_output:\n{ref_output[i]}\n\nmodel_output:\n{model_output[i]}",
            )

    def _save_model_and_expected_output(
        self,
        module: torch.nn.Module,
        buffer: exir.ExirExportedProgram,
        inputs: Tuple[torch.Tensor],
        dir_name: Literal,
    ) -> None:
        # Save the input data list to be executed
        input_list = ""
        for idx, _ in enumerate(inputs):
            input_name = f"input_0_{idx}.raw"
            input_list += input_name + " "
        input_list = input_list.strip() + "\n"

        ref_output = module(*inputs)

        # Save the expected output data to be verified
        ref_outputs = []
        if isinstance(ref_output, collections.OrderedDict):
            ref_outputs.append(ref_output["out"].detach())
        elif isinstance(ref_output, tuple):
            for output in ref_output:
                ref_outputs.append(output.detach())
        else:
            ref_outputs.append(ref_output.detach())

        pte_fname = f"{dir_name}/qnn_executorch_test.pte"
        with open(pte_fname, "wb") as file:
            file.write(buffer)

        return input_list, ref_outputs, pte_fname

    def lower_module_and_test_output(
        self,
        module: torch.nn.Module,
        sample_inputs: Tuple[torch.Tensor],
        expected_partitions: int = 1,
        assert_output_equal: bool = True,
        skip_node_id_set: set = None,
        skip_node_op_set: set = None,
    ):
        qnn_partitioner = QnnPartitioner(
            self.compiler_specs, skip_node_id_set, skip_node_op_set
        )
        delegated_program = capture_program(module, sample_inputs)
        delegated_program.exported_program = to_backend(
            delegated_program.exported_program, qnn_partitioner
        )
        exec_prog = delegated_program.to_executorch()

        # Assert the backend name is qnn
        self.assertEqual(
            len(exec_prog.program.execution_plan[0].delegates),
            expected_partitions,
        )
        for i in range(expected_partitions):
            self.assertEqual(
                exec_prog.program.execution_plan[0].delegates[i].id,
                QnnBackend.__name__,
            )

        # Check numerics
        if assert_output_equal:
            with tempfile.TemporaryDirectory() as tmp_dir:
                (
                    input_list,
                    ref_outputs,
                    pte_fname,
                ) = self._save_model_and_expected_output(
                    module,
                    exec_prog.buffer,
                    sample_inputs,
                    tmp_dir,
                )

                device_output_dir = f"{tmp_dir}/outputs"
                device_outputs = []

                def post_process():
                    for i, f in enumerate(os.listdir(device_output_dir)):
                        filename = os.path.join(device_output_dir, f)
                        output = np.fromfile(
                            filename, dtype=ref_outputs[i].numpy().dtype
                        )
                        output = torch.from_numpy(output).reshape(ref_outputs[i].shape)
                        device_outputs.append(output)

                adb = SimpleADB(
                    qnn_sdk=os.getenv("QNN_SDK_ROOT"),
                    artifact_path=self.build_folder,
                    pte_path=pte_fname,
                    workspace="/data/local/tmp/qnn_executorch_test",
                    device_id=self.device,
                    host_id=self.host,
                    soc_model=self.model,
                    error_only=self.error_only,
                )
                adb.push(inputs=[sample_inputs], input_list=input_list)
                adb.execute()
                adb.pull(output_path=tmp_dir, callback=post_process)
                self._assert_outputs_equal(device_outputs, ref_outputs)

    def get_qdq_module(
        self,
        module: torch.nn.Module,
        inputs: Tuple[torch.Tensor],
        is_conv_per_channel: Optional[bool] = True,
        custom_quant_annotations: Tuple[Callable] = (),
        use_16bit_quant: Optional[bool] = False,
    ) -> torch.fx.GraphModule:
        m = torch._export.capture_pre_autograd_graph(module, inputs)

        quantizer = QnnQuantizer()
        quantizer.add_custom_quant_annotations(custom_quant_annotations)
        quantizer.set_per_channel_quant(is_conv_per_channel)

        if use_16bit_quant:
            quantizer.add_16bit_quant_ops(quantizer.SUPPORTED_OPS)
            quantizer.set_bit16_op_quant_config(get_default_16bit_qnn_ptq_config())

        prepared = prepare_pt2e(m, quantizer)
        prepared(*inputs)
        quantized_module = convert_pt2e(prepared)
        nodes = {node.target for node in quantized_module.graph.nodes}
        q_and_dq = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.quantize_per_channel.default,
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
        }
        self.assertTrue(nodes.intersection(q_and_dq))
        return quantized_module
