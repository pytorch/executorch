# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import collections
import unittest
from typing import Any, Callable, Literal, Optional, Tuple

import torch

from executorch import exir
from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
from executorch.backends.qualcomm.qnn_preprocess import QnnBackend

from executorch.backends.qualcomm.qnn_quantizer import QnnQuantizer
from executorch.backends.qualcomm.utils.utils import (
    capture_program,
    generate_qnn_executorch_compiler_spec,
    SoCModel,
)
from executorch.examples.portable.utils import _EDGE_COMPILE_CONFIG
from executorch.exir.backend.backend_api import to_backend
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e


def get_qdq_module(
    module: torch.nn.Module,
    inputs: Tuple[torch.Tensor],
    is_conv_per_channel: Optional[bool] = True,
    custom_quant_annotations: Tuple[Callable] = (),
) -> torch.fx.GraphModule:
    m = torch._export.capture_pre_autograd_graph(module, inputs)

    quantizer = QnnQuantizer()
    quantizer.add_custom_quant_annotations(custom_quant_annotations)
    quantizer.set_per_channel_quant(is_conv_per_channel)

    prepared = prepare_pt2e(m, quantizer)
    prepared(*inputs)
    return convert_pt2e(prepared)


def capture_graph_for_qnn(
    module: torch.nn.Module,
    inputs: Tuple[torch.Tensor],
) -> exir.ExirExportedProgram:
    return exir.capture(
        module, inputs, config=exir.CaptureConfig(enable_aot=True, _unlift=True)
    ).to_edge(_EDGE_COMPILE_CONFIG)


def save_model_and_expected_output(
    module: torch.nn.Module,
    buffer: exir.ExirExportedProgram,
    inputs: Tuple[torch.Tensor],
    model_name: Literal,
    export_pte: bool = False,
) -> None:
    if not export_pte:
        return

    input_list = ""
    for idx, inp in enumerate(inputs):
        input_name = f"input_{idx}.raw"
        inp.detach().numpy().tofile(input_name)
        input_list += input_name + " "
    input_list = input_list.strip() + "\n"
    with open("input_list.txt", "w") as file:
        file.write(input_list)

    ref_output = module(*inputs)

    if isinstance(ref_output, collections.OrderedDict):
        output_name = "expected_output"
        filename = f"{output_name}_0.raw"
        print(f"Saving expected output to {filename}")
        ref_output["out"].detach().numpy().tofile(filename)
    elif isinstance(ref_output, tuple):
        i = 0
        for output in ref_output:
            output_name = "expected_output"
            filename = f"{output_name}_{i}.raw"
            print(f"Saving expected output to {filename}")
            output.detach().numpy().tofile(filename)
            i += 1
    else:
        output_name = "expected_output"
        filename = f"{output_name}_0.raw"
        print(f"Saving expected output to {filename}")
        ref_output.detach().numpy().tofile(filename)

    filename = f"{model_name}.pte"
    print(f"Saving exported program to {filename}")
    with open(filename, "wb") as file:
        file.write(buffer)


class TestQNN(unittest.TestCase):
    def assert_outputs_equal(self, model_output, ref_output):
        # Compare the result from executor and eager mode direclty
        if isinstance(ref_output, tuple) or isinstance(ref_output, list):
            # Multiple outputs executor always returns tuple, even if there is one output
            self.assertTrue(len(ref_output) == len(model_output))
            for i in range(len(ref_output)):
                self.assertTrue(
                    torch.allclose(
                        model_output[i], ref_output[i], atol=1e-03, rtol=1e-03
                    )
                )
        else:
            # If one output, eager returns tensor while executor tuple of size 1
            self.assertTrue(
                torch.allclose(model_output[0], ref_output, atol=1e-03, rtol=1e-03)
            )

    def lower_module_and_test_output(
        self,
        module: Any,
        sample_inputs: Tuple[torch.Tensor],
        use_partitioner: bool = True,
        is_fp16: bool = False,
        soc_model: SoCModel = SoCModel.SM8550,
        debug: bool = False,
        saver: bool = False,
    ) -> exir.ExirExportedProgram:
        class WrappedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.one_module = module

            def forward(self, *args):
                return self.one_module(*args)

        compiler_specs = generate_qnn_executorch_compiler_spec(
            is_fp16=is_fp16, soc_model=soc_model, debug=debug, saver=saver
        )
        QnnPartitioner.set_compiler_spec(compiler_specs)
        if use_partitioner:
            delegated_program = capture_program(module, sample_inputs)
            delegated_program.exported_program = to_backend(
                delegated_program.exported_program, QnnPartitioner()
            )
            exec_prog = delegated_program.to_executorch()
        else:
            edge_program = capture_graph_for_qnn(WrappedModule(), sample_inputs)
            delegated_program = to_backend(
                QnnBackend.__name__,
                edge_program.exported_program,
                compiler_specs,
            )

            exported_program = capture_graph_for_qnn(delegated_program, sample_inputs)
            exec_prog = exported_program.to_executorch()

        print("Graph Module with delegate:")
        exec_prog.dump_graph_module().print_readable()

        # Assert the backend name is qnn
        self.assertEqual(
            exec_prog.program.execution_plan[0].delegates[0].id,
            QnnBackend.__name__,
        )

        return exec_prog.buffer
