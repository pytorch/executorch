# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from executorch import exir
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
from executorch.exir.tests.common import register_additional_test_aten_ops
from torch.ao.quantization import (  # @manual
    float_qparams_weight_only_qconfig,
    get_default_qconfig_mapping,
)
from torch.ao.quantization.backend_config.executorch import (
    get_executorch_backend_config,
)

from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.quantize_fx import (
    _convert_to_reference_decomposed_fx,
    prepare_fx,
)
from torch.export import export
from torch.nn import functional as F

from torch.testing import FileCheck


class TestQuantFusionPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        register_additional_test_aten_ops()

    def test_add(self) -> None:
        class M(torch.nn.Module):
            def forward(self, x, y):
                # edge case, doesn't work yet, but we can add a fusion
                # pattern to enable it if needed
                # return x + x
                return x + y

        example_inputs = (torch.randn(1, 5), torch.randn(1, 5))
        m = M().eval()
        # TODO: define qconfig_mapping specifically for executorch
        qconfig_mapping = get_default_qconfig_mapping("qnnpack")
        m = prepare_fx(
            m,
            qconfig_mapping,
            example_inputs,
            backend_config=get_executorch_backend_config(),
        )
        m = _convert_to_reference_decomposed_fx(m)
        config = EdgeCompileConfig(_check_ir_validity=False)
        m = to_edge(export(m, example_inputs), compile_config=config)
        # QuantFusionPass should be part of to_executorch() config, separating it out so that we can check the graph.
        m = m.transform([QuantFusionPass(_fix_node_meta_val=True)])
        # check that we are using functional variant of q/dq/add
        FileCheck().check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default"
        ).check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_add_default"
        ).check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default"
        ).run(
            m.exported_program().graph_module.code
        )
        m = m.to_executorch()
        # check that we are using out variant of q/dq/add
        FileCheck().check("torch.ops.quantized_decomposed.add.out").run(
            m.exported_program().graph_module.code
        )

    def test_reshape(self) -> None:
        class M(torch.nn.Module):
            def forward(self, x, y):
                x = x + y
                x = x.reshape(1, x.numel())
                return x

        example_inputs = (torch.randn(3, 5), torch.randn(3, 5))
        m = M().eval()
        # TODO: define qconfig_mapping specifically for executorch
        qconfig_mapping = get_default_qconfig_mapping("qnnpack")
        m = prepare_fx(
            m,
            qconfig_mapping,
            example_inputs,
            backend_config=get_executorch_backend_config(),
        )
        m(*example_inputs)
        m = _convert_to_reference_decomposed_fx(m)
        config = EdgeCompileConfig(_check_ir_validity=False)
        m = to_edge(export(m, example_inputs), compile_config=config)
        # QuantFusionPass should be part of to_executorch() config, separating it out so that we can check the graph.
        m = m.transform([QuantFusionPass(_fix_node_meta_val=True)])
        # check that we are using functional variant of q/dq/add/reshape
        # make sure we only have two quant and one dequant since the q/dq around reshape
        # should be fused
        FileCheck().check_count(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default",
            2,
            exactly=True,
        ).check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_add_default"
        ).check(
            "executorch_exir_dialects_edge__ops_aten_view_copy_default"
        ).check_count(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default",
            1,
            exactly=True,
        ).run(
            m.exported_program().graph_module.code
        )

        m = m.to_executorch(exir.ExecutorchBackendConfig(remove_view_copy=False))
        # check that we are using out variant of q/dq/add
        FileCheck().check("torch.ops.quantized_decomposed.add.out").check(
            "torch.ops.aten.view_copy.out"
        ).run(m.exported_program().graph_module.code)

    def test_slice(self) -> None:
        """We don't proactively quantize slice today, but we'll fuse the dq-slice-q

        pattern into a int8 slice operator, we can revist this later to
        see if proactively quantize slice is needed or not
        """

        class M(torch.nn.Module):
            def forward(self, x, y):
                x = x + y
                x = x[1:]
                y = y[1:]
                x = x + y
                return x

        example_inputs = (torch.randn(3, 5), torch.randn(3, 5))
        m = M().eval()
        # TODO: define qconfig_mapping specifically for executorch
        qconfig_mapping = get_default_qconfig_mapping("qnnpack")
        m = prepare_fx(
            m,
            qconfig_mapping,
            example_inputs,
            backend_config=get_executorch_backend_config(),
        )
        m = _convert_to_reference_decomposed_fx(m)
        config = EdgeCompileConfig(_check_ir_validity=False)
        m = to_edge(export(m, example_inputs), compile_config=config)
        # QuantFusionPass should be part of to_executorch() config, separating it out so that we can check the graph.
        m = m.transform([QuantFusionPass(_fix_node_meta_val=True)])
        # check that we are using functional variant of q/dq/add/slice
        # make sure we only have one quant and one dequant since the q/dq around slice
        # should be fused
        FileCheck().check_count(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default",
            2,
            exactly=True,
        ).check("executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor").check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default"
        ).check(
            "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor"
        ).check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_add_default"
        ).check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default"
        ).run(
            m.exported_program().graph_module.code
        )

        m = m.to_executorch()
        # check that we are using out variant of add and slice_copy
        FileCheck().check("torch.ops.quantized_decomposed.add.out").check(
            "torch.ops.aten.slice_copy.Tensor_out"
        ).run(m.exported_program().graph_module.code)

    def test_cat(self) -> None:
        class M(torch.nn.Module):
            def forward(self, x, y):
                x = torch.cat([x, x], dim=0)
                return x

        example_inputs = (torch.randn(3, 5), torch.randn(3, 5))
        m = M().eval()
        # TODO: define qconfig_mapping specifically for executorch
        qconfig_mapping = get_default_qconfig_mapping("qnnpack")
        m = prepare_fx(
            m,
            qconfig_mapping,
            example_inputs,
            backend_config=get_executorch_backend_config(),
        )
        m(*example_inputs)
        m = _convert_to_reference_decomposed_fx(m)
        config = EdgeCompileConfig(_check_ir_validity=False)
        m = to_edge(export(m, example_inputs), compile_config=config)
        # QuantFusionPass should be part of to_executorch() config, separating it out so that we can check the graph.
        m = m.transform([QuantFusionPass()])
        # check that we are using functional variant of q/dq/cat
        FileCheck().check_count(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default",
            1,
            exactly=True,
        ).check("executorch_exir_dialects_edge__ops_aten_cat_default").check_count(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default",
            1,
            exactly=True,
        ).run(
            m.exported_program().graph_module.code
        )

        m = m.to_executorch()
        # Note: quantized add is not fused since the qparams are the same and current subgraph_rewriter
        # doesn't work for the case when single graph node map to two different pattern node
        # one work around would be to add new patterns for the case when qparams are the same
        # for quantized add pattern, but this may not be needed in real use case, we can
        # add this workaround if needed in another diff
        FileCheck().check_count(
            "torch.ops.quantized_decomposed.quantize_per_tensor.out", 1, exactly=True
        ).check("torch.ops.aten.cat.out").check_count(
            "torch.ops.quantized_decomposed.dequantize_per_tensor.out", 1, exactly=True
        ).run(
            m.exported_program().graph_module.code
        )

    def test_embedding_byte(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=12)

            def forward(self, indices):
                return self.emb(indices)

        for qconfig in [float_qparams_weight_only_qconfig]:
            m = M().eval()
            indices = torch.tensor(
                [
                    9,
                    6,
                    5,
                    7,
                    8,
                    8,
                    9,
                    2,
                    8,
                    6,
                    6,
                    9,
                    1,
                    6,
                    8,
                    8,
                    3,
                    2,
                    3,
                    6,
                    3,
                    6,
                    5,
                    7,
                    0,
                    8,
                    4,
                    6,
                    5,
                    8,
                    2,
                    3,
                ]
            )
            example_inputs = (indices,)
            # TODO: define qconfig_mapping specifically for executorch
            qconfig_mapping = get_default_qconfig_mapping("qnnpack")
            qconfig_mapping = qconfig_mapping.set_object_type(
                torch.nn.Embedding, qconfig
            )
            m = prepare_fx(
                m,
                qconfig_mapping,
                example_inputs,
                backend_config=get_executorch_backend_config(),
            )
            m(*example_inputs)
            m = _convert_to_reference_decomposed_fx(m)
            compile_config = EdgeCompileConfig(
                _check_ir_validity=False,
                _use_edge_ops=True,
            )
            m = to_edge(export(m, example_inputs), compile_config=compile_config)
            # QuantFusionPass should be part of to_executorch() config, separating it out so that we can check the graph.
            m = m.transform([QuantFusionPass(_fix_node_meta_val=True)])
            # check that we are using functional variant of q/dq/cat
            FileCheck().check(
                "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_channel_default",
            ).check(
                "executorch_exir_dialects_edge__ops_quantized_decomposed_embedding_byte_default"
            ).run(
                m.exported_program().graph_module.code
            )

            # TODO: enable after the out variants of quantize_per_channel is supported
            # m = m.to_executorch()
            # FileCheck().check(
            #     "executorch_exir_dialects_edge__ops_quantized_decomposed.quantize_per_channel.out",
            # ).check("executorch_exir_dialects_edge__ops_quantized_decomposed.embedding_byte.out"
            # ).run(
            #     m.dump_graph_module().code
            # )

    def test_embedding_byte_functional(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.rand((3, 2))

            def forward(self, indices):
                return F.embedding(indices, self.weight)

        for qconfig in [float_qparams_weight_only_qconfig]:
            m = M().eval()
            indices = torch.tensor(
                [
                    0,
                ]
            )
            example_inputs = (indices,)

            qconfig_mapping = QConfigMapping().set_object_type(
                F.embedding,
                qconfig,
            )

            m = prepare_fx(
                m,
                qconfig_mapping,
                example_inputs,
                backend_config=get_executorch_backend_config(),
            )
            m(*example_inputs)
            m = _convert_to_reference_decomposed_fx(m)
            compile_config = EdgeCompileConfig(
                _check_ir_validity=False,
                _use_edge_ops=True,
            )
            m = to_edge(export(m, example_inputs), compile_config=compile_config)
            # QuantFusionPass should be part of to_executorch() config, separating it out so that we can check the graph.
            m = m.transform([QuantFusionPass(_fix_node_meta_val=True)])
            # check that we are using functional variant of q/dq/cat
            FileCheck().check(
                "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_channel_default",
            ).check(
                "executorch_exir_dialects_edge__ops_quantized_decomposed_embedding_byte_default"
            ).run(
                m.exported_program().graph_module.code
            )

            # TODO: enable after the out variants of quantize_per_channel is supported
            # m = m.to_executorch()
            # FileCheck().check(
            #     "executorch_exir_dialects_edge__ops_quantized_decomposed.quantize_per_channel.out",
            # ).check("executorch_exir_dialects_edge__ops_quantized_decomposed.embedding_byte.out"
            # ).run(
            #     m.dump_graph_module().code
            # )
