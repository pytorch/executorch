# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree

# pyre-unsafe

from collections import Counter
from typing import Tuple

import torch
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torch.ao.quantization import default_per_channel_symmetric_qnnpack_qconfig
from torch.ao.quantization.qconfig import (
    float_qparams_weight_only_qconfig,
    per_channel_weight_observer_range_neg_127_to_127,
    QConfig,
    weight_observer_range_neg_127_to_127,
)
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.export import export_for_training
from torch.testing._internal.common_quantization import (
    NodeSpec as ns,
    TestHelperModules,
)

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    TemporaryFileName,
)
from torchao.quantization.pt2e import (
    allow_exported_model_train_eval,
    compare_results,
    extract_results_from_loggers,
    FROM_NODE_KEY,
    prepare_for_propagation_comparison,
)

from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torchao.quantization.pt2e.quantizer import ComposableQuantizer, Quantizer
from torchao.quantization.pt2e.quantizer.embedding_quantizer import EmbeddingQuantizer
from torchao.testing.pt2e.utils import (
    PT2ENumericDebuggerTestCase,
    PT2EQuantizationTestCase,
)


class TestQuantizePT2E(PT2EQuantizationTestCase):
    def _quantize(self, m, quantizer, example_inputs, is_qat: bool = False):
        # resetting dynamo cache
        torch._dynamo.reset()

        m = export_for_training(m, example_inputs, strict=True).module()
        if is_qat:
            m = prepare_qat_pt2e(m, quantizer)
        else:
            m = prepare_pt2e(m, quantizer)
        m(*example_inputs)
        m = convert_pt2e(m)
        return m

    def _get_pt2e_quantized_linear(
        self, is_per_channel: bool = False
    ) -> torch.fx.GraphModule:
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(
            is_per_channel=is_per_channel
        )
        quantizer.set_global(operator_config)
        example_inputs = (torch.randn(2, 2),)
        m = M().eval()
        return self._quantize(m, quantizer, example_inputs)

    def test_dont_fold_other_constant(self) -> None:
        """Make sure the constant propagation does not apply to things unrelated to
        quantization
        """

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)
                self.dont_fold_me = torch.nn.Parameter(torch.randn(2, 2))

            def forward(self, x):
                t = self.dont_fold_me.t()
                return self.linear(x) + t

        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=False)
        # only quantize linear, so add is not quantized and the constant Tensor
        # should not be folded
        quantizer.set_module_type(torch.nn.Linear, operator_config)
        example_inputs = (torch.randn(2, 2),)
        m = M().eval()
        m = self._quantize(m, quantizer, example_inputs)
        node_occurrence = {
            # quantize op for weight node is folded
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 2,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 3,
            # transpose op not folded
            ns.call_function(torch.ops.aten.t.default): 1,
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_fold_all_ops_before_quantize(self) -> None:
        """Test folding all ops that's before quantized operator:
        Before:
            get_attr(weight) -> transpose -> quantize -> dequantize
        After:
            get_attr(folded_weight) -> dequantize
        """

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.randn(2, 2)

            def forward(self, x):
                t = self.weight.t()
                return torch.nn.functional.linear(x, t)

        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=False)
        quantizer.set_global(operator_config)
        example_inputs = (torch.randn(2, 2),)
        m = M().eval()
        m = self._quantize(m, quantizer, example_inputs)
        node_occurrence = {
            # quantize op for weight node is folded
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 2,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 3,
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_composable_quantizer_throw(self) -> None:
        class BadQuantizer(Quantizer):
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for n in gm.graph.nodes:
                    n.meta["quantization_annotation"] = None

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(quantization_config)
        bad_quantizer = BadQuantizer()
        composable_quantizer = ComposableQuantizer([quantizer, bad_quantizer])
        m_eager = TestHelperModules.ConvLinearWPermute().eval()
        example_inputs = (torch.randn(2, 3, 4, 4),)
        self.assertRaises(
            RuntimeError,
            lambda: self._test_quantizer(
                m_eager, example_inputs, composable_quantizer, {}
            ),
        )

    def test_composable_quantizer_linear_conv(self) -> None:
        dynamic_quantizer = XNNPACKQuantizer()
        quantization_config_dynamic = get_symmetric_quantization_config(
            is_per_channel=False, is_dynamic=True
        )
        dynamic_quantizer.set_operator_type(
            torch.ops.aten.linear.default, quantization_config_dynamic
        )
        static_quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        static_quantizer.set_operator_type(
            torch.ops.aten.conv2d.default, quantization_config
        )
        # Note that dynamic quantization must be applied first here.
        # this is because static quantizer also quantizes linear with static qspec
        # and if we apply static_quantizer first then dynamic_quantizer cannot be applied
        composable_quantizer = ComposableQuantizer(
            [dynamic_quantizer, static_quantizer]
        )
        m_eager = TestHelperModules.ConvLinearWPermute().eval()

        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 1,
            # note: quantize op for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 4,
            # note: quantize op for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        act_affine_quant_obs = (
            torch.ao.quantization.observer.PlaceholderObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_tensor_affine,
                quant_min=-128,
                quant_max=127,
                eps=2**-12,
                is_dynamic=True,
            )
        )
        dynamic_qconfig = QConfig(
            activation=act_affine_quant_obs,
            weight=weight_observer_range_neg_127_to_127,
        )
        # Test with 2d inputs
        example_inputs = (torch.randn(2, 3, 4, 4),)
        qconfig = default_per_channel_symmetric_qnnpack_qconfig
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        qconfig_mapping.set_object_type(torch.nn.Linear, dynamic_qconfig)
        # Had to turn off check against fx because fx quant workflow does not seem
        # to propagate observers for permute node for this model.
        # Suprisingly it does propagate it for EmbeddingConvLinearModule
        # TODO: Figure out the right behavior for propagation
        self._test_quantizer(
            m_eager,
            example_inputs,
            composable_quantizer,
            node_occurrence,
            [],
            False,
            qconfig_mapping,
        )

    def test_embedding_conv_linear_quantization(self) -> None:
        m_eager = TestHelperModules.EmbeddingConvLinearModule().eval()
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
        indices = torch.unsqueeze(indices, 0)
        example_inputs = (indices,)

        embedding_quantizer = EmbeddingQuantizer()
        dynamic_quantizer = XNNPACKQuantizer()
        quantization_config_dynamic = get_symmetric_quantization_config(
            is_per_channel=True, is_dynamic=True
        )
        dynamic_quantizer.set_operator_type(
            torch.ops.aten.linear.default, quantization_config_dynamic
        )
        static_quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        static_quantizer.set_operator_type(
            torch.ops.aten.conv2d.default, quantization_config
        )
        composed_quantizer = ComposableQuantizer(
            [embedding_quantizer, dynamic_quantizer, static_quantizer]
        )

        act_affine_quant_obs = (
            torch.ao.quantization.observer.PlaceholderObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_tensor_affine,
                quant_min=-128,
                quant_max=127,
                eps=2**-12,
                is_dynamic=True,
            )
        )
        dynamic_qconfig = QConfig(
            activation=act_affine_quant_obs,
            weight=per_channel_weight_observer_range_neg_127_to_127,
        )
        qconfig = default_per_channel_symmetric_qnnpack_qconfig
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        qconfig_mapping.set_object_type(torch.nn.Linear, dynamic_qconfig)
        qconfig_mapping = qconfig_mapping.set_object_type(
            torch.nn.Embedding, float_qparams_weight_only_qconfig
        )

        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 4,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 4,
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 1,
            # note: quantize op for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 3,
        }
        self._test_quantizer(
            m_eager,
            example_inputs,
            composed_quantizer,
            node_occurrence,
            [],
            True,
            qconfig_mapping,
        )

    def test_disallow_eval_train(self) -> None:
        m = TestHelperModules.ConvWithBNRelu(relu=True)
        example_inputs = (torch.rand(3, 3, 5, 5),)

        # Before export: this is OK
        m.eval()
        m.train()

        # After export: this is not OK
        m = export_for_training(m, example_inputs, strict=True).module()
        with self.assertRaises(NotImplementedError):
            m.eval()
        with self.assertRaises(NotImplementedError):
            m.train()

        # After prepare: still not OK
        quantizer = XNNPACKQuantizer()
        m = prepare_qat_pt2e(m, quantizer)  # pyre-ignore[6]
        with self.assertRaises(NotImplementedError):
            m.eval()
        with self.assertRaises(NotImplementedError):
            m.train()

        # After convert: still not OK
        m = convert_pt2e(m)
        with self.assertRaises(NotImplementedError):
            m.eval()
        with self.assertRaises(NotImplementedError):
            m.train()

    def _get_bn_train_eval_ops(self) -> Tuple[torch._ops.OpOverload]:
        return (
            torch.ops.aten.batch_norm.default,
            torch.ops.aten.batch_norm.default,
        )

    def _get_node(
        self, m: torch.fx.GraphModule, target: torch._ops.OpOverload
    ) -> torch.fx.Node:
        """
        Return the first node matching the specified target, throwing an exception
        if no such batch norm node is found.
        """
        for n in m.graph.nodes:
            if n.target == target:
                return n
        raise ValueError("Did not find node with target ", target)

    def test_allow_exported_model_train_eval(self) -> None:
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(3)
                self.dropout = torch.nn.Dropout(0.5)

            def forward(self, x):
                x = self.bn(x)
                x = self.dropout(x)
                return x

        m = M().train()
        example_inputs = (torch.randn(1, 3, 3, 3),)
        bn_train_op, bn_eval_op = self._get_bn_train_eval_ops()  # pyre-ignore[23]
        m = export_for_training(m, example_inputs, strict=True).module()

        def _assert_ops_are_correct(m: torch.fx.GraphModule, train: bool) -> None:
            bn_op = bn_train_op if train else bn_eval_op
            bn_node = self._get_node(m, bn_op)
            self.assertTrue(bn_node is not None)
            dropout_node = self._get_node(m, torch.ops.aten.dropout.default)
            self.assertEqual(dropout_node.args[2], train)

        # Before wrapping: this is not OK
        with self.assertRaises(NotImplementedError):
            m.eval()
        with self.assertRaises(NotImplementedError):
            m.train()

        # After wrapping: does not error and swaps the ops accordingly
        allow_exported_model_train_eval(m)  # pyre-ignore[6]
        m.eval()
        _assert_ops_are_correct(m, train=False)  # pyre-ignore[6]
        m.train()
        _assert_ops_are_correct(m, train=True)  # pyre-ignore[6]

        # After prepare but before wrapping: this is not OK
        quantizer = XNNPACKQuantizer()
        m = prepare_qat_pt2e(m, quantizer)  # pyre-ignore[6]
        with self.assertRaises(NotImplementedError):
            m.eval()
        with self.assertRaises(NotImplementedError):
            m.train()

        # After prepare and after wrapping: does not error and swaps the ops accordingly
        allow_exported_model_train_eval(m)
        m.eval()
        _assert_ops_are_correct(m, train=False)
        m.train()
        _assert_ops_are_correct(m, train=True)

        # After convert but before wrapping: this is not OK
        m = convert_pt2e(m, fold_quantize=True)
        with self.assertRaises(NotImplementedError):
            m.eval()
        with self.assertRaises(NotImplementedError):
            m.train()

        # After convert and after wrapping: does not error and swaps the ops accordingly
        allow_exported_model_train_eval(m)
        m.eval()
        _assert_ops_are_correct(m, train=False)
        m.train()
        _assert_ops_are_correct(m, train=True)

    def test_constant_prop_preserve_metadata(self) -> None:
        """Test to make sure the get_attr node for const propagated weight Tensor gets the correct
        metadata (from original get_attr node from weight)
        """

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config()
        quantizer.set_global(operator_config)
        example_inputs = (torch.randn(2, 2),)
        m = M().eval()
        m = export_for_training(m, example_inputs, strict=True).module()
        weight_meta = None
        for n in m.graph.nodes:  # pyre-ignore[16]
            if (
                n.op == "get_attr"
                and next(iter(n.users)).target == torch.ops.aten.linear.default
            ):
                weight_meta = n.meta
                break
        assert weight_meta is not None, "Expect to find metadata for weight node"

        m = prepare_pt2e(m, quantizer)  # pyre-ignore[6]
        m(*example_inputs)
        m = convert_pt2e(m)

        for n in m.graph.nodes:
            if n.op == "get_attr" and "frozen_param" in n.target:
                for key in n.meta:
                    if key != FROM_NODE_KEY:
                        self.assertEqual(n.meta[key], weight_meta[key])

    def test_reentrant(self) -> None:
        """Test we can safely call quantization apis multiple times"""
        m = TestHelperModules.ConvBnReLU2dAndLinearReLU()
        example_inputs = (torch.randn(3, 3, 10, 10),)

        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=True, is_qat=True)
        )
        m.conv_bn_relu = export_for_training(  # pyre-ignore[8]
            m.conv_bn_relu, example_inputs, strict=True
        ).module()
        m.conv_bn_relu = prepare_qat_pt2e(m.conv_bn_relu, quantizer)  # pyre-ignore[6,8]
        m(*example_inputs)
        m.conv_bn_relu = convert_pt2e(m.conv_bn_relu)  # pyre-ignore[6, 8]

        quantizer = XNNPACKQuantizer().set_module_type(
            torch.nn.Linear, get_symmetric_quantization_config(is_per_channel=False)
        )
        m = export_for_training(m, example_inputs, strict=True).module()
        m = prepare_pt2e(m, quantizer)  # pyre-ignore[6]
        m = convert_pt2e(m)

        node_occurrence = {
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 4,
            # one for weight
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 5,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_channel.default
            ): 1,
        }
        node_list = [
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.conv2d.default),
            ns.call_function(torch.ops.aten.relu.default),
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.linear.default),
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
        ]
        self.checkGraphModuleNodes(
            m, expected_node_occurrence=node_occurrence, expected_node_list=node_list
        )

    def test_groupwise_per_channel_quant(self) -> None:
        m = TestHelperModules.GroupwiseConv2d()
        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(operator_config)
        example_inputs = m.example_inputs()
        m = self._quantize(m, quantizer, example_inputs)
        # make sure it runs
        m(*example_inputs)

    def test_preserve_nn_module_stack(self) -> None:
        """Test we can preserve nn_module_stack on replaced pattern's nodes"""
        m = TestHelperModules.ConvBnReLU2dAndLinearReLU()
        example_inputs = (torch.randn(3, 3, 10, 10),)

        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=True, is_qat=True)
        )

        def check_nn_module(node: torch.fx.Node) -> None:
            self.assertTrue("nn_module_stack" in node.meta)
            self.assertTrue(
                "ConvWithBNRelu" in node.meta["nn_module_stack"]["L__self__"][1]
            )

        m.conv_bn_relu = export_for_training(  # pyre-ignore[8]
            m.conv_bn_relu, example_inputs, strict=True
        ).module()
        for node in m.conv_bn_relu.graph.nodes:  # pyre-ignore[16]
            if node.op not in ["placeholder", "output", "get_attr"]:
                check_nn_module(node)
        m.conv_bn_relu = prepare_qat_pt2e(m.conv_bn_relu, quantizer)  # pyre-ignore[6,8]
        for node in m.conv_bn_relu.graph.nodes:  # pyre-ignore[16]
            if node.name == "mul":
                check_nn_module(node)

    def test_speed(self) -> None:
        import time  # noqa: F401

        def dynamic_quantize_pt2e(model, example_inputs) -> torch.fx.GraphModule:
            torch._dynamo.reset()
            model = export_for_training(model, example_inputs, strict=True).module()
            # Per channel quantization for weight
            # Dynamic quantization for activation
            # Please read a detail: https://fburl.com/code/30zds51q
            embedding_quantizer = EmbeddingQuantizer()
            dynamic_quantizer = XNNPACKQuantizer()
            operator_config_dynamic = get_symmetric_quantization_config(
                is_per_channel=True, is_dynamic=True
            )
            dynamic_quantizer.set_global(operator_config_dynamic)
            composed_quantizer = ComposableQuantizer(
                [embedding_quantizer, dynamic_quantizer]
            )
            # prev = time.time()
            model = prepare_qat_pt2e(model, composed_quantizer)  # pyre-ignore[6]
            # cur = time.time()
            # print("prepare time:", cur - prev)
            # Without Calibraiton, scale/zero value will have an initialized value of 1.0
            # Per channel quantization needs a proper scale/zero shape/value to work properly.
            # So we need to run calibration before converting to quantized model.
            model(*example_inputs)
            # prev = time.time()
            model = convert_pt2e(model)
            # cur = time.time()
            # uncomment to see the time
            # print("convert time:", cur - prev)
            return model

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x)

        m = M().eval()
        example_inputs = (torch.randn(5, 5),)
        _ = dynamic_quantize_pt2e(m, example_inputs)

    def test_multi_users_without_output_observer(self) -> None:
        """
        Test the case in which a node is used by multiple users,
        and had its output observer removed.
        """

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                x = self.conv(x)
                return x, x + 1

        example_inputs = (torch.randn(1, 3, 5, 5),)
        m = M()
        m = export_for_training(m, example_inputs, strict=True).module()
        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(),
        )
        m = prepare_pt2e(m, quantizer)  # pyre-ignore[6]
        m(*example_inputs)

        # Remove output observer
        observer_to_remove = None
        for n in m.graph.nodes:
            if n.op == "output":
                observer_to_remove = n.args[0][0]
                assert observer_to_remove.op == "call_module"
                assert observer_to_remove.target.startswith("activation_post_process_")
                break
        assert observer_to_remove is not None
        observer_to_remove.replace_all_uses_with(observer_to_remove.args[0])
        m.graph.erase_node(observer_to_remove)
        m.recompile()

        # Convert should succeed
        m = convert_pt2e(m)
        m(*example_inputs)

    def test_fold_quantize(self) -> None:
        """Test to make sure the quantized model gets quantized weight (quantize_per_tensor op is folded)"""
        m = self._get_pt2e_quantized_linear()
        node_occurrence = {
            # quantize op for weight node is folded
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 2,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 3,
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_fold_quantize_per_channel(self) -> None:
        """Test to make sure the quantized model gets quantized weight (quantize_per_channel op is folded)"""
        m = self._get_pt2e_quantized_linear(is_per_channel=True)
        node_occurrence = {
            # quantize op for weight node is folded
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 2,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_channel.default
            ): 1,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 2,
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_save_load(self) -> None:
        """Test save/load a quantized model"""
        m = self._get_pt2e_quantized_linear()
        example_inputs = (torch.randn(2, 2),)
        ref_res = m(*example_inputs)

        with TemporaryFileName() as fname:
            # serialization
            quantized_ep = torch.export.export(m, example_inputs, strict=True)
            torch.export.save(quantized_ep, fname)
            # deserialization
            loaded_ep = torch.export.load(fname)
            loaded_quantized_model = loaded_ep.module()
            res = loaded_quantized_model(*example_inputs)
            self.assertEqual(ref_res, res)


instantiate_parametrized_tests(TestQuantizePT2E)


class TestXNNPACKQuantizerNumericDebugger(PT2ENumericDebuggerTestCase):

    def test_quantize_pt2e_preserve_handle(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = export_for_training(m, example_inputs, strict=True)
        m = ep.module()

        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=False)
        )
        m = prepare_pt2e(m, quantizer)
        from_node_source_map = self._extract_from_node_source(m)
        node_name_equip_with_output_observer = [
            "conv2d",
            "conv1d",
            "squeeze",
        ]
        res_counter = Counter(from_node_source_map.values())
        repeated_from_node_source = [
            from_node_source_map[n_name]
            for n_name in node_name_equip_with_output_observer
        ]
        # 3 infos were repeated because we copy over the info from node to its output observer
        # torch.ops.aten.conv2d.default, torch.ops.aten.squeeze.dim and torch.ops.aten.conv1d.default
        for from_node_source in repeated_from_node_source:
            self.assertEqual(res_counter[from_node_source], 2)

        m(*example_inputs)
        m = convert_pt2e(m)
        self._assert_each_node_has_from_node_source(m)
        from_node_source_map = self._extract_from_node_source(m)
        res_counter = Counter(from_node_source_map.values())
        # same set of infos where repeated, because we copy over the info from observer/fake_quant to
        # quantize/dequantize node
        repeated_from_node_source = [
            from_node_source_map[n_name]
            for n_name in node_name_equip_with_output_observer
        ]
        for from_node_source in repeated_from_node_source:
            self.assertEqual(res_counter[from_node_source], 3)

    def test_extract_results_from_loggers(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = export_for_training(m, example_inputs, strict=True)
        m = ep.module()
        m_ref_logger = prepare_for_propagation_comparison(m)

        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=False)
        )
        m = prepare_pt2e(m, quantizer)
        m(*example_inputs)
        m = convert_pt2e(m)
        m_quant_logger = prepare_for_propagation_comparison(m)

        m_ref_logger(*example_inputs)
        m_quant_logger(*example_inputs)
        ref_results = extract_results_from_loggers(m_ref_logger)
        quant_results = extract_results_from_loggers(m_quant_logger)
        comparison_results = compare_results(ref_results, quant_results)
        for node_summary in comparison_results.values():
            if len(node_summary.results) > 0:
                self.assertGreaterEqual(node_summary.results[0].sqnr, 35)

    def test_extract_results_from_loggers_list_output(self):
        m = TestHelperModules.Conv2dWithSplit()
        example_inputs = m.example_inputs()
        ep = export_for_training(m, example_inputs, strict=True)
        m = ep.module()
        m_ref_logger = prepare_for_propagation_comparison(m)

        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=False)
        )
        m = prepare_pt2e(m, quantizer)
        m(*example_inputs)
        m = convert_pt2e(m)
        m_quant_logger = prepare_for_propagation_comparison(m)

        m_ref_logger(*example_inputs)
        m_quant_logger(*example_inputs)
        ref_results = extract_results_from_loggers(m_ref_logger)
        quant_results = extract_results_from_loggers(m_quant_logger)
        comparison_results = compare_results(ref_results, quant_results)
        for node_summary in comparison_results.values():
            if len(node_summary.results) > 0:
                sqnr = node_summary.results[0].sqnr
                if isinstance(sqnr, list):
                    for sqnr_i in sqnr:
                        self.assertGreaterEqual(sqnr_i, 35)
                else:
                    self.assertGreaterEqual(sqnr, 35)
