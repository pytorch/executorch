# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Tuple

import torch
from executorch.backends.cuda.cuda_partitioner import (
    _WEIGHT_OFFLOAD_ENABLE_SPEC_KEY,
    _WEIGHT_OFFLOAD_PIN_FQNS_SPEC_KEY,
    CudaPartitioner,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import PartitionResult
from torch.export import export


class TestCudaPartitioner(unittest.TestCase):
    """
    Test CUDA partitioner functionality.

    After CUDA partitioning, there should be exactly one partitioned graph that contains
    all operators from the input graph. This means all operators should be tagged with
    the same delegation tag, indicating they will all be executed by the CUDA backend.
    """

    def _get_partition_result(
        self, module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...]
    ) -> PartitionResult:
        """Helper method to get partition result for a given module."""
        # Export the model
        exported_program = export(module, inputs, strict=True)

        # Create partitioner and compile specs
        partitioner = CudaPartitioner([])

        # Get partition result
        partition_result = partitioner.partition(exported_program)

        # Verify partition result structure
        self.assertIsNotNone(partition_result)
        self.assertTrue(hasattr(partition_result, "tagged_exported_program"))
        self.assertTrue(hasattr(partition_result, "partition_tags"))

        return partition_result

    def _check_fully_partitioned(self, partition_result: PartitionResult) -> bool:
        """Check if the graph is fully partitioned (all operators have the same tag)."""
        tagged_nodes = []
        untagged_ops = []

        for node in partition_result.tagged_exported_program.graph.nodes:
            if node.op == "call_function":
                if hasattr(node, "meta") and "delegation_tag" in node.meta:
                    tagged_nodes.append(node)
                else:
                    untagged_ops.append(node)

        # Check if we have any tagged nodes
        if not tagged_nodes:
            return False

        # Check if all tagged nodes have the same tag
        first_tag = tagged_nodes[0].meta["delegation_tag"]
        all_same_tag = all(
            node.meta.get("delegation_tag") == first_tag for node in tagged_nodes
        )

        # Should have no untagged operations for full partitioning
        fully_partitioned = len(untagged_ops) == 0 and all_same_tag

        return fully_partitioned

    def test_simple_add_partition(self):
        """
        Test that CUDA partitioner creates exactly one partition containing all operators.
        Simple element-wise addition should result in a single graph with all ops tagged identically.
        """

        class AddModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        module = AddModule()
        inputs = (torch.randn(3, 4), torch.randn(3, 4))

        partition_result = self._get_partition_result(module, inputs)
        fully_partitioned = self._check_fully_partitioned(partition_result)

        self.assertTrue(
            fully_partitioned,
            "Graph should be fully partitioned with all operators having the same tag",
        )

    def test_conv2d_partition(self):
        """
        Test that CUDA partitioner creates exactly one partition containing all operators.
        Conv2D operation should result in a single graph with all ops tagged identically.
        """

        class Conv2dModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)

        module = Conv2dModule()
        inputs = (torch.randn(1, 3, 32, 32),)

        partition_result = self._get_partition_result(module, inputs)
        fully_partitioned = self._check_fully_partitioned(partition_result)

        self.assertTrue(
            fully_partitioned,
            "Graph should be fully partitioned with all operators having the same tag",
        )

    def test_linear_partition(self):
        """
        Test that CUDA partitioner creates exactly one partition containing all operators.
        Linear layer operation should result in a single graph with all ops tagged identically.
        """

        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(128, 64)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        module = LinearModule()
        inputs = (torch.randn(8, 128),)

        partition_result = self._get_partition_result(module, inputs)
        fully_partitioned = self._check_fully_partitioned(partition_result)

        self.assertTrue(
            fully_partitioned,
            "Graph should be fully partitioned with all operators having the same tag",
        )

    def test_unused_constant_tagging(self):
        """
        Test that constant nodes without users are properly tagged with delegation_tag.

        When a graph contains constants (parameters, buffers, or lifted tensor constants)
        that are not used by any operations, the CUDA partitioner should still tag them
        with the delegation_tag. This ensures all constant data is properly handled during
        delegation, even if they have no users in the graph.
        """

        class ModuleWithUnusedConst(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Register a buffer that won't be used in forward
                self.register_buffer("unused_buffer", torch.randn(10, 10))
                # Also register a used parameter
                self.weight = torch.nn.Parameter(torch.randn(5, 5))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Only use the weight parameter, not the unused_buffer
                return x + self.weight

        module = ModuleWithUnusedConst()
        inputs = (torch.randn(5, 5),)

        # Get partition result
        partition_result = self._get_partition_result(module, inputs)

        # Find all placeholder nodes (these represent constants, parameters, buffers, and inputs)
        constant_placeholders = []
        input_placeholders = []

        for node in partition_result.tagged_exported_program.graph.nodes:
            if node.op == "placeholder":
                # Check if this is a constant (param, buffer, or lifted tensor constant)
                from torch._export.utils import (
                    is_buffer,
                    is_lifted_tensor_constant,
                    is_param,
                )

                is_constant = (
                    is_param(partition_result.tagged_exported_program, node)
                    or is_buffer(partition_result.tagged_exported_program, node)
                    or is_lifted_tensor_constant(
                        partition_result.tagged_exported_program, node
                    )
                )

                if is_constant:
                    constant_placeholders.append(node)
                else:
                    input_placeholders.append(node)

        # Verify we have constant placeholders
        self.assertGreater(
            len(constant_placeholders),
            0,
            "Expected to find constant placeholders in the graph",
        )

        # Check that all constant placeholders are tagged, including unused ones
        untagged_constants = []
        for node in constant_placeholders:
            if "delegation_tag" not in node.meta:
                untagged_constants.append(node.name)

        self.assertEqual(
            len(untagged_constants),
            0,
            f"All constant placeholders should be tagged. Found untagged constants: {untagged_constants}",
        )

        # Verify all tagged constants have the expected tag
        expected_tag = "tag0"
        for node in constant_placeholders:
            actual_tag = node.meta.get("delegation_tag")
            self.assertEqual(
                actual_tag,
                expected_tag,
                f"Constant placeholder {node.name} has tag '{actual_tag}' but expected '{expected_tag}'",
            )

    # ----------------------------------------------------------------
    # Weight-offload public kwargs
    # ----------------------------------------------------------------

    def _find_specs(self, partitioner, key):
        # Reach into the partitioner's stored compile_spec list. The
        # base AotiPartitioner stashes it as
        # ``self.delegation_spec.compile_specs``; we read it back
        # here to verify the public kwargs produced the expected
        # internal entries.
        return [s for s in partitioner.delegation_spec.compile_specs if s.key == key]

    def test_partitioner_public_kwargs_round_trip(self):
        partitioner = CudaPartitioner(
            [],
            weight_offload=True,
            weight_offload_pin_fqns=["w1"],
        )
        enable_specs = self._find_specs(partitioner, _WEIGHT_OFFLOAD_ENABLE_SPEC_KEY)
        pin_specs = self._find_specs(partitioner, _WEIGHT_OFFLOAD_PIN_FQNS_SPEC_KEY)
        self.assertEqual(len(enable_specs), 1, "expected one enable spec")
        self.assertEqual(enable_specs[0].value, b"1")
        self.assertEqual(len(pin_specs), 1, "expected one pin spec")
        self.assertEqual(pin_specs[0].value, b"w1")

    def test_partitioner_dedupes_pin_fqns(self):
        partitioner = CudaPartitioner(
            [],
            weight_offload=True,
            weight_offload_pin_fqns=["w1", "w2", "w1"],
        )
        pin_specs = self._find_specs(partitioner, _WEIGHT_OFFLOAD_PIN_FQNS_SPEC_KEY)
        self.assertEqual(len(pin_specs), 1)
        # Deduped first-seen order: w1, w2.
        self.assertEqual(pin_specs[0].value, b"w1\x00w2")

    def test_partitioner_rejects_pin_without_enable(self):
        with self.assertRaisesRegex(ValueError, "weight_offload=False"):
            CudaPartitioner(
                [],
                weight_offload=False,
                weight_offload_pin_fqns=["w1"],
            )

    def test_partitioner_rejects_bare_str_pin_fqns(self):
        with self.assertRaisesRegex(TypeError, "bare str"):
            CudaPartitioner(
                [],
                weight_offload=True,
                weight_offload_pin_fqns="w1",  # type: ignore[arg-type]
            )

    def test_partitioner_rejects_non_string_pin_fqn(self):
        with self.assertRaisesRegex(TypeError, "must be strings"):
            CudaPartitioner(
                [],
                weight_offload=True,
                weight_offload_pin_fqns=["w1", 42],  # type: ignore[list-item]
            )

    def test_partitioner_rejects_non_list_pin_fqns(self):
        # Tuples, sets, dict_keys, generators are all rejected by the
        # strict list-only check. Callers must cast to list explicitly
        # so the first-seen-order contract is preserved.
        for bad in (
            ("w1",),  # tuple
            {"w1"},  # set (nondeterministic iteration)
            (fqn for fqn in ["w1"]),  # generator (one-shot)
        ):
            with self.assertRaisesRegex(TypeError, "must be a list"):
                CudaPartitioner(
                    [],
                    weight_offload=True,
                    weight_offload_pin_fqns=bad,  # type: ignore[arg-type]
                )

    def test_partitioner_rejects_any_mixed_channel(self):
        # Same key — kwarg + raw internal enable both set.
        with self.assertRaisesRegex(ValueError, "_weight_offload_internal_"):
            CudaPartitioner(
                [CompileSpec(_WEIGHT_OFFLOAD_ENABLE_SPEC_KEY, b"1")],
                weight_offload=True,
            )

        # Different key — kwarg sets enable, raw sets pin_fqns.
        with self.assertRaisesRegex(ValueError, "_weight_offload_internal_"):
            CudaPartitioner(
                [CompileSpec(_WEIGHT_OFFLOAD_PIN_FQNS_SPEC_KEY, b"w1")],
                weight_offload=True,
            )

        # Raw internal specs WITHOUT any public kwarg — still allowed
        # (preserves the test stack that builds raw specs).
        partitioner = CudaPartitioner(
            [CompileSpec(_WEIGHT_OFFLOAD_ENABLE_SPEC_KEY, b"1")],
        )
        enable_specs = self._find_specs(partitioner, _WEIGHT_OFFLOAD_ENABLE_SPEC_KEY)
        self.assertEqual(len(enable_specs), 1)

    def test_partitioner_rejects_non_default_target_device_with_offload(self):
        """``weight_offload=True`` plus ``target_device != cuda:0`` is
        rejected: payload + runtime hard-code device 0 today, so any
        other target would silently end up on the wrong GPU."""
        from executorch.exir.passes.propagate_device_pass import (
            TARGET_DEVICE_COMPILE_SPEC_KEY,
        )

        # Wrong index: cuda:1.
        with self.assertRaisesRegex(ValueError, "currently requires"):
            CudaPartitioner(
                [CompileSpec(TARGET_DEVICE_COMPILE_SPEC_KEY, b"cuda:1")],
                weight_offload=True,
            )

        # Wrong device type entirely: cpu.
        with self.assertRaisesRegex(ValueError, "currently requires"):
            CudaPartitioner(
                [CompileSpec(TARGET_DEVICE_COMPILE_SPEC_KEY, b"cpu")],
                weight_offload=True,
            )

        # Explicit cuda:0 is fine.
        p_explicit = CudaPartitioner(
            [CompileSpec(TARGET_DEVICE_COMPILE_SPEC_KEY, b"cuda:0")],
            weight_offload=True,
        )
        self.assertEqual(
            len(self._find_specs(p_explicit, _WEIGHT_OFFLOAD_ENABLE_SPEC_KEY)),
            1,
        )

        # Bare ``cuda`` (no index) parses to index 0 and is also fine.
        p_bare = CudaPartitioner(
            [CompileSpec(TARGET_DEVICE_COMPILE_SPEC_KEY, b"cuda")],
            weight_offload=True,
        )
        self.assertEqual(
            len(self._find_specs(p_bare, _WEIGHT_OFFLOAD_ENABLE_SPEC_KEY)),
            1,
        )

        # Non-default target_device is fine when offload is OFF.
        p_off = CudaPartitioner(
            [CompileSpec(TARGET_DEVICE_COMPILE_SPEC_KEY, b"cuda:1")],
            weight_offload=False,
        )
        self.assertEqual(
            len(self._find_specs(p_off, _WEIGHT_OFFLOAD_ENABLE_SPEC_KEY)),
            0,
        )

    def test_partitioner_internal_keys_match_pass(self):
        """The four string constants inlined in cuda_partitioner.py
        must equal the canonical constants exported by
        weight_offload_pass.py. Drift would mean the public kwargs
        emit specs the runtime doesn't recognize."""
        from executorch.backends.cuda.cuda_partitioner import (
            _WEIGHT_OFFLOAD_ENABLE_SPEC_KEY as part_enable,
            _WEIGHT_OFFLOAD_INTERNAL_KEY_PREFIX as part_prefix,
            _WEIGHT_OFFLOAD_PIN_FQNS_SPEC_KEY as part_pin,
        )
        from executorch.backends.cuda.passes.weight_offload_pass import (
            COMPILE_SPEC_KEY_ENABLE,
            COMPILE_SPEC_KEY_PIN_FQNS,
        )

        self.assertEqual(part_enable, COMPILE_SPEC_KEY_ENABLE)
        self.assertEqual(part_pin, COMPILE_SPEC_KEY_PIN_FQNS)
        # Prefix is the shared underscore-prefix that defines the
        # internal channel.
        self.assertTrue(part_enable.startswith(part_prefix))
        self.assertTrue(part_pin.startswith(part_prefix))
