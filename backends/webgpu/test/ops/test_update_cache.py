# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""fp32 update_cache (KV-cache write) export tests via VulkanPartitioner.

Verifies the export/delegation side here; on-GPU numerics are checked by the
dedicated native test `test/native/test_update_cache.cpp`: single-shot cases
(non-zero input_pos + varied shapes) via `export_update_cache_cases`, and the
multi-step advancing-input_pos replay (mirroring VulkanSDPATest) via
`export_update_cache_replay`. update_cache scatters a projected value tensor
[1, S, H, D] into the KV cache [1, Cmax, H, D] at the sequence offset input_pos.
"""

import os
import unittest

import torch

# Importing custom_ops registers torch.ops.llama.update_cache (the schema lives
# in the C++ AOT lib loaded here).
from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
    VulkanPartitioner,
)
from executorch.backends.vulkan.serialization.vulkan_graph_serialize import (
    extract_vk_flatbuffer,
    flatbuffer_to_vk_graph,
)
from executorch.examples.models.gemma4.webgpu_partitioner import (
    build_webgpu_partitioner,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.schema import DataLocation, DelegateCall, KernelCall
from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
from torch.export.graph_signature import InputKind, OutputKind


class UpdateCacheModule(torch.nn.Module):
    """Writes the projected value into the KV cache at input_pos."""

    def __init__(self, input_pos: int = 0) -> None:
        super().__init__()
        self.input_pos = input_pos

    def forward(self, value: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        return torch.ops.llama.update_cache(value, cache, self.input_pos)


class DynamicUpdateCacheModule(torch.nn.Module):
    """Writes at the live scalar selected from the position tensor."""

    def forward(
        self,
        value: torch.Tensor,
        cache: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.llama.update_cache(value, cache, input_pos[0].item())


class RegisteredBufferDynamicUpdateCacheModule(torch.nn.Module):
    """Mirrors Gemma's non-persistent registered KV cache."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "cache",
            torch.zeros(1, 1024, 2, 4),
            persistent=False,
        )

    def forward(
        self,
        value: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        torch.ops.llama.update_cache(value, self.cache, input_pos[0].item())
        return self.cache[:, :1]


class IntermediateDynamicUpdateCacheModule(torch.nn.Module):
    """Feeds an intermediate value with a live sequence shape to the cache."""

    def forward(
        self,
        value: torch.Tensor,
        cache: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        produced = torch.sigmoid(value)
        return torch.ops.llama.update_cache(
            produced,
            cache,
            input_pos[0].item(),
        )


def _lower_dynamic_update_cache(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    dynamic_shapes: tuple[object, ...],
) -> tuple[torch.export.ExportedProgram, object]:
    with torch._dynamo.config.patch(capture_scalar_outputs=True):
        exported = torch.export.export(
            model,
            inputs,
            dynamic_shapes=dynamic_shapes,
        )
    lowered = to_edge_transform_and_lower(
        exported,
        partitioner=[build_webgpu_partitioner("8da4w+emb4")],
    ).to_executorch()
    return exported, lowered


def _export_dynamic_update_cache() -> tuple[torch.export.ExportedProgram, object]:
    value = torch.zeros(1, 512, 2, 4)
    cache = torch.zeros(1, 1024, 2, 4)
    input_pos = torch.zeros(1, dtype=torch.long)
    value_seq = torch.export.Dim("value_seq", min=1, max=512)
    cache_capacity = torch.export.Dim("cache_capacity", min=512, max=1024)
    return _lower_dynamic_update_cache(
        DynamicUpdateCacheModule(),
        (value, cache, input_pos),
        ({1: value_seq}, {1: cache_capacity}, None),
    )


def _export_registered_program() -> tuple[
    torch.export.ExportedProgram,
    tuple[torch.Tensor, torch.Tensor],
    tuple[object, object],
]:
    value = torch.zeros(1, 512, 2, 4)
    input_pos = torch.zeros(1, dtype=torch.long)
    value_seq = torch.export.Dim("registered_value_seq", min=1, max=512)
    inputs = (value, input_pos)
    dynamic_shapes = ({1: value_seq}, None)
    with torch._dynamo.config.patch(capture_scalar_outputs=True):
        exported = torch.export.export(
            RegisteredBufferDynamicUpdateCacheModule(),
            inputs,
            dynamic_shapes=dynamic_shapes,
        )
    return exported, inputs, dynamic_shapes


def _export_registered_dynamic_update_cache() -> tuple[
    torch.export.ExportedProgram, object
]:
    contract, inputs, dynamic_shapes = _export_registered_program()
    _, lowered = _lower_dynamic_update_cache(
        RegisteredBufferDynamicUpdateCacheModule(), inputs, dynamic_shapes
    )
    return contract, lowered


def _export_intermediate_dynamic_update_cache() -> tuple[
    torch.export.ExportedProgram, object
]:
    value = torch.zeros(1, 512, 2, 4)
    cache = torch.zeros(1, 768, 2, 4)
    input_pos = torch.zeros(1, dtype=torch.long)
    value_seq = torch.export.Dim("intermediate_value_seq", min=1, max=512)
    return _lower_dynamic_update_cache(
        IntermediateDynamicUpdateCacheModule(),
        (value, cache, input_pos),
        ({1: value_seq}, None, None),
    )


class TestUpdateCache(unittest.TestCase):
    def _assert_live_update_cache_symint(
        self,
        exported: torch.export.ExportedProgram,
    ) -> None:
        update_nodes = [
            node
            for node in exported.graph_module.graph.nodes
            if node.target == torch.ops.llama.update_cache.default
        ]
        self.assertEqual(1, len(update_nodes))
        start_pos = update_nodes[0].args[2]
        self.assertIsInstance(start_pos, torch.fx.Node)
        self.assertIsInstance(start_pos.meta.get("val"), torch.SymInt)

    def _assert_exact_delegate_chain(
        self,
        program: object,
        expected_chain: list[str] | None = None,
    ) -> None:
        self.assertEqual(1, len(program.execution_plan))
        plan = program.execution_plan[0]
        self.assertEqual(
            ["VulkanBackend"],
            [delegate.id for delegate in plan.delegates],
        )
        self.assertEqual(1, len(plan.delegates))
        delegate = plan.delegates[0]
        self.assertEqual(DataLocation.INLINE, delegate.processed.location)
        self.assertGreaterEqual(delegate.processed.index, 0)
        self.assertLess(delegate.processed.index, len(program.backend_delegate_data))
        payload = program.backend_delegate_data[delegate.processed.index].data
        vk_graph = flatbuffer_to_vk_graph(extract_vk_flatbuffer(payload))
        self.assertEqual(
            expected_chain
            or ["et_vk.select_as_symint.default", "update_cache.default"],
            [operator.name for operator in vk_graph.chain],
        )

    def _assert_user_cache_writeback(
        self,
        program: object,
        expected_delegate_chain: list[str],
    ) -> None:
        self.assertEqual(1, len(program.execution_plan))
        plan = program.execution_plan[0]
        calls = [
            instruction.instr_args
            for chain in plan.chains
            for instruction in chain.instructions
        ]
        delegate_calls = [call for call in calls if isinstance(call, DelegateCall)]
        kernel_calls = [call for call in calls if isinstance(call, KernelCall)]
        self.assertEqual(1, len(delegate_calls), repr(plan))
        self.assertEqual(3, len(kernel_calls), repr(plan))
        self.assertEqual(
            [
                ("aten::copy", "out"),
                ("aten::copy", "out"),
                ("aten::copy_", ""),
            ],
            [
                (
                    plan.operators[call.op_index].name,
                    plan.operators[call.op_index].overload,
                )
                for call in kernel_calls
            ],
        )
        self.assertEqual(
            [("aten::copy", "out"), ("aten::copy_", "")],
            [(operator.name, operator.overload) for operator in plan.operators],
        )

        cache_input = plan.inputs[1]
        first_stage, second_stage, writeback = kernel_calls
        delegate_output = first_stage.args[1]
        self.assertIn(delegate_output, delegate_calls[0].args)
        self.assertNotIn(delegate_output, plan.inputs)
        self.assertEqual(cache_input, first_stage.args[0])
        self.assertEqual(first_stage.args[-2], first_stage.args[-1])
        first_stage_output = first_stage.args[-1]
        self.assertEqual(cache_input, second_stage.args[0])
        self.assertEqual(first_stage_output, second_stage.args[1])
        self.assertEqual(second_stage.args[-2], second_stage.args[-1])
        second_stage_output = second_stage.args[-1]
        self.assertEqual(cache_input, writeback.args[0])
        self.assertEqual(second_stage_output, writeback.args[1])
        self.assertEqual(cache_input, writeback.args[-1])
        self.assertIn(cache_input, plan.outputs)
        self._assert_exact_delegate_chain(program, expected_delegate_chain)

    def _export_and_check(self, model, example_inputs) -> None:
        ep = torch.export.export(model, example_inputs)
        et_program = to_edge_transform_and_lower(
            ep, partitioner=[VulkanPartitioner()]
        ).to_executorch()

        found_vulkan = False
        for plan in et_program.executorch_program.execution_plan:
            for delegate in plan.delegates:
                if delegate.id == "VulkanBackend":
                    found_vulkan = True
                    break
        self.assertTrue(found_vulkan, "Expected VulkanBackend delegate in .pte")

    def test_update_cache_prefill_small(self) -> None:
        # input_pos=0 prefill: value [1,S=2,H=2,D=4] into cache [1,Cmax=8,H=2,D=4].
        value = torch.randn(1, 2, 2, 4)
        cache = torch.zeros(1, 8, 2, 4)
        self._export_and_check(UpdateCacheModule(0), (value, cache))

    def test_update_cache_gqa_shapes(self) -> None:
        # GQA-style: fewer kv heads, larger head dim.
        value = torch.randn(1, 3, 2, 8)
        cache = torch.zeros(1, 16, 2, 8)
        self._export_and_check(UpdateCacheModule(0), (value, cache))

    def test_registered_cache_keeps_live_position_and_has_no_portable_call(
        self,
    ) -> None:
        exported, lowered = _export_registered_dynamic_update_cache()
        self._assert_live_update_cache_symint(exported)
        buffers = [
            input_spec
            for input_spec in exported.graph_signature.input_specs
            if input_spec.kind == InputKind.BUFFER
        ]
        self.assertEqual(1, len(buffers))
        self.assertEqual("cache", buffers[0].target)
        self.assertFalse(buffers[0].persistent)
        self.assertTrue(
            all(
                output.target != "cache"
                for output in exported.graph_signature.output_specs
                if output.kind == OutputKind.USER_OUTPUT
            )
        )

        program = lowered.executorch_program
        plan = program.execution_plan[0]
        calls = [
            instruction.instr_args
            for chain in plan.chains
            for instruction in chain.instructions
        ]
        self.assertEqual([], plan.operators)
        self.assertEqual([], [call for call in calls if isinstance(call, KernelCall)])
        self.assertEqual(
            1,
            len([call for call in calls if isinstance(call, DelegateCall)]),
        )
        self._assert_exact_delegate_chain(
            program,
            [
                "et_vk.prepack.default",
                "et_vk.select_as_symint.default",
                "update_cache.default",
                "aten.slice_copy.Tensor",
            ],
        )

    def test_user_cache_keeps_live_position_and_exact_mutation_writeback(
        self,
    ) -> None:
        exported, lowered = _export_dynamic_update_cache()
        self._assert_live_update_cache_symint(exported)

        self._assert_user_cache_writeback(
            lowered.executorch_program,
            ["et_vk.select_as_symint.default", "update_cache.default"],
        )

    def test_intermediate_cache_characterization(self) -> None:
        exported, lowered = _export_intermediate_dynamic_update_cache()
        self._assert_live_update_cache_symint(exported)
        update_node = next(
            node
            for node in exported.graph_module.graph.nodes
            if node.target == torch.ops.llama.update_cache.default
        )
        self.assertEqual(torch.ops.aten.sigmoid.default, update_node.args[0].target)
        self._assert_user_cache_writeback(
            lowered.executorch_program,
            [
                "aten.sigmoid.default",
                "et_vk.select_as_symint.default",
                "update_cache.default",
            ],
        )


def export_update_cache_model(output_path: str) -> None:
    """Export an update_cache model to .pte for the native runtime test.

    Shapes match the native test: value [1,S=2,H=2,D=4] into cache
    [1,Cmax=8,H=2,D=4] at input_pos=0. Example tensor *values* here are only for
    tracing; the native test supplies its own deterministic inputs at runtime.
    """
    S, H, D, Cmax = 2, 2, 4, 8
    model = UpdateCacheModule(0)
    value = torch.zeros(1, S, H, D)
    cache = torch.zeros(1, Cmax, H, D)
    ep = torch.export.export(model, (value, cache))
    et_program = to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)
    print(f"Exported {output_path}")


# (name, S, H, D, Cmax, input_pos) -- mirrors kCases in
# test/native/test_update_cache.cpp. Covers non-zero input_pos (the dst_offset
# path) and a second head_dim/n_heads shape. All writes stay in-bounds.
_NATIVE_CASES = [
    ("prefill", 2, 2, 4, 8, 0),
    ("offset", 2, 2, 4, 8, 5),
    ("shape_b", 3, 4, 8, 16, 0),
    ("shape_b_offset", 3, 4, 8, 16, 10),
]


def export_update_cache_cases(out_dir: str) -> None:
    """Export one .pte per native test case (input_pos baked).

    The native test supplies deterministic inputs and computes the integer-exact
    scatter reference inline, so only the .pte (shapes + input_pos baked) is
    written here -- no golden file.
    """
    os.makedirs(out_dir, exist_ok=True)
    for name, s, h, d, cmax, input_pos in _NATIVE_CASES:
        model = UpdateCacheModule(input_pos)
        value = torch.zeros(1, s, h, d)
        cache = torch.zeros(1, cmax, h, d)
        ep = torch.export.export(model, (value, cache))
        et_program = to_edge_transform_and_lower(
            ep, partitioner=[VulkanPartitioner()]
        ).to_executorch()
        with open(os.path.join(out_dir, f"{name}.pte"), "wb") as f:
            f.write(et_program.buffer)
        print(f"Exported {name}.pte (input_pos={input_pos})")


# (name, num_kv_heads, head_dim, seq_lens) -- mirrors the VulkanSDPATest param
# sets (sdpa_test.cpp:855-881). Cmax = sum(seq_lens) (exact fit). The native test
# threads the returned cache across steps as input_pos advances by seq_len.
_REPLAY_SEQS = [
    ("seqA", 4, 4, [3, 1, 1, 5, 1, 1, 2]),
    ("seqB", 2, 8, [3, 1, 1, 5, 1, 1]),
    ("llama3", 8, 128, [111, 1, 1, 1, 57, 1, 1]),
]


def export_update_cache_replay(out_dir: str) -> None:
    """Export one .pte per replay step (seq_len + input_pos baked).

    Mirrors Vulkan's multi-step advancing-input_pos cache accumulation; the
    native test feeds the returned cache into the next step and checks the
    integer-exact scatter golden after each write -- no golden file.
    """
    os.makedirs(out_dir, exist_ok=True)
    for name, h, d, seqs in _REPLAY_SEQS:
        cmax = sum(seqs)
        input_pos = 0
        for idx, s in enumerate(seqs):
            model = UpdateCacheModule(input_pos)
            value = torch.zeros(1, s, h, d)
            cache = torch.zeros(1, cmax, h, d)
            ep = torch.export.export(model, (value, cache))
            et_program = to_edge_transform_and_lower(
                ep, partitioner=[VulkanPartitioner()]
            ).to_executorch()
            fname = f"{name}_step{idx}_S{s}_pos{input_pos}.pte"
            with open(os.path.join(out_dir, fname), "wb") as f:
                f.write(et_program.buffer)
            print(f"Exported {fname}")
            input_pos += s


# (name, value_shape, cache_shape, dtype) -- each violates one runtime guard but
# still delegates to VulkanBackend at export (ATen's update_cache meta allows
# it). The WebGPU backend must reject each at graph build; the native test
# asserts a graceful delegate error (no crash, no silent-wrong output). The
# other guards (head_dim/n_heads mismatch, non-4D, out-of-bounds start_pos) are
# rejected by ATen at export, so they cannot be baked into a .pte.
_NEGATIVE_CASES = [
    ("neg_batch", (2, 2, 2, 4), (2, 8, 2, 4), torch.float32),  # batch must be 1
    ("neg_fp16", (1, 2, 2, 4), (1, 8, 2, 4), torch.float16),  # fp32-only
]


def export_update_cache_negative(out_dir: str) -> None:
    """Export guard-violating .pte's the WebGPU backend must reject at build.

    Asserts each still delegates to VulkanBackend, so the native test exercises
    the runtime guard rather than a CPU-fallback path.
    """
    os.makedirs(out_dir, exist_ok=True)
    for name, vshape, cshape, dtype in _NEGATIVE_CASES:
        model = UpdateCacheModule(0)
        value = torch.zeros(*vshape, dtype=dtype)
        cache = torch.zeros(*cshape, dtype=dtype)
        ep = torch.export.export(model, (value, cache))
        et_program = to_edge_transform_and_lower(
            ep, partitioner=[VulkanPartitioner()]
        ).to_executorch()
        delegated = any(
            d.id == "VulkanBackend"
            for plan in et_program.executorch_program.execution_plan
            for d in plan.delegates
        )
        if not delegated:
            raise RuntimeError(f"{name}: expected VulkanBackend delegation")
        with open(os.path.join(out_dir, f"{name}.pte"), "wb") as f:
            f.write(et_program.buffer)
        print(f"Exported {name}.pte")


def export_dynamic_update_cache(output_path: str) -> None:
    """Export the one-PTE dynamic position/sequence/capacity fixture."""
    _, program = _export_dynamic_update_cache()
    with open(output_path, "wb") as output:
        output.write(program.buffer)
    print(f"Exported {output_path}")


def export_intermediate_dynamic_update_cache(output_path: str) -> None:
    """Export the post-fixpoint intermediate-value runtime fixture."""
    _, program = _export_intermediate_dynamic_update_cache()
    with open(output_path, "wb") as output:
        output.write(program.buffer)
    print(f"Exported {output_path}")
