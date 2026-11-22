# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ast
import contextlib
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from executorch.backends.cuda.runtime_owned_kv import RuntimeOwnedKVCompileBuffers


PREFIX = "__et_offgraph_kv_"


def program(**buffers):
    return types.SimpleNamespace(state_dict={}, constants=buffers)


def empty_view(shape, stride=None, offset=0):
    if stride is None:
        value = torch.empty(shape, dtype=torch.bfloat16)
    else:
        required = offset + 1 + sum((n - 1) * s for n, s in zip(shape, stride))
        value = torch.as_strided(
            torch.empty(required, dtype=torch.bfloat16), shape, stride, offset
        )
    value.untyped_storage().resize_(0)
    return value


class RuntimeOwnedKVCompileBuffersTest(unittest.TestCase):
    def test_production_view_clones_without_materializing_cpu_zeros(self):
        source = empty_view((1, 131072 * 65, 2, 128))
        buffers = RuntimeOwnedKVCompileBuffers(
            program(**{PREFIX + "layer_0_k": source}), PREFIX
        )
        with mock.patch.object(torch, "zeros", side_effect=AssertionError("zeros")):
            clone = buffers.cpu_placeholder(source)
        self.assertEqual(clone.shape, source.shape)
        self.assertEqual(clone.stride(), source.stride())
        self.assertEqual(clone.dtype, source.dtype)
        self.assertEqual(clone.device.type, "cpu")
        self.assertEqual(clone.untyped_storage().nbytes(), 0)
        self.assertTrue(buffers.owns_empty(clone))

    def test_placeholder_preserves_offset_and_noncontiguous_strides(self):
        source = empty_view((2, 3), (9, 2), 4)
        buffers = RuntimeOwnedKVCompileBuffers(
            program(**{PREFIX + "layer_0_k": source}), PREFIX
        )
        clone = buffers.cpu_placeholder(source)
        self.assertEqual(clone.storage_offset(), 4)
        self.assertEqual(clone.stride(), (9, 2))
        self.assertEqual(clone.shape, (2, 3))
        self.assertEqual(clone.untyped_storage().nbytes(), 0)

    def test_same_shape_graph_owned_cache_is_not_runtime_owned(self):
        owned, graph_owned = empty_view((2, 3)), empty_view((2, 3))
        self.assertEqual(owned.data_ptr(), graph_owned.data_ptr())
        buffers = RuntimeOwnedKVCompileBuffers(
            program(**{PREFIX + "layer_0_k": owned, "kv_cache.k": graph_owned}),
            PREFIX,
        )
        self.assertTrue(buffers.owns_empty(owned))
        self.assertFalse(buffers.owns_empty(graph_owned))
        with self.assertRaises(ValueError):
            buffers.cpu_placeholder(graph_owned)

    def test_graph_owned_alias_excludes_shared_storage(self):
        owned = empty_view((2, 3))
        buffers = RuntimeOwnedKVCompileBuffers(
            program(**{PREFIX + "layer_0_k": owned, "kv_cache.alias": owned}),
            PREFIX,
        )
        self.assertFalse(buffers.owns_empty(owned))

    def test_real_runtime_metadata_preserves_nonzero_payload(self):
        value = torch.tensor([4, 9], dtype=torch.int32)
        buffers = RuntimeOwnedKVCompileBuffers(
            program(**{PREFIX + "owner_step": value}), PREFIX
        )
        self.assertFalse(buffers.owns_empty(value))
        torch.testing.assert_close(value, torch.tensor([4, 9], dtype=torch.int32))

    def test_program_identity_and_placeholder_lifetime_do_not_leak(self):
        value = empty_view((2, 3))
        first = RuntimeOwnedKVCompileBuffers(
            program(**{PREFIX + "layer_0_k": value}), PREFIX
        )
        clone = first.cpu_placeholder(value)
        second = RuntimeOwnedKVCompileBuffers(program(), PREFIX)
        self.assertFalse(second.owns_empty(value))
        self.assertFalse(second.owns_empty(clone))


class ConstantInfoCodegenTest(unittest.TestCase):
    @staticmethod
    @contextlib.contextmanager
    def lowered_constants(tensors):
        from torch._inductor import config
        from torch._inductor.graph import GraphLowering
        from torch._inductor.virtualized import V

        graph = torch.fx.Graph()
        graph.output(())
        module = torch.fx.GraphModule({}, graph)
        with config.patch({"cpp_wrapper": True}):
            lowering = GraphLowering(
                module, (), cpp_wrapper=True, aot_mode=True, is_inference=True
            )
            with V.set_graph_handler(lowering):
                lowering.run()
                lowering.constants = {
                    f"constant_{index}": value for index, value in enumerate(tensors)
                }
                lowering.allocated_constant_name = {
                    name: PREFIX + name for name in lowering.constants
                }
                yield lowering

    def test_real_inductor_constructor_preserves_extent_without_payload(self):
        from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu

        owned = empty_view((1, 131072 * 65, 2, 128))
        graph_owned = empty_view((2, 3))
        payload = torch.tensor([4, 9], dtype=torch.int32)
        buffers = RuntimeOwnedKVCompileBuffers(
            program(**{PREFIX + "layer_0_k": owned, "graph": graph_owned}), PREFIX
        )
        for alignment in (1, 64):
            with self.subTest(alignment=alignment), self.lowered_constants(
                [owned, graph_owned, payload]
            ):
                original = CppWrapperCpu()
                original.codegen_model_constructor()
                self.assertIn(
                    "constants_info_[0].data_size = 0;", original.prefix.getvalue()
                )
                wrapper = CppWrapperCpu()
                prefix = wrapper.prefix
                with mock.patch.object(
                    torch, "zeros", side_effect=AssertionError("zeros")
                ):
                    with buffers.constant_info_sizes(
                        wrapper, [owned, graph_owned, payload], alignment
                    ):
                        wrapper.codegen_model_constructor()
                self.assertIs(wrapper.prefix, prefix)
                expected = owned.numel() * owned.element_size()
                generated = wrapper.prefix.getvalue()
                self.assertIn(f"constants_info_[0].data_size = {expected};", generated)
                self.assertIn("constants_info_[1].data_size = 0;", generated)
                self.assertIn("constants_info_[2].data_size = 64;", generated)
                self.assertIn(
                    "constants_info_[0].shape = {1, 8519680, 2, 128};", generated
                )
                self.assertEqual(owned.untyped_storage().nbytes(), 0)
                torch.testing.assert_close(
                    payload, torch.tensor([4, 9], dtype=torch.int32)
                )

    def test_real_constructor_preserves_noncontiguous_extent_and_offset(self):
        from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu

        value = empty_view((2, 3), (9, 2), 4)
        buffers = RuntimeOwnedKVCompileBuffers(
            program(**{PREFIX + "layer_0_k": value}), PREFIX
        )
        for alignment, expected in ((1, 36), (64, 64)):
            with self.subTest(alignment=alignment), self.lowered_constants([value]):
                wrapper = CppWrapperCpu()
                with buffers.constant_info_sizes(wrapper, [value], alignment):
                    wrapper.codegen_model_constructor()
                generated = wrapper.prefix.getvalue()
                self.assertIn(f"constants_info_[0].data_size = {expected};", generated)
                self.assertIn("constants_info_[0].offset = 4;", generated)
                self.assertIn("constants_info_[0].stride = {9, 2};", generated)
                self.assertEqual(value.untyped_storage().nbytes(), 0)

    def test_constant_sizes_preserve_prefix_and_reject_changed_emission(self):
        from torch._inductor.utils import IndentedBuffer

        value = empty_view((2, 3))
        buffers = RuntimeOwnedKVCompileBuffers(
            program(**{PREFIX + "layer_0_k": value}), PREFIX
        )
        for emitted in (None, "constants_info_[0].data_size = 17;"):
            wrapper = types.SimpleNamespace(prefix=IndentedBuffer())
            original = wrapper.prefix
            with self.subTest(emitted=emitted), self.assertRaises(RuntimeError):
                with buffers.constant_info_sizes(wrapper, [value], 64):
                    if emitted is not None:
                        wrapper.prefix.writeline(emitted)
            self.assertIs(wrapper.prefix, original)
        wrapper = types.SimpleNamespace(prefix=IndentedBuffer())
        original = wrapper.prefix
        with self.assertRaisesRegex(RuntimeError, "original failure"):
            with buffers.constant_info_sizes(wrapper, [value], 64):
                raise RuntimeError("original failure")
        self.assertIs(wrapper.prefix, original)

    @unittest.skipUnless(torch.cuda.is_available(), "Actual CUDA compile context")
    def test_actual_backend_context_installs_codegen_hook_and_restores_nested(self):
        from executorch.backends.cuda.cuda_backend import _CPU_CLONE_GUARD, CudaBackend
        from executorch.exir.backend.compile_spec_schema import CompileSpec
        from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu

        value = empty_view((2, 3))
        lowered = program(**{PREFIX + "layer_0_k": value})
        original = CppWrapperCpu.codegen_model_constructor
        previous = getattr(_CPU_CLONE_GUARD, "runtime_owned_kv", None)
        specs = [CompileSpec("low_memory_mode", b"ON")]
        with self.lowered_constants([value]):
            with CudaBackend.get_aoti_program_compile_context_manager(lowered, specs):
                with CudaBackend.get_aoti_program_compile_context_manager(
                    lowered, specs
                ):
                    wrapper = CppWrapperCpu()
                    wrapper.codegen_model_constructor()
                    self.assertIn(
                        "constants_info_[0].data_size = 64;", wrapper.prefix.getvalue()
                    )
                self.assertTrue(_CPU_CLONE_GUARD.runtime_owned_kv.owns_empty(value))
            self.assertIs(CppWrapperCpu.codegen_model_constructor, original)
            self.assertIs(_CPU_CLONE_GUARD.runtime_owned_kv, previous)
            wrapper = CppWrapperCpu()
            wrapper.codegen_model_constructor()
            self.assertIn(
                "constants_info_[0].data_size = 0;", wrapper.prefix.getvalue()
            )


class ProgramCompileContextTest(unittest.TestCase):
    @staticmethod
    def method(path, class_name, name, namespace):
        tree = ast.parse(path.read_text())
        cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
        method = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        method.decorator_list = []
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        return namespace[name]

    def test_existing_compile_spec_hook_is_still_called(self):
        root = Path(__file__).parents[2]
        method = self.method(
            root / "aoti/aoti_backend.py",
            "AotiBackend",
            "get_aoti_program_compile_context_manager",
            {},
        )
        seen = []

        class ExistingSubclass:
            @classmethod
            def get_extra_aoti_compile_context_manager(cls, specs):
                seen.append(specs)
                return contextlib.nullcontext("existing")

        specs = object()
        with method(ExistingSubclass, object(), specs) as result:
            self.assertEqual(result, "existing")
        self.assertEqual(seen, [specs])

    def test_cuda_scope_uses_final_program_and_restores_after_exception(self):
        root = Path(__file__).parents[2]
        guard = threading.local()
        namespace = {
            "_CPU_CLONE_GUARD": guard,
            "RuntimeOwnedKVCompileBuffers": RuntimeOwnedKVCompileBuffers,
            "OFFGRAPH_KV_FQN_PREFIX": PREFIX,
        }
        method = contextlib.contextmanager(
            self.method(
                root / "cuda/cuda_backend.py",
                "CudaBackend",
                "get_aoti_program_compile_context_manager",
                namespace,
            )
        )
        value = empty_view((2, 3))
        lowered = program(**{PREFIX + "layer_0_k": value})
        sentinel = object()
        guard.runtime_owned_kv = sentinel

        class ExistingContext:
            @classmethod
            def get_extra_aoti_compile_context_manager(cls, specs):
                return contextlib.nullcontext()

        with self.assertRaisesRegex(RuntimeError, "abort"):
            with method(ExistingContext, lowered, []):
                self.assertTrue(guard.runtime_owned_kv.owns_empty(value))
                with method(ExistingContext, program(), []):
                    self.assertFalse(guard.runtime_owned_kv.owns_empty(value))
                self.assertTrue(guard.runtime_owned_kv.owns_empty(value))
                raise RuntimeError("abort")
        self.assertIs(guard.runtime_owned_kv, sentinel)

    def test_actual_cpu_clone_and_serialization_hooks_keep_owned_views_empty(self):
        source = Path(__file__).parents[1] / "cuda_backend.py"
        tree = ast.parse(source.read_text())
        context = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_compile_time_cpu_clones"
        )
        names = {
            "_full_zeros_preserving_strides",
            "_is_emptied",
            "_cpu_clone_preserve_strides",
            "_get_const_synthesize_zeros",
        }
        functions = [
            node
            for node in (*tree.body, *context.body)
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        self.assertEqual(len(functions), len(names))
        guard = threading.local()
        namespace = {
            "torch": torch,
            "sys": sys,
            "_CPU_CLONE_GUARD": guard,
            "orig_clone": lambda tensor: tensor.clone(),
            "orig_get_const": lambda module, name: module[name],
        }
        exec(
            compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
            namespace,
        )
        owned, graph_owned = empty_view((2, 3)), empty_view((2, 3))
        guard.runtime_owned_kv = RuntimeOwnedKVCompileBuffers(
            program(**{PREFIX + "layer_0_k": owned, "kv_cache.k": graph_owned}),
            PREFIX,
        )

        def _unlift_graph(tensor):
            return namespace["_cpu_clone_preserve_strides"](tensor)

        clone = _unlift_graph(owned)
        self.assertEqual(clone.untyped_storage().nbytes(), 0)
        self.assertIs(
            namespace["_get_const_synthesize_zeros"]({"k": clone}, "k"), clone
        )
        graph_clone = _unlift_graph(graph_owned)
        torch.testing.assert_close(
            graph_clone, torch.zeros((2, 3), dtype=torch.bfloat16)
        )
        graph_constant = namespace["_get_const_synthesize_zeros"](
            {"k": graph_owned}, "k"
        )
        torch.testing.assert_close(graph_constant, graph_clone)
        payload = torch.tensor([4, 9], dtype=torch.int32)
        torch.testing.assert_close(_unlift_graph(payload), payload)

    def test_preprocess_passes_final_program_to_context_hook(self):
        source = Path(__file__).parents[2] / "aoti/aoti_backend.py"
        tree = ast.parse(source.read_text())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get_aoti_program_compile_context_manager"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            [ast.unparse(arg) for arg in calls[0].args],
            ["device_edge_program", "compile_specs"],
        )


if __name__ == "__main__":
    unittest.main()
