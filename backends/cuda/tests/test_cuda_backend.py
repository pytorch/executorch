# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.exir.backend.compile_spec_schema import CompileSpec


class CudaBackendTest(unittest.TestCase):
    def test_low_memory_mode_materializes_lazy_jit_constants(self):
        from torch._inductor.graph import GraphLowering

        first = torch.empty_strided((4, 8), (8, 1))
        second = torch.empty_strided((4, 8), (8, 1))
        first.untyped_storage().resize_(0)
        second.untyped_storage().resize_(0)
        ordinary = torch.ones(2)
        original_constants = {
            "first": first,
            "second": second,
            "ordinary": ordinary,
        }
        graph = SimpleNamespace(constants=original_constants)

        def inspect_constants(graph_lowering, *args):
            self.assertIsNot(graph_lowering.constants, original_constants)
            self.assertIs(graph_lowering.constants["ordinary"], ordinary)
            for name in ("first", "second"):
                value = graph_lowering.constants[name]
                self.assertEqual((4, 8), value.shape)
                self.assertEqual((8, 1), value.stride())
                self.assertEqual(first.device, value.device)
                self.assertEqual(
                    value.numel() * value.element_size(),
                    value.untyped_storage().nbytes(),
                )
                self.assertEqual(0, torch.count_nonzero(value))
            self.assertEqual(
                graph_lowering.constants["first"].data_ptr(),
                graph_lowering.constants["second"].data_ptr(),
            )
            raise RuntimeError("stop after inspection")

        with mock.patch.object(
            GraphLowering,
            "_run_jit_variant_for_autotune",
            inspect_constants,
        ):
            with CudaBackend.get_extra_aoti_compile_context_manager(
                [CompileSpec(key="low_memory_mode", value=b"ON")]
            ):
                with self.assertRaisesRegex(RuntimeError, "stop after inspection"):
                    GraphLowering._run_jit_variant_for_autotune(
                        graph, None, None, None, []
                    )

        self.assertIs(graph.constants, original_constants)
        self.assertIs(graph.constants["first"], first)
        self.assertIs(graph.constants["second"], second)
        self.assertEqual(0, first.untyped_storage().nbytes())
        self.assertEqual(0, second.untyped_storage().nbytes())

    def test_low_memory_mode_clones_emptied_mutated_input(self):
        from torch._inductor import compile_fx

        emptied = torch.empty_strided((4, 8), (8, 1))
        emptied.untyped_storage().resize_(0)

        def extract_real_inputs():
            return compile_fx.clone_preserve_strides(emptied)

        with CudaBackend.get_extra_aoti_compile_context_manager(
            [CompileSpec(key="low_memory_mode", value=b"ON")]
        ):
            clone = extract_real_inputs()

        self.assertEqual(emptied.shape, clone.shape)
        self.assertEqual(emptied.stride(), clone.stride())
        self.assertEqual(emptied.device, clone.device)
        self.assertEqual(
            clone.numel() * clone.element_size(), clone.untyped_storage().nbytes()
        )
        self.assertEqual(0, torch.count_nonzero(clone))
