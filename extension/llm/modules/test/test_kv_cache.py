# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from typing import Callable, Tuple

import torch
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes.init_mutable_pass import InitializedMutableBufferPass

from executorch.extension.export_util.utils import save_pte_program
from executorch.extension.llm.modules.kv_cache import KVCache as InferenceKVCache

from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)
from executorch.runtime import Runtime
from torch.testing import assert_close
from torchtune.modules.kv_cache import KVCache


def generate_cache_inputs(
    seq_len: int,
    batch_size: int = 1,
    num_kv_heads: int = 64,
    head_dim: int = 8,
) -> Tuple[torch.Tensor, ...]:
    """Helper to generate k_val and v_val for both et and tt caches."""
    k_val = torch.ones(batch_size, seq_len, num_kv_heads, head_dim)
    v_val = torch.ones(batch_size, seq_len, num_kv_heads, head_dim)

    # For torchtune, the kv cache takes in transposed k and v.
    k_val_trans = k_val.transpose(1, 2)
    v_val_trans = v_val.transpose(1, 2)

    return (k_val, v_val, k_val_trans, v_val_trans)


class KVCacheTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 1
        self.max_seq_len = 10
        self.num_kv_heads = 1  # For testing purposes, usually this is 64.
        self.head_dim = 8
        self.dtype = torch.float

        self.tt_kv_cache = KVCache(
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
        )
        self.et_kv_cache = InferenceKVCache(
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            transpose_cache=False,
        )

    def _test_kv_cache(self, et_cache_module: Callable):
        """
        Given an executorch kv cache anywhere along the export chain, compare it's results
        against torchtune and run basic tests.
        """
        prefill_seq_len = 3
        k_val, v_val, k_val_trans, v_val_trans = generate_cache_inputs(
            prefill_seq_len, self.batch_size, self.num_kv_heads, self.head_dim
        )

        et_res = et_cache_module(k_val, v_val)
        tt_res = self.tt_kv_cache.update(k_val_trans, v_val_trans)
        tt_res_transposed = (tt_res[0].transpose(1, 2), tt_res[1].transpose(1, 2))

        # Check torchtune matches executorch.
        assert_close(et_res, tt_res_transposed)

        # Check the values are correct, all rows in the seq_len dim should be
        # filled with 1s up to and including the 3rd.
        et_k_cache = et_res[0]
        for i in range(prefill_seq_len):
            self.assertTrue(et_k_cache[0][i][0][0] == 1)
        self.assertTrue(et_k_cache[0][prefill_seq_len][0][0] == 0)

        """Case 2: Token-by-token (seq_len = 0)"""
        seq_len = 1
        k_val, v_val, k_val_trans, v_val_trans = generate_cache_inputs(
            seq_len, self.batch_size, self.num_kv_heads, self.head_dim
        )

        et_res = et_cache_module(k_val, v_val)
        tt_res = self.tt_kv_cache.update(k_val_trans, v_val_trans)
        tt_res_transposed = (tt_res[0].transpose(1, 2), tt_res[1].transpose(1, 2))

        # Check torchtune matches executorch.
        assert_close(tt_res_transposed, et_res)

        # All rows should be filled with 1s up to 3 + 1th row.
        et_k_cache = et_res[0]
        for i in range(prefill_seq_len + 1):
            self.assertTrue(et_k_cache[0][i][0][0] == 1)

        self.assertTrue(et_k_cache[0][prefill_seq_len + 1][0][0] == 0)

    def export_kv_cache(
        self,
        kv_cache: torch.nn.Module,
    ) -> torch.export.ExportedProgram:
        # Wrapper since torch.export only exports forward().
        class EtCacheWrapper(torch.nn.Module):
            def __init__(self, kv_cache: torch.nn.Module):
                super().__init__()
                self.kv_cache = kv_cache

            def forward(self, k_val: torch.Tensor, v_val: torch.Tensor):
                return self.kv_cache.update(k_val, v_val)

        dim = torch.export.Dim("seq_len_dim", min=1, max=self.max_seq_len)
        exported_kv_cache = torch.export.export(
            EtCacheWrapper(self.et_kv_cache),
            (
                torch.Tensor(self.batch_size, 3, self.num_kv_heads, self.head_dim),
                torch.Tensor(self.batch_size, 3, self.num_kv_heads, self.head_dim),
            ),  # 3 as example prefill seq_len.
            dynamic_shapes={
                "k_val": {
                    0: torch.export.Dim.STATIC,
                    1: dim,
                    2: torch.export.Dim.STATIC,
                    3: torch.export.Dim.STATIC,
                },
                "v_val": {
                    0: torch.export.Dim.STATIC,
                    1: dim,
                    2: torch.export.Dim.STATIC,
                    3: torch.export.Dim.STATIC,
                },
            },
            strict=True,
        )
        return exported_kv_cache

    def test_kv_cache_eager(self):
        self._test_kv_cache(self.et_kv_cache.update)

    def test_kv_cache_export(self):
        exported_kv_cache = self.export_kv_cache(self.et_kv_cache)
        self._test_kv_cache(exported_kv_cache.module())

    def test_kv_cache_edge(self):
        exported_kv_cache = self.export_kv_cache(self.et_kv_cache)
        edge_program = to_edge(
            exported_kv_cache,
            compile_config=EdgeCompileConfig(
                _core_aten_ops_exception_list=[torch.ops.aten._assert_async.msg],
                _check_ir_validity=False,
            ),
        )
        self._test_kv_cache(edge_program._edge_programs["forward"].module())

    def test_kv_cache_executorch(self):
        exported_kv_cache = self.export_kv_cache(self.et_kv_cache)
        edge_program = to_edge(
            exported_kv_cache,
            compile_config=EdgeCompileConfig(
                _core_aten_ops_exception_list=[torch.ops.aten._assert_async.msg],
                _check_ir_validity=False,
            ),
        )
        et_config = ExecutorchBackendConfig(
            passes=[InitializedMutableBufferPass(["kv_cache_pos"])],
        )
        et_program = edge_program.to_executorch(config=et_config)

        runtime = Runtime.get()
        program = runtime.load_program(et_program.buffer)
        method = program.load_method("forward")

        # Since method.execute expects a tuple of args.
        def wrapped_callable(k_val: torch.Tensor, v_val: torch.Tensor) -> torch.Tensor:
            return method.execute((k_val, v_val))

        self._test_kv_cache(wrapped_callable)

    def test_kv_cache_executorch_from_file(self):
        exported_kv_cache = self.export_kv_cache(self.et_kv_cache)
        edge_program = to_edge(
            exported_kv_cache,
            compile_config=EdgeCompileConfig(
                _core_aten_ops_exception_list=[torch.ops.aten._assert_async.msg],
                _check_ir_validity=False,
            ),
        )
        et_config = ExecutorchBackendConfig(
            passes=[InitializedMutableBufferPass(["kv_cache_pos"])],
        )
        et_program = edge_program.to_executorch(config=et_config)

        with tempfile.TemporaryDirectory() as tempdir:
            pte_path = save_pte_program(et_program, "test_et_kv_cache", tempdir)
            with open(pte_path, "rb") as f:
                model_bytes = f.read()
            loaded_et_program = _load_for_executorch_from_buffer(model_bytes)

            # Since method.execute expects a tuple of args.
            def wrapped_callable(
                k_val: torch.Tensor, v_val: torch.Tensor
            ) -> torch.Tensor:
                return loaded_et_program.forward((k_val, v_val))

            self._test_kv_cache(wrapped_callable)
