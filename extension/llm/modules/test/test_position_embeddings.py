# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest

import torch
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.extension.llm.modules import (
    replace_tile_positional_embedding,
    replace_tiled_token_positional_embedding,
    TiledTokenPositionalEmbedding,
    TilePositionalEmbedding,
)
from executorch.runtime import Runtime
from torch._inductor.package import load_package, package_aoti
from torch.testing import assert_close
from torchtune.models.clip import (
    TiledTokenPositionalEmbedding as TuneTiledTokenPositionalEmbedding,
    TilePositionalEmbedding as TuneTilePositionalEmbedding,
)


class TilePositionalEmbeddingTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.tpe = TilePositionalEmbedding(4, 1280)
        self.ref_tpe = TuneTilePositionalEmbedding(4, 1280)
        self.x = torch.randn(1, 4, 1600, 1280)
        self.aspect_ratio = torch.tensor([[1, 1]])
        num_tiles_dim = torch.export.Dim("num_tiles", min=1, max=4)
        num_tokens = torch.export.Dim("num_tokens", min=1, max=1600)

        self.dynamic_shape = {
            0: 1,  # batch
            1: num_tiles_dim,  # num tiles
            2: num_tokens,  # num tokens
            3: 1280,  # embedding dim
        }

    def test_tile_positional_embedding_smoke(self):
        y = self.tpe(self.x, self.aspect_ratio)
        ref_y = self.ref_tpe(self.x, self.aspect_ratio)

        self.assertTrue(torch.allclose(y, ref_y))

    def test_tile_positional_embedding_export(self):

        tpe_ep = torch.export.export(
            self.tpe,
            (self.x, self.aspect_ratio),
            dynamic_shapes=(
                self.dynamic_shape,
                None,
            ),  # assuming aspect ratio is static
        )

        y = tpe_ep.module()(self.x, self.aspect_ratio)
        ref_y = self.ref_tpe(self.x, self.aspect_ratio)

        self.assertTrue(torch.allclose(y, ref_y))

    def test_tile_positional_embedding_aoti(self):
        so = torch._export.aot_compile(
            self.tpe,
            args=(self.x, self.aspect_ratio),
            options={"aot_inductor.package": True},
            dynamic_shapes=(
                self.dynamic_shape,
                None,
            ),  # assuming aspect ratio is static
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = package_aoti(os.path.join(tmpdir, "tpe.pt2"), so)
            tpe_aoti = load_package(path)

            y = tpe_aoti(self.x, self.aspect_ratio)
            ref_y = self.ref_tpe(self.x, self.aspect_ratio)

            self.assertTrue(torch.allclose(y, ref_y))

    def test_tile_positional_embedding_et(self):
        tpe_ep = torch.export.export(
            self.tpe,
            (self.x, self.aspect_ratio),
            dynamic_shapes=(
                self.dynamic_shape,
                None,
            ),  # assuming aspect ratio is static
        )
        et_program = to_edge(
            tpe_ep,
            compile_config=EdgeCompileConfig(
                _core_aten_ops_exception_list=[
                    torch.ops.aten.sym_constrain_range_for_size.default,
                    torch.ops.aten._assert_scalar.default,
                    torch.ops.aten._local_scalar_dense.default,
                ]
            ),
        ).to_executorch()
        runtime = Runtime.get()
        program = runtime.load_program(et_program.buffer)
        method = program.load_method("forward")
        y = method.execute((self.x, self.aspect_ratio))
        ref_y = self.ref_tpe(self.x, self.aspect_ratio)

        self.assertTrue(torch.allclose(y[0], ref_y))

    def test_replace_tile_positional_embedding(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tpe = TuneTilePositionalEmbedding(4, 1280)

            def forward(self, x, aspect_ratio):
                return self.tpe(x, aspect_ratio)

        m = Module()
        m = replace_tile_positional_embedding(m)
        self.assertTrue(isinstance(m.tpe, TilePositionalEmbedding))


class TiledTokenPositionalEmbeddingTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.tpe = TiledTokenPositionalEmbedding(4, 1280, 40, 1)
        self.ref_tpe = TuneTiledTokenPositionalEmbedding(4, 1280, 40, 1)
        self.tpe.load_state_dict(self.ref_tpe.state_dict())
        self.x = torch.randn(1, 4, 1601, 1280)
        self.aspect_ratio = torch.tensor([[1, 2]])
        num_tiles_dim = torch.export.Dim("num_tiles", min=1, max=4)

        self.dynamic_shape = {
            0: 1,  # batch
            1: num_tiles_dim,  # num tiles
            2: 1601,  # num tokens
            3: 1280,  # embedding dim
        }

    def test_tiled_token_positional_embedding_smoke(self):
        y = self.tpe(self.x, self.aspect_ratio)
        ref_y = self.ref_tpe(self.x, self.aspect_ratio)

        assert_close(y, ref_y)

    def test_tiled_token_positional_embedding_export(self):

        tpe_ep = torch.export.export(
            self.tpe,
            (self.x, self.aspect_ratio),
            dynamic_shapes=(
                self.dynamic_shape,
                None,
            ),  # assuming aspect ratio is static
        )

        y = tpe_ep.module()(self.x, self.aspect_ratio)
        ref_y = self.ref_tpe(self.x, self.aspect_ratio)

        assert_close(y, ref_y)

    @unittest.skip(reason="TODO(T207740932): test is flaky")
    def test_tiled_token_positional_embedding_aoti(self):
        tpe_ep = torch.export.export(
            self.tpe,
            (self.x, self.aspect_ratio),
            dynamic_shapes=(
                self.dynamic_shape,
                None,
            ),  # assuming aspect ratio is static
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = torch._inductor.aoti_compile_and_package(
                tpe_ep,
                package_path=os.path.join(tmpdir, "tpe.pt2"),
            )
            tpe_aoti = load_package(path)

            y = tpe_aoti(self.x, self.aspect_ratio)
            ref_y = self.ref_tpe(self.x, self.aspect_ratio)

            assert_close(y, ref_y)

    def test_tiled_token_positional_embedding_et(self):
        tpe_ep = torch.export.export(
            self.tpe,
            (self.x, self.aspect_ratio),
            dynamic_shapes=(
                self.dynamic_shape,
                None,
            ),  # assuming aspect ratio is static
        )
        et_program = to_edge(
            tpe_ep,
            compile_config=EdgeCompileConfig(
                _core_aten_ops_exception_list=[
                    torch.ops.aten.sym_constrain_range_for_size.default,
                    torch.ops.aten._assert_scalar.default,
                    torch.ops.aten._local_scalar_dense.default,
                ]
            ),
        ).to_executorch()
        runtime = Runtime.get()
        program = runtime.load_program(et_program.buffer)
        method = program.load_method("forward")
        y = method.execute((self.x, self.aspect_ratio))
        ref_y = self.ref_tpe(self.x, self.aspect_ratio)

        assert_close(y[0], ref_y)

    def test_replace_tiled_token_positional_embedding(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tpe = TuneTiledTokenPositionalEmbedding(4, 1280, 40, 1)

            def forward(self, x, aspect_ratio):
                return self.tpe(x, aspect_ratio)

        m = Module()
        m = replace_tiled_token_positional_embedding(m)
        self.assertTrue(isinstance(m.tpe, TiledTokenPositionalEmbedding))
