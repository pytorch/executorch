# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest
from pathlib import Path

import torch
from executorch.examples.models.muse_glimmer.source_transformations.sampler import (
    _argmax_tile,
    argmax_last_dim,
    sample,
    sample_token,
)
from torch._inductor.utils import fresh_cache, run_and_get_code


def _reference_sample(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    logits = logits / temperature.clamp(min=1e-6)
    noise = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
    return (logits + gumbel).argmax(dim=-1, keepdim=True).float()


def _gather_tile_argmax(values: torch.Tensor) -> torch.Tensor:
    """A tiled argmax whose offset is an indirect load (``gather``)."""
    tile_max, tile_arg = values.view(-1, 64, 64).max(dim=-1)
    best_tile = tile_max.argmax(dim=-1, keepdim=True)
    return best_tile * 64 + tile_arg.gather(-1, best_tile)


class _Function(torch.nn.Module):
    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return self.fn(*args)


def _aoti_compile(fn, args, package_path: str):
    """Exports ``fn`` and compiles it with AOTInductor under the CUDA backend's
    compile options and context, leaving out its packaging and linking options
    so that torch can load the package. Returns the loaded package and the
    source of every kernel generated for it."""
    # Imported here so the CPU tests also run where the CUDA backend is absent.
    from executorch.backends.cuda.cuda_backend import CudaBackend

    options = {
        key: value
        for key, value in CudaBackend.get_aoti_compile_options([]).items()
        if not key.startswith("aot_inductor.")
    }
    with fresh_cache(), CudaBackend.get_extra_aoti_compile_context_manager([]):
        program = torch.export.export(_Function(fn), args)
        package = torch._inductor.aoti_compile_and_package(
            program, package_path=package_path, inductor_configs=options
        )
        cache_dir = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
        sources = [path.read_text() for path in cache_dir.rglob("*.py")]
    return torch._inductor.aoti_load_package(package), sources


def _edge_case_logits() -> list[torch.Tensor]:
    """Random logits, ties across tiles, an all -inf row and NaN, on CUDA."""
    vocab = 202048
    tile = _argmax_tile(vocab)
    torch.manual_seed(0)
    ties = torch.zeros(1, vocab, device="cuda")
    ties[0, [3 * tile + 7, tile + 9, vocab - 1]] = 1.0
    nan = torch.full((1, vocab), float("-inf"), device="cuda")
    nan[0, [150000, 7000]] = float("nan")
    return [
        torch.randn(1, vocab, device="cuda") * 5,
        ties,
        torch.full((1, vocab), float("-inf"), device="cuda"),
        nan,
    ]


class ArgmaxLastDimTest(unittest.TestCase):
    def test_matches_argmax(self) -> None:
        torch.manual_seed(0)
        for rows, vocab in ((1, 202048), (3, 5000), (2, 1024), (1, 1000), (2, 1)):
            values = torch.randn(rows, vocab)
            self.assertTrue(
                torch.equal(
                    argmax_last_dim(values), values.argmax(dim=-1, keepdim=True)
                ),
                f"rows={rows} vocab={vocab}",
            )

    def test_leading_dims(self) -> None:
        torch.manual_seed(0)
        for shape in ((2, 3, 5000), (4000,)):
            values = torch.randn(*shape)
            self.assertTrue(
                torch.equal(
                    argmax_last_dim(values), values.argmax(dim=-1, keepdim=True)
                ),
                f"shape={shape}",
            )

    def test_ties_resolve_to_first_index(self) -> None:
        tile = _argmax_tile(3000)
        values = torch.zeros(3, 3000)
        values[0, [5, 1500, 2999]] = 1.0
        values[1, [2 * tile, tile, tile + 6]] = 1.0
        expected = values.argmax(dim=-1, keepdim=True)
        self.assertEqual(expected.flatten().tolist(), [5, tile, 0])
        self.assertTrue(torch.equal(argmax_last_dim(values), expected))

    def test_tile_balances_the_two_reductions(self) -> None:
        for vocab in (1, 2, 3000, 202048):
            tile = _argmax_tile(vocab)
            tiles = -(-vocab // tile)
            self.assertEqual(tile & (tile - 1), 0, f"vocab={vocab}")
            self.assertGreaterEqual(tile * tile, vocab, f"vocab={vocab}")
            self.assertLess(tile * tile, 4 * vocab, f"vocab={vocab}")
            self.assertLessEqual(tiles, tile, f"vocab={vocab}")

    def test_negative_infinity_and_nan(self) -> None:
        values = torch.full((2, 2500), float("-inf"))
        values[1, [2100, 1800]] = float("nan")
        expected = values.argmax(dim=-1, keepdim=True)
        self.assertEqual(expected.flatten().tolist(), [0, 1800])
        self.assertTrue(torch.equal(argmax_last_dim(values), expected))

    def test_sample_matches_single_reduction(self) -> None:
        logits = torch.randn(1, 202048) * 5.0
        for temperature in (0.7, 1.0):
            t = torch.tensor([temperature])
            torch.manual_seed(1234)
            got = sample(logits, t)
            torch.manual_seed(1234)
            expected = _reference_sample(logits, t)
            self.assertTrue(torch.equal(got, expected), f"temperature={temperature}")

    def test_zero_temperature_is_argmax(self) -> None:
        logits = torch.randn(2, 202048)
        logits[0, 7] = logits[0, 200001] = logits[0].max() + 1.0
        logits[1, 150000] = logits[1].max() + 1e-6
        expected = logits.argmax(dim=-1, keepdim=True)
        self.assertEqual(expected.flatten().tolist(), [7, 150000])
        for seed in range(4):
            torch.manual_seed(seed)
            got = sample_token(logits, torch.tensor([0.0]))
            self.assertTrue(torch.equal(got, expected), f"seed={seed}")

    def test_temperature_just_above_zero_samples(self) -> None:
        logits = torch.zeros(1, 5000)
        logits[0, [3, 4000]] = 1.0
        for temperature in (0.0, 1e-6, 2e-6):
            picks = set()
            for seed in range(16):
                torch.manual_seed(seed)
                picks.add(sample_token(logits, torch.tensor([temperature])).item())
            expected = {3} if temperature == 0.0 else {3, 4000}
            self.assertEqual(picks, expected, f"temperature={temperature}")


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class CompiledSamplerCudaTest(unittest.TestCase):
    """``sample_token`` compiled by Inductor on CUDA, as the export runs it."""

    def setUp(self) -> None:
        torch._dynamo.reset()
        self.sample_token = torch.compile(sample_token)
        self.greedy = torch.tensor([0.0], device="cuda")

    def assert_greedy_matches_argmax(self, logits: torch.Tensor) -> None:
        # Compiles afresh, so the generated code of every call is checked.
        with torch._inductor.config.patch(fx_graph_cache=False):
            got, codes = run_and_get_code(self.sample_token, logits, self.greedy)
        self.assertTrue(torch.equal(got, logits.argmax(dim=-1, keepdim=True)))
        self.assertTrue(any("@triton.jit" in code for code in codes))
        for code in codes:
            self.assertNotIn("device_assert", code)

    def test_random(self) -> None:
        torch.manual_seed(0)
        self.assert_greedy_matches_argmax(torch.randn(1, 202048, device="cuda") * 5)

    def test_ties_across_tiles(self) -> None:
        tile = _argmax_tile(202048)
        for dtype in (torch.float32, torch.bfloat16):
            logits = torch.zeros(1, 202048, device="cuda", dtype=dtype)
            logits[0, [3 * tile + 7, tile + 9, 202047]] = 1.0
            self.assert_greedy_matches_argmax(logits)
            self.assertEqual(logits.argmax().item(), tile + 9)

    def test_negative_infinity_and_nan(self) -> None:
        logits = torch.full((1, 202048), float("-inf"), device="cuda")
        self.assert_greedy_matches_argmax(logits)
        logits[0, [150000, 7000]] = float("nan")
        self.assert_greedy_matches_argmax(logits)

    def test_device_assert_check_sees_an_indirect_load(self) -> None:
        # The checks above are only meaningful if Inductor's guard on an
        # indirect load still reads "device_assert" in the generated code.
        with torch._inductor.config.patch(fx_graph_cache=False):
            _, codes = run_and_get_code(
                torch.compile(_gather_tile_argmax),
                torch.randn(1, 4096, device="cuda"),
            )
        self.assertTrue(any("device_assert" in code for code in codes))

    @torch._inductor.config.patch(fallback_random=True)
    def test_positive_temperature_matches_single_reduction(self) -> None:
        # fallback_random draws the same noise as eager under the same seed.
        torch.manual_seed(0)
        logits = torch.randn(1, 202048, device="cuda") * 5.0
        for temperature in (0.7, 1.0):
            t = torch.tensor([temperature], device="cuda")
            for seed in range(4):
                torch.manual_seed(seed)
                got = self.sample_token(logits, t)
                torch.manual_seed(seed)
                expected = _reference_sample(logits, t).long()
                self.assertTrue(
                    torch.equal(got, expected),
                    f"temperature={temperature} seed={seed}",
                )


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class ExportedSamplerCudaTest(unittest.TestCase):
    """``sample_token`` exported and compiled ahead of time with AOTInductor
    under the CUDA backend's compile options."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.workdir = tempfile.TemporaryDirectory()
        cls.sampler, cls.sources = _aoti_compile(
            sample_token,
            (torch.zeros(1, 202048, device="cuda"), torch.zeros(1, device="cuda")),
            os.path.join(cls.workdir.name, "sampler.pt2"),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.workdir.cleanup()

    def test_greedy_matches_argmax(self) -> None:
        greedy = torch.zeros(1, device="cuda")
        for logits in _edge_case_logits():
            expected = logits.argmax(dim=-1, keepdim=True)
            for _ in range(2):
                self.assertTrue(torch.equal(self.sampler(logits, greedy), expected))

    def test_generated_code_has_no_device_assert(self) -> None:
        self.assertTrue(any("@triton.jit" in source for source in self.sources))
        for source in self.sources:
            self.assertNotIn("device_assert", source)

    def test_device_assert_check_sees_an_indirect_load(self) -> None:
        _, sources = _aoti_compile(
            _gather_tile_argmax,
            (torch.zeros(1, 4096, device="cuda"),),
            os.path.join(self.workdir.name, "gather.pt2"),
        )
        self.assertTrue(any("device_assert" in source for source in sources))

    def test_positive_temperature(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(1, 202048, device="cuda")
        got = self.sampler(logits, torch.ones(1, device="cuda"))
        self.assertEqual(got.shape, (1, 1))
        self.assertEqual(got.dtype, torch.int64)
        self.assertTrue(0 <= got.item() < 202048)
