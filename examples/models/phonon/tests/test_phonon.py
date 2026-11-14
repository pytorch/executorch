# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fast checks on a tiny random Phonon: the export programs against the eager
reference, the backend module swaps, checkpoint loading, the I/O contract of
exported programs, and the host-side text handling and noise.

  python -m pytest examples/models/phonon/tests -q
"""

import copy
import json
import tempfile
import unittest

import numpy as np
import torch

from executorch.examples.models.phonon.export_phonon import (
    lower_phonon,
    quantize_asymmetric_int4,
    quantize_scheme,
    source_transforms_for,
)
from executorch.examples.models.phonon.model import (
    build_programs,
    example_inputs,
    io_contract,
    mimi_zero_state,
    Phonon,
    PhononConfig,
    program_metadata,
    StreamingConvTranspose1d,
)
from executorch.examples.models.phonon.run_phonon import (
    EagerPrograms,
    frame_budget,
    NoiseGenerator,
    PhononRunner,
    prepare_text,
    split_sentences,
)
from executorch.examples.models.phonon.source_transformations import (
    compute_weights_in,
    ComputeIn,
    ConvTranspose1dAsConv1d,
    use_webgpu_compat_ops,
)
from executorch.examples.models.phonon.tests.reference import (
    FlowLMReference,
    MimiReference,
)
from executorch.exir import to_edge_transform_and_lower
from torch.export import export

TINY = PhononConfig(
    dim=64,
    num_heads=2,
    num_layers=2,
    ffn_dim=128,
    text_bins=50,
    latent_dim=8,
    flow_depth=2,
    duration_dim=16,
    sample_rate=1600,
    frame_rate=50.0,
    mimi_dim=32,
    n_filters=8,
    ratios=(2, 2),
    mimi_heads=2,
    mimi_layers=2,
    mimi_ffn_dim=64,
    mimi_context=12,
    max_context=64,
    max_prefill=16,
)


def random_model(cfg: PhononConfig = TINY, seed: int = 0) -> Phonon:
    """A Phonon with every parameter random (interleaved RoPE layout)."""
    torch.manual_seed(seed)
    model = Phonon(cfg).eval().requires_grad_(False)
    for p in model.parameters():
        p.copy_(torch.randn_like(p) * 0.2)
    return model


def official_config() -> dict:
    """The fields of gradium/phonon-31m-en's config.json that are parsed."""
    transformer = {
        "transformer_num_heads": 8,
        "transformer_num_layers": 2,
        "transformer_context": 250,
        "transformer_max_period": 10000.0,
        "transformer_dim_feedforward": 2048,
    }
    return {
        "languages": ["en"],
        "flow_lm": {
            "d_model": 512,
            "num_heads": 8,
            "num_layers": 6,
            "dim_feedforward": 2048,
            "max_period": 10000.0,
            "n_bins": 4000,
            "lut_dim": 512,
            "flow_dim": 512,
            "flow_depth": 6,
            "ldim": 32,
            "time_rms_norm": "rms",
            "gelu": "erf",
        },
        "mimi": {
            "channels": 1,
            "sample_rate": 24000,
            "frame_rate": 12.5,
            "dimension": 512,
            "n_filters": 64,
            "n_residual_layers": 1,
            "ratios": [6, 5, 4],
            "kernel_size": 7,
            "last_kernel_size": 3,
            "residual_kernel_size": 3,
            "compress": 2,
            "gelu": "erf",
            **transformer,
        },
        "lsd_decode_steps": 1,
        "eos_threshold": 4.0,
        "conditioners": [
            {
                "type": "lut",
                "lut": {"n_bins": 31, "dim": 16, "possible_values": ["0", "1", "2"]},
                "name": "num_speakers",
            },
            {
                "type": "continuous",
                "continuous": {
                    "scale_factor": 1000.0,
                    "dim": 128,
                    "max_period": 10000.0,
                },
                "name": "duration_delta",
            },
        ],
        "fuser": {"sum": ["num_speakers", "duration_delta"]},
    }


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


class TestConfig(unittest.TestCase):
    def test_official_config_matches_defaults(self):
        cfg = PhononConfig.from_dict(official_config())
        self.assertEqual(cfg, PhononConfig())
        self.assertEqual(cfg.frame_samples, 1920)
        self.assertEqual(cfg.upsample_stride, 16)

    def test_rejects_unsupported_variant(self):
        data = official_config()
        data["lsd_decode_steps"] = 4
        with self.assertRaises(ValueError):
            PhononConfig.from_dict(data)


class TestPrograms(unittest.TestCase):
    def setUp(self):
        self.ref_model = random_model()
        self.model = copy.deepcopy(self.ref_model).to_half_split_rope()

    def _check_flow_lm(self, attention: str):
        cfg = TINY
        flow = build_programs(self.model, attention)["flow_lm"]
        ref = FlowLMReference(self.ref_model)
        g = torch.Generator().manual_seed(1)
        voice = torch.randn(1, 6, cfg.dim, generator=g)
        text = torch.randn(1, 4, cfg.dim, generator=g)
        zeros = torch.zeros(1, cfg.latent_dim)
        with torch.no_grad():
            flow(voice, torch.tensor([0]), zeros)
            flow(text, torch.tensor([6]), zeros)
            ref.prompt(voice)
            ref.prompt(text)
            embeds, prev, pos = self.model.flow_lm.bos_embeds(), None, 10
            for _ in range(6):
                noise = torch.randn(1, cfg.latent_dim, generator=g)
                latent, eos, embeds = flow(embeds, torch.tensor([pos]), noise)
                ref_latent, ref_eos = ref.step(prev, noise)
                self.assertLess(max_diff(latent, ref_latent.view(1, -1)), 1e-4)
                self.assertLess(max_diff(eos, ref_eos), 1e-4)
                prev, pos = ref_latent, pos + 1

    def test_flow_lm_custom_sdpa_matches_reference(self):
        self._check_flow_lm("custom_sdpa")

    def test_flow_lm_sdpa_matches_reference(self):
        self._check_flow_lm("sdpa")

    def test_mimi_matches_reference(self):
        cfg = TINY
        mimi = build_programs(self.model)["mimi_decode"]
        ref = MimiReference(self.ref_model)
        g = torch.Generator().manual_seed(2)
        state = mimi_zero_state(cfg)
        # Enough frames to slide the transformer window several times.
        for _ in range(6):
            latent = torch.randn(1, 1, cfg.latent_dim, generator=g)
            with torch.no_grad():
                pcm, *state = mimi(latent, *state)
            self.assertEqual(tuple(pcm.shape), (1, 1, cfg.frame_samples))
            self.assertLess(max_diff(pcm, ref.decode(latent)), 1e-4)

    def test_stateful_mimi_matches_state_io(self):
        mimi = build_programs(self.model)["mimi_decode"]
        stateful = build_programs(self.model, mimi_state_in_program=True)["mimi_decode"]
        g = torch.Generator().manual_seed(3)
        # Two utterances: the second reset must clear the first one's state.
        for _ in range(2):
            state = mimi_zero_state(TINY)
            for k in range(4):
                latent = torch.randn(1, 1, TINY.latent_dim, generator=g)
                with torch.no_grad():
                    want, *state = mimi(latent, *state)
                    got = stateful(latent, torch.tensor([1.0 if k == 0 else 0.0]))
                self.assertLess(max_diff(got, want), 1e-6)

    def test_weight_names_are_unique_across_programs(self):
        # The CUDA backend shares a weight between methods when the names match.
        programs = build_programs(self.model, "sdpa")
        for transform in source_transforms_for("cuda", False):
            transform(programs)
        compute_weights_in(programs, torch.bfloat16)
        owner = {}
        for name, program in programs.items():
            for fqn, tensor in [*program.named_parameters(), *program.named_buffers()]:
                if fqn in owner and owner[fqn][1] is not tensor:
                    self.fail(f"{fqn} is in {owner[fqn][0]} and {name}")
                owner[fqn] = (name, tensor)

    def test_webgpu_compat_ops_preserve_outputs(self):
        inputs = example_inputs(TINY)
        programs = build_programs(self.model)
        with torch.no_grad():
            before = {n: programs[n](*inputs[n]) for n in ("flow_lm", "mimi_decode")}
        # norm1/norm2 x 2 layers; 2 x 2 residual + 3 decoder ELUs; 2 time
        # embedder + 2 x 2 res block + 1 final SiLUs; 3 transposed convs.
        self.assertEqual(use_webgpu_compat_ops(programs), 4 + 7 + 7 + 3)
        for p in programs.values():
            if hasattr(p, "reset"):
                p.reset()
        with torch.no_grad():
            for name, outputs in before.items():
                for a, b in zip(outputs, programs[name](*inputs[name])):
                    self.assertLess(max_diff(a, b), 1e-4)

    def _check_swaps(self, backend: str, counts: list[int]):
        inputs = example_inputs(TINY)
        programs = build_programs(self.model)
        with torch.no_grad():
            before = programs["mimi_decode"](*inputs["mimi_decode"])
        self.assertEqual(
            [t(programs) for t in source_transforms_for(backend, False)], counts
        )
        with torch.no_grad():
            after = programs["mimi_decode"](*inputs["mimi_decode"])
        for a, b in zip(before, after):
            self.assertLess(max_diff(a, b), 1e-4)

    def test_vulkan_swaps_preserve_outputs(self):
        # 7 ELUs, the depthwise upsampler, 2 decoder transposed convs.
        self._check_swaps("vulkan", [7, 1, 2])

    def test_xnnpack_swaps_preserve_outputs(self):
        # The depthwise upsampler, 2 decoder transposed convs.
        self._check_swaps("xnnpack", [1, 2])

    def test_cuda_swaps_preserve_outputs(self):
        self._check_swaps("cuda", [1, 2])

    def test_mlx_swaps_preserve_outputs(self):
        # 7 ELUs, the depthwise upsampler, 2 decoder transposed convs.
        self._check_swaps("mlx", [7, 1, 2])

    def test_conv_transpose_as_conv_matches(self):
        torch.manual_seed(0)
        for stride, steps, bias in (
            (1, 1, True),
            (4, 16, False),
            (5, 1, True),
            (6, 16, True),
        ):
            convtr = torch.nn.ConvTranspose1d(
                6, 3, 2 * stride, stride=stride, bias=bias
            )
            x = torch.randn(1, 6, steps)
            with torch.no_grad():
                got = ConvTranspose1dAsConv1d(convtr)(x)
                self.assertLess(max_diff(got, convtr(x)), 1e-5, stride)

    def test_compute_weights_in_keeps_float32_io(self):
        # --dtype bf16 is for CUDA and MLX, which apply the Mimi swaps first.
        inputs = example_inputs(TINY)
        programs = build_programs(self.model)
        with torch.no_grad():
            before = programs["mimi_decode"](*inputs["mimi_decode"])
        for transform in source_transforms_for("cuda", False):
            transform(programs)
        self.assertGreater(compute_weights_in(programs, torch.bfloat16), 0)
        mimi = programs["mimi_decode"]
        convtrs = [
            m.convtr for m in mimi.modules() if isinstance(m, StreamingConvTranspose1d)
        ]
        self.assertEqual(len(convtrs), 3)
        for convtr in convtrs:
            self.assertIsInstance(convtr, ComputeIn)
        with torch.no_grad():
            after = mimi(*inputs["mimi_decode"])
        for a, b in zip(before, after):
            self.assertEqual(b.dtype, torch.float32)
            self.assertLess(max_diff(a, b), 0.05 * (a.abs().max().item() + 1e-6))


class TestCheckpoint(unittest.TestCase):
    def test_strict_load_drops_only_training_tensors(self):
        from safetensors.torch import save_file

        model = random_model()
        state = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
        state["flow_lm.flow.w_s_t.0.weight"] = torch.zeros(32, 2)
        with tempfile.TemporaryDirectory() as d:
            save_file(state, f"{d}/model.safetensors")
            loaded = Phonon.from_checkpoint(f"{d}/model.safetensors", TINY)
            self.assertEqual(loaded.rope_layout, "half_split")
            del state["flow_lm.out_eos.bias"]
            save_file(state, f"{d}/model.safetensors")
            with self.assertRaises(RuntimeError):
                Phonon.from_checkpoint(f"{d}/model.safetensors", TINY)


class TestExport(unittest.TestCase):
    def _lower(self, backend: str):
        model = random_model().to_half_split_rope()
        program, delegation, metadata = lower_phonon(model, backend)
        return model, program, delegation, metadata

    def test_portable_program_matches_contract_and_eager(self):
        model, program, _, metadata = self._lower("portable")
        try:
            import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401
            from executorch.extension.pybindings.portable_lib import (
                _load_for_executorch_from_buffer,
            )
        except ImportError:
            self.skipTest("ExecuTorch pybindings are not available")
        module = _load_for_executorch_from_buffer(program.buffer)
        contract = io_contract(TINY)
        for method, spec in contract.items():
            meta = module.method_meta(method)
            self.assertEqual(meta.num_inputs(), len(spec["inputs"]))
            self.assertEqual(meta.num_outputs(), len(spec["outputs"]))
            for i, tensor in enumerate(spec["inputs"]):
                sizes = list(meta.input_tensor_meta(i).sizes())
                want = [TINY.max_prefill if d == "S" else d for d in tensor["shape"]]
                self.assertEqual(sizes, want, f"{method} input {i}")
        self.assertEqual(module.run_method("sample_rate", [])[0], TINY.sample_rate)
        eager = build_programs(model)
        inputs = example_inputs(TINY)
        with torch.no_grad():
            for method in contract:
                want = eager[method](*inputs[method])
                want = want if isinstance(want, tuple) else (want,)
                got = module.run_method(method, list(inputs[method]))
                for a, b in zip(got, want):
                    self.assertLess(max_diff(a, b), 1e-4, method)

    def test_xnnpack_program_runs_from_file(self):
        # The runners load the program from a file, which frees delegate data
        # after init; a load from a buffer never frees it.
        model, program, _, _ = self._lower("xnnpack")
        try:
            import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401
            from executorch.extension.pybindings.portable_lib import (
                _load_for_executorch,
            )
        except ImportError:
            self.skipTest("ExecuTorch pybindings are not available")
        inputs = example_inputs(TINY)["mimi_decode"]
        with torch.no_grad():
            want = build_programs(model)["mimi_decode"](*inputs)
        with tempfile.TemporaryDirectory() as d:
            with open(f"{d}/phonon.pte", "wb") as f:
                program.write_to_file(f)
            got = _load_for_executorch(f"{d}/phonon.pte").run_method(
                "mimi_decode", list(inputs)
            )
        for a, b in zip(got, want):
            self.assertLess(max_diff(a, b), 1e-4)

    def test_stateful_mimi_program_keeps_state_between_calls(self):
        try:
            from executorch.extension.pybindings.portable_lib import (
                _load_for_executorch,
            )
        except ImportError:
            self.skipTest("ExecuTorch pybindings are not available")
        model = random_model().to_half_split_rope()
        stateful = build_programs(model, mimi_state_in_program=True)["mimi_decode"]
        inputs = example_inputs(TINY, mimi_state_in_program=True)["mimi_decode"]
        program = to_edge_transform_and_lower(
            {"mimi_decode": export(stateful, inputs, strict=False)}
        ).to_executorch()
        g = torch.Generator().manual_seed(4)
        with tempfile.TemporaryDirectory() as d:
            with open(f"{d}/mimi.pte", "wb") as f:
                program.write_to_file(f)
            module = _load_for_executorch(f"{d}/mimi.pte")
            for reset in (1.0, 0.0, 0.0, 1.0, 0.0):
                args = (torch.randn(1, 1, TINY.latent_dim, generator=g),)
                args += (torch.tensor([reset]),)
                with torch.no_grad():
                    want = stateful(*args)
                got = module.run_method("mimi_decode", list(args))[0]
                self.assertLess(max_diff(got, want), 1e-4)

    def test_bf16_only_on_supported_backends(self):
        with self.assertRaises(ValueError):
            lower_phonon(random_model().to_half_split_rope(), "vulkan", dtype="bf16")

    def test_quantize_picks_the_scheme_of_the_backend(self):
        self.assertEqual(quantize_scheme("xnnpack", "4bit", None), ("8da4w", "fp32"))
        self.assertEqual(quantize_scheme("vulkan", "4bit", None), ("4w", "fp32"))
        self.assertEqual(quantize_scheme("cuda", "4bit", None), ("4w", "bf16"))
        self.assertEqual(quantize_scheme("mlx", "8bit", "bf16"), ("8w", "bf16"))
        self.assertEqual(quantize_scheme("vulkan", None, None), (None, "fp32"))

    def test_quantize_only_with_a_kernel(self):
        for backend, quantize, dtype in (
            ("vulkan", "8bit", None),
            ("portable", "4bit", None),
            ("cuda", "4bit", "fp32"),
            ("xnnpack", None, "bf16"),
        ):
            with self.assertRaises(ValueError, msg=f"{backend} {quantize}"):
                quantize_scheme(backend, quantize, dtype)

    def test_asymmetric_int4_skips_incompatible_linears(self):
        torch.manual_seed(0)
        module = torch.nn.Sequential(torch.nn.Linear(64, 16), torch.nn.Linear(16, 8))
        x = torch.randn(4, 64)
        want = module(x)
        quantize_asymmetric_int4(module, 32)
        self.assertIsNot(type(module[0].weight), torch.nn.Parameter)
        self.assertIs(type(module[1].weight), torch.nn.Parameter)
        self.assertLess(max_diff(module(x), want), 0.1)

    def test_xnnpack_lowering_delegates(self):
        _, _, delegation, _ = self._lower("xnnpack")
        for method in ("voice_embed", "flow_lm", "mimi_decode"):
            self.assertGreater(delegation[method]["delegates"], 0, method)


class TestHost(unittest.TestCase):
    def test_prepare_text(self):
        self.assertEqual(prepare_text("  hello   world "), ("Hello world.", 3))
        self.assertEqual(
            prepare_text("one two three four five?"), ("One two three four five?", 1)
        )

    def test_prepare_text_uses_ascii_rules(self):
        # Same rules as the C++ runner: only ASCII letters and digits count.
        self.assertEqual(prepare_text("élan"), ("élan.", 3))
        self.assertEqual(prepare_text("café"), ("Café", 3))

    def test_noise_generator(self):
        # The first splitmix64 output for seed 0.
        self.assertEqual(
            NoiseGenerator(0)._uniform(), (0xE220A8397B1DCDAF >> 11) * 2.0**-53
        )
        noise = NoiseGenerator(3).normal(20000, 1.0)
        self.assertLess(abs(noise.mean().item()), 0.03)
        self.assertLess(abs(noise.std().item() - 1.0), 0.03)
        self.assertTrue(
            torch.equal(NoiseGenerator(3).normal(32, 0.5), noise[:32] * 0.5)
        )
        self.assertFalse(torch.equal(NoiseGenerator(4).normal(32, 1.0), noise[:32]))

    def test_split_sentences(self):
        count = lambda s: len(s.split())  # noqa: E731
        text = "One two three. Four five six seven. Eight."
        self.assertEqual(
            split_sentences(text, count, 4),
            ["One two three.", "Four five six seven.", "Eight."],
        )
        self.assertEqual(split_sentences(text, count, 100), [text])
        self.assertEqual(split_sentences("a b c d e", count, 2), ["a b", "c d", "e"])

    def test_runner_on_eager_programs(self):
        model = random_model().to_half_split_rope()
        programs = EagerPrograms(build_programs(model), program_metadata(model))
        runner = PhononRunner(programs, lambda text: [1, 2, 3][: len(text.split())])
        speaker = torch.randn(1, 6, TINY.dim)
        pcm, info = runner.synthesize("hi there", speaker, temperature=0.0)
        self.assertGreater(info["frames"], 0)
        self.assertLessEqual(info["frames"], frame_budget(2, TINY.frame_rate))
        self.assertEqual(pcm.size, info["frames"] * TINY.frame_samples)
        again, _ = runner.synthesize("hi there", speaker, temperature=0.0)
        self.assertTrue((pcm == again).all())
        stateful = PhononRunner(
            EagerPrograms(
                build_programs(model, mimi_state_in_program=True),
                program_metadata(model, mimi_state_in_program=True),
            ),
            lambda text: [1, 2, 3][: len(text.split())],
        )
        for _ in range(2):
            got, _ = stateful.synthesize("hi there", speaker, temperature=0.0)
            self.assertLess(np.abs(got - pcm).max(), 1e-5)

    def test_contract_is_json_serializable(self):
        json.dumps(io_contract(PhononConfig()))
        self.assertEqual(len(io_contract(PhononConfig())["mimi_decode"]["inputs"]), 16)
        stateful = io_contract(PhononConfig(), mimi_state_in_program=True)
        self.assertEqual(
            [x["name"] for x in stateful["mimi_decode"]["inputs"]], ["latent", "reset"]
        )
        self.assertEqual(len(stateful["mimi_decode"]["outputs"]), 1)


if __name__ == "__main__":
    unittest.main()
