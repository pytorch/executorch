# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ``CoreAIBackend.preprocess`` and its AOT-compile config.

Covers all four asset-delivery combinations: source ``.aimodel`` vs
AOT-compiled ``.aimodelc``, each either inline (embedded in the .pte) or as a
sidecar, plus ``AOTCompileConfig`` parsing. The compiled-delivery cases mock
``coreai-build`` so they run without the Metal Toolchain; the real-toolchain
integration is in :class:`CoreAIAOTCompileTest` (gated on ``coreai-build``).
"""

import contextlib
import itertools
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

from executorch.backends.apple.coreai.compiler.preprocess import (
    _aot_compile_options,
    _compile_aot,
    _compiled_arch_bundles,
    _convert_to_aiprogram,
    _run_coreai_build,
    AOTCompileConfig,
    COMPILE_SPEC_KEYS,
    coreai_sidecar_dir,
    CoreAIBackend,
)
from executorch.backends.apple.coreai.partition.partitioner import CoreAIPartitioner
from executorch.backends.apple.coreai.runtime.test.export_fixtures import (
    _validate_asset_metadata,
)
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.lowered_backend_module import (
    executorch_call_delegate,
    get_lowered_backend_modules,
)


def _coreai_build_available() -> bool:
    try:
        return (
            subprocess.run(
                ["xcrun", "--find", "coreai-build"], capture_output=True
            ).returncode
            == 0
        )
    except FileNotFoundError:
        return False


class _Elementwise(nn.Module):
    # Parameter-free so the raw edge program converts directly, keeping these
    # tests focused on preprocess delivery/packaging (weight handling is covered
    # by the delegation e2e tests).
    def forward(self, x):
        return x + x * 2.0


class _OrderedArithmetic(nn.Module):
    def forward(self, z, a):
        difference = z - a
        scaled = z + (a + a)
        return scaled, difference


def _ordered_edge_program(dtype=torch.float32):
    inputs = (torch.full((2, 4), 7, dtype=dtype), torch.full((2, 4), 2, dtype=dtype))
    ep = torch.export.export(_OrderedArithmetic().eval(), inputs)
    return to_edge(ep).exported_program(), inputs


def _edge_program():
    ep = torch.export.export(_Elementwise().eval(), (torch.randn(2, 8),))
    return to_edge(ep).exported_program()


def _fake_run_coreai_build(aimodel_path, out_dir, opts):
    """Stand-in for ``xcrun coreai-build``: drop fake per-arch .aimodelc dirs.

    The contents vary with the options so tests can tell one build's output
    from another's, as real compiled bundles would.
    """
    for arch in opts["architectures"] or ["h15g"]:
        bundle = Path(out_dir) / f"model.{arch}.aimodelc"
        bundle.mkdir(parents=True, exist_ok=True)
        (bundle / "model.mil").write_bytes(f"fake-compiled:{opts['platform']}".encode())


def _aot_spec(config: dict) -> CompileSpec:
    return CompileSpec(
        COMPILE_SPEC_KEYS.AOT_COMPILE_CONFIG.value, json.dumps(config).encode()
    )


_SIDECAR_SPEC = CompileSpec(COMPILE_SPEC_KEYS.USES_SIDECAR.value, b"1")
_MOCK_BUILD = "executorch.backends.apple.coreai.compiler.preprocess._run_coreai_build"


class AOTCompileConfigTest(unittest.TestCase):
    """``AOTCompileConfig`` / ``_aot_compile_options`` parsing (no coreai-build)."""

    def test_parses_known_fields(self):
        opts = _aot_compile_options(
            [_aot_spec({"platform": "iOS", "architectures": ["h17p"]})]
        )
        self.assertEqual(opts["platform"], "iOS")
        self.assertEqual(opts["architectures"], ["h17p"])

    def test_defaults_when_empty(self):
        opts = _aot_compile_options([_aot_spec({})])
        self.assertEqual(opts["platform"], "macOS")
        self.assertEqual(opts["architectures"], [])
        self.assertEqual(opts["preferred_compute"], "none")
        self.assertFalse(opts["expect_frequent_reshapes"])

    def test_rejects_unexpected_field(self):
        # A typo like "platfrom" must fail loudly, not silently default.
        with self.assertRaises(ValueError):
            _aot_compile_options([_aot_spec({"platfrom": "iOS"})])

    def test_config_json_roundtrip(self):
        cfg = AOTCompileConfig(
            platform="iOS",
            preferred_compute="neural-engine",
            architectures=["h17p"],
            expect_frequent_reshapes=True,
        )
        self.assertEqual(AOTCompileConfig.from_json(cfg.to_json()), cfg)

    def test_config_from_dict_rejects_unexpected(self):
        with self.assertRaises(ValueError):
            AOTCompileConfig.from_dict({"platfrom": "iOS"})

    def test_config_from_dict_rejects_non_list_architectures(self):
        # A bare string splats into ['h', '1', '5', 'g'], which becomes four
        # bogus --architecture flags rather than a clear error.
        with self.assertRaises(ValueError):
            AOTCompileConfig.from_dict({"architectures": "h15g"})

    def test_empty_config_roundtrip(self):
        self.assertEqual(
            AOTCompileConfig.from_json(AOTCompileConfig().to_json()),
            AOTCompileConfig(),
        )


class AOTCompilerTest(unittest.TestCase):
    def test_compiler_flags_preserve_architecture_order_and_options(self):
        opts = _aot_compile_options(
            [
                _aot_spec(
                    {
                        "platform": "iOS",
                        "architectures": ["h17p", "h15g"],
                        "preferred_compute": "neural-engine",
                        "expect_frequent_reshapes": True,
                    }
                ),
                CompileSpec(COMPILE_SPEC_KEYS.MIN_DEPLOYMENT_VERSION.value, b"27.5"),
            ]
        )
        with mock.patch(
            "executorch.backends.apple.coreai.compiler.preprocess.subprocess.run",
            return_value=mock.Mock(returncode=0),
        ) as run:
            _run_coreai_build(Path("model.aimodel"), Path("compiled"), opts)
        run.assert_called_once_with(
            [
                "xcrun",
                "coreai-build",
                "compile",
                "model.aimodel",
                "--output",
                "compiled",
                "--platform",
                "iOS",
                "--min-deployment-version",
                "27.5",
                "--preferred-compute",
                "neural-engine",
                "--architecture",
                "h17p",
                "--architecture",
                "h15g",
                "--expect-frequent-reshapes",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

    def test_default_compiler_flags_leave_architectures_to_sdk(self):
        with mock.patch(
            "executorch.backends.apple.coreai.compiler.preprocess.subprocess.run",
            return_value=mock.Mock(returncode=0),
        ) as run:
            _run_coreai_build(
                Path("model.aimodel"), Path("compiled"), _aot_compile_options([])
            )
        self.assertEqual(
            run.call_args.args[0],
            [
                "xcrun",
                "coreai-build",
                "compile",
                "model.aimodel",
                "--output",
                "compiled",
                "--platform",
                "macOS",
            ],
        )

    @mock.patch(_MOCK_BUILD, side_effect=_fake_run_coreai_build)
    def test_aot_preserves_sdk_source_hash(self, _build):
        program = _convert_to_aiprogram(
            _edge_program(), input_names=["input_0"], output_names=["output_0"]
        )
        with tempfile.TemporaryDirectory() as d:
            model_hash, _ = _compile_aot(program, _aot_compile_options([]), Path(d))
            self.assertEqual(
                model_hash, (Path(d) / "model.aimodel" / "main.hash").read_bytes().hex()
            )

    def test_bundle_names_are_validated(self):
        for name in (
            "wrong.h15g.aimodelc",
            "model.h15g.aimodel",
            "model..aimodelc",
            "model...aimodelc",
            "model....aimodelc",
            "model.h15g\\\\other.aimodelc",
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                (Path(d) / name).mkdir()
                with self.assertRaisesRegex(
                    RuntimeError, "invalid coreai-build bundle"
                ):
                    _compiled_arch_bundles(Path(d))

    def test_duplicate_architectures_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            bundle = Path(d) / "model.h15g.aimodelc"
            bundle.mkdir()
            with mock.patch.object(
                Path, "iterdir", return_value=iter([bundle, bundle])
            ):
                with self.assertRaisesRegex(RuntimeError, "duplicate.*architecture"):
                    _compiled_arch_bundles(Path(d))

    def test_empty_compiler_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, "produced no .aimodelc bundles"):
                _compiled_arch_bundles(Path(d))

    def test_architectures_come_from_actual_sorted_bundle_names(self):
        with tempfile.TemporaryDirectory() as d:
            for arch in ("h17p", "h15g"):
                (Path(d) / f"model.{arch}.aimodelc").mkdir()
            self.assertEqual(
                _compiled_arch_bundles(Path(d)),
                [
                    (arch, Path(d) / f"model.{arch}.aimodelc")
                    for arch in ("h15g", "h17p")
                ],
            )


class AssetBindingsTest(unittest.TestCase):
    """The same converter bindings must survive all four delivery modes."""

    @mock.patch(_MOCK_BUILD, side_effect=_fake_run_coreai_build)
    def test_ordered_bindings_are_stable_across_delivery_and_dtype(self, _build):
        from executorch.backends.apple.coreai.compiler.io_compat import coreai_io_names

        for dtype in (torch.float32, torch.float16):
            edge, inputs = _ordered_edge_program(dtype)
            actual = edge.module()(*inputs)
            torch.testing.assert_close(actual[0], torch.full_like(inputs[0], 11))
            torch.testing.assert_close(actual[1], torch.full_like(inputs[0], 5))
            original_convert = _convert_to_aiprogram
            converted = []

            def convert(
                *args, original_convert=original_convert, converted=converted, **kwargs
            ):
                program = original_convert(*args, **kwargs)
                converted.append(coreai_io_names(program))
                return program

            with mock.patch(
                "executorch.backends.apple.coreai.compiler.preprocess._convert_to_aiprogram",
                side_effect=convert,
            ):
                for aot, sidecar in itertools.product((False, True), repeat=2):
                    with self.subTest(dtype=dtype, aot=aot, sidecar=sidecar):
                        with tempfile.TemporaryDirectory() as d:
                            specs = [_aot_spec({})] if aot else []
                            if sidecar:
                                specs.append(_SIDECAR_SPEC)
                            with (
                                coreai_sidecar_dir(d)
                                if sidecar
                                else contextlib.nullcontext()
                            ):
                                result = CoreAIBackend.preprocess(edge, specs)
                            manifest = json.loads(result.processed_bytes)
                            _validate_asset_metadata(
                                manifest, Path(d), result.data_store_output
                            )
                            self.assertEqual(manifest["version"], 2)
                            self.assertEqual(manifest["function"], "main")
                            self.assertEqual(
                                manifest["input_names"], ["input_0", "input_1"]
                            )
                            self.assertEqual(
                                manifest["output_names"], ["output_0", "output_1"]
                            )
                            self.assertEqual(
                                converted[-1],
                                (manifest["input_names"], manifest["output_names"]),
                            )
                            self.assertNotIn("runtime_supported", manifest)
                            if aot:
                                self.assertEqual(
                                    manifest["archs"],
                                    {"h15g": f"{manifest['hash']}/model.h15g.aimodelc"},
                                )
                                continue
                            self.assertEqual(
                                manifest["path"], f"{manifest['hash']}/model.aimodel"
                            )
                            if sidecar:
                                bundle = Path(d) / manifest["path"]
                                self.assertEqual(
                                    (bundle / "main.hash").read_bytes().hex(),
                                    manifest["hash"],
                                )
                                self.assertIsInstance(manifest["files"], dict)
                            else:
                                store = result.data_store_output
                                key = (
                                    f"coreai/{manifest['hash']}/model.aimodel/main.hash"
                                )
                                self.assertEqual(
                                    store.buffers[
                                        store.pte_data[key].buffer_index
                                    ].hex(),
                                    manifest["hash"],
                                )

    def test_unnamed_conversion_mock_is_not_given_fabricated_bindings(self):
        edge = _edge_program()
        unnamed = _convert_to_aiprogram(edge)
        with mock.patch(
            "executorch.backends.apple.coreai.compiler.preprocess._convert_to_aiprogram",
            return_value=unnamed,
        ):
            for aot in (False, True):
                with self.subTest(aot=aot):
                    with self.assertRaisesRegex(ValueError, "names mismatch"):
                        CoreAIBackend.preprocess(edge, [_aot_spec({})] if aot else [])

    @mock.patch(_MOCK_BUILD, side_effect=_fake_run_coreai_build)
    def test_mutated_buffers_keep_actual_names_and_flag_runtime_unsupported(
        self, _build
    ):
        from executorch.backends.apple.coreai.compiler.io_compat import coreai_io_names

        class MutatedBuffer(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("count", torch.zeros(2, 4))

            def forward(self, x):
                self.count.add_(1.0)
                return x + self.count

        edge = to_edge(
            torch.export.export(MutatedBuffer(), (torch.ones(2, 4),))
        ).exported_program()
        expected = coreai_io_names(_convert_to_aiprogram(edge))
        for aot, sidecar in itertools.product((False, True), repeat=2):
            with self.subTest(aot=aot, sidecar=sidecar):
                with tempfile.TemporaryDirectory() as d:
                    specs = [_aot_spec({})] if aot else []
                    if sidecar:
                        specs.append(_SIDECAR_SPEC)
                    with coreai_sidecar_dir(d) if sidecar else contextlib.nullcontext():
                        result = CoreAIBackend.preprocess(edge, specs)
                manifest = json.loads(result.processed_bytes)
                self.assertEqual(manifest["version"], 2)
                self.assertEqual(manifest["function"], "main")
                self.assertFalse(manifest["runtime_supported"])
                self.assertEqual(manifest["runtime_unsupported_reason"], "stateful")
                self.assertEqual(
                    (manifest["input_names"], manifest["output_names"]), expected
                )

    def test_mutated_user_inputs_are_rejected(self):
        """ExecuTorch hands a mutated input across, but Core AI makes it state."""

        class MutatedInput(nn.Module):
            def forward(self, x):
                x.add_(1.0)
                return x * 2.0

        edge = to_edge(
            torch.export.export(MutatedInput(), (torch.ones(2, 4),))
        ).exported_program()
        with self.assertRaisesRegex(ValueError, "tensor/non-tensor mismatch"):
            CoreAIBackend.preprocess(edge, [])


class AimodelPreprocessTest(unittest.TestCase):
    """Source ``.aimodel`` delivery (no coreai-build)."""

    def test_inline_embeds_files_in_nds(self):
        result = CoreAIBackend.preprocess(_edge_program(), [])
        manifest = json.loads(result.processed_bytes)
        self.assertEqual(manifest["packaging"], "inline")
        self.assertTrue(manifest["files"], "expected at least one asset file")
        out = result.data_store_output
        self.assertIsNotNone(out)
        for rel in manifest["files"]:
            self.assertIn(f"coreai/{manifest['hash']}/{rel}", out.pte_data)
        self.assertEqual(out.external_data, {})  # nothing external for inline

    def test_inline_keys_keep_the_bundle_directory(self):
        """Embedded keys must reconstruct to the shape sidecar writes on disk.

        The bundle name has to survive flattening, otherwise unpacking cannot
        tell that these files belong to a ``.aimodel``.
        """
        manifest = json.loads(
            CoreAIBackend.preprocess(_edge_program(), []).processed_bytes
        )
        self.assertTrue(
            all(rel.startswith("model.aimodel/") for rel in manifest["files"]),
            manifest["files"],
        )

    def test_sidecar_writes_bundle_and_does_not_embed(self):
        with tempfile.TemporaryDirectory() as d:
            with coreai_sidecar_dir(d):
                result = CoreAIBackend.preprocess(_edge_program(), [_SIDECAR_SPEC])
            manifest = json.loads(result.processed_bytes)
            self.assertEqual(manifest["packaging"], "sidecar")
            # Only the relative, hash-keyed path is referenced (no build path).
            self.assertEqual(manifest["path"], f"{manifest['hash']}/model.aimodel")
            self.assertNotIn(d, result.processed_bytes.decode())
            bundle = Path(d) / manifest["path"]
            self.assertTrue(bundle.is_dir())
            self.assertTrue((bundle / "main.mlirb").exists())
            self.assertIsNone(result.data_store_output)  # nothing embedded

    def test_sidecar_without_env_var_raises(self):
        # uses_sidecar set but COREAI_SIDECAR_DIR unset -> fail fast.
        env = {k: v for k, v in os.environ.items() if k != "COREAI_SIDECAR_DIR"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                CoreAIBackend.preprocess(_edge_program(), [_SIDECAR_SPEC])

    def test_inline_with_env_var_warns(self):
        # Env var set but delegate is inline -> soft warning, not an error.
        import executorch.backends.apple.coreai.compiler.preprocess as cp

        with tempfile.TemporaryDirectory() as d:
            with coreai_sidecar_dir(d):
                cp._WARNED_SIDECAR_ENV_IGNORED = False  # allow the once-warning
                with self.assertLogs(cp.logger, level="WARNING") as cm:
                    result = CoreAIBackend.preprocess(_edge_program(), [])
        self.assertEqual(json.loads(result.processed_bytes)["packaging"], "inline")
        self.assertTrue(any("uses_sidecar=True" in m for m in cm.output))

    def test_sidecar_dir_rejects_a_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "notadir"
            path.write_text("x")
            with self.assertRaisesRegex(RuntimeError, "not a directory"):
                with coreai_sidecar_dir(str(path)):
                    pass

    def test_sidecar_dir_ignores_loose_files(self):
        """Only asset directories conflict; ``.DS_Store`` must not block a build."""
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".DS_Store").write_bytes(b"\x00")
            with coreai_sidecar_dir(d):
                result = CoreAIBackend.preprocess(_edge_program(), [_SIDECAR_SPEC])
            self.assertEqual(json.loads(result.processed_bytes)["packaging"], "sidecar")

    def test_sidecar_dir_rejects_existing_assets(self):
        with tempfile.TemporaryDirectory() as d:
            with coreai_sidecar_dir(d):
                CoreAIBackend.preprocess(_edge_program(), [_SIDECAR_SPEC])
            with self.assertRaisesRegex(RuntimeError, "already holds assets"):
                with coreai_sidecar_dir(d):
                    pass

    def test_coreai_sidecar_dir_sets_and_restores_env(self):
        self.assertNotIn("COREAI_SIDECAR_DIR", os.environ)
        with coreai_sidecar_dir("/tmp/scoped"):
            self.assertEqual(os.environ["COREAI_SIDECAR_DIR"], "/tmp/scoped")
        self.assertNotIn("COREAI_SIDECAR_DIR", os.environ)  # restored on exit

    def test_end_to_end_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            ep = torch.export.export(nn.Linear(8, 8).eval(), (torch.randn(2, 8),))
            with coreai_sidecar_dir(d):
                to_edge_transform_and_lower(
                    ep, partitioner=[CoreAIPartitioner(uses_sidecar=True)]
                )
            bundles = list(Path(d).glob("*/model.aimodel"))
            self.assertEqual(len(bundles), 1)
            self.assertTrue((bundles[0] / "main.mlirb").exists())


def _lower_with_break(partitioner):
    """Lower a model whose middle op is tagged, forcing two delegates."""
    from executorch.backends.apple.coreai import get_default_passes
    from executorch.backends.apple.coreai.partition.partitioner import do_not_delegate
    from executorch.exir.pass_base import PassResult

    class _BreakPass:
        def __call__(self, gm):
            for n in gm.graph.nodes:
                if n.op == "call_function" and "mul" in str(n.target):
                    do_not_delegate(n)
            return PassResult(gm, True)

    class _Chain(nn.Module):
        def forward(self, x):
            return (x + 1.0) * 2.0 - 3.0

    ep = torch.export.export(_Chain().eval(), (torch.randn(4, 4),))
    return to_edge_transform_and_lower(
        ep,
        transform_passes=list(get_default_passes()) + [_BreakPass()],
        partitioner=[partitioner],
    )


class MinDeploymentVersionTest(unittest.TestCase):
    """One OS floor, one spelling, whatever the user wrote.

    ``coreai.authoring.OSVersion`` accepts only ``"v27"`` while
    ``coreai-build --min-deployment-version`` rejects it and wants a numeric
    version, so the spec is normalized once. Canonicalizing to major.minor also
    keeps the manifest identical across equivalent spellings.
    """

    def _manifest(self, raw: bytes) -> dict:
        specs = [
            CompileSpec(COMPILE_SPEC_KEYS.MIN_DEPLOYMENT_VERSION.value, raw),
        ]
        return json.loads(
            CoreAIBackend.preprocess(_edge_program(), specs).processed_bytes
        )

    def test_equivalent_spellings_normalize(self):
        for raw in (b"v27", b"27", b"27.0"):
            with self.subTest(raw.decode()):
                self.assertEqual(self._manifest(raw)["min_deployment_version"], "27.0")

    def test_unset_is_reported_as_none(self):
        manifest = json.loads(
            CoreAIBackend.preprocess(_edge_program(), []).processed_bytes
        )
        self.assertIsNone(manifest["min_deployment_version"])

    def test_aimodel_reports_the_floor_it_can_apply(self):
        """``save_asset`` takes an OSVersion, which has no minor version.

        Reporting the raw spec would advertise a floor the asset does not have.
        """
        manifest = self._manifest(b"27.5")
        self.assertEqual(manifest["min_deployment_version"], "27.0")

    @mock.patch(_MOCK_BUILD, side_effect=_fake_run_coreai_build)
    def test_aot_keeps_the_minor_version(self, build):
        """coreai-build accepts major[.minor[.patch]], so it is not truncated."""
        manifest = json.loads(
            CoreAIBackend.preprocess(
                _edge_program(),
                [
                    _aot_spec({"architectures": ["h15g"]}),
                    CompileSpec(
                        COMPILE_SPEC_KEYS.MIN_DEPLOYMENT_VERSION.value, b"27.5"
                    ),
                ],
            ).processed_bytes
        )
        self.assertEqual(manifest["min_deployment_version"], "27.5")
        self.assertEqual(build.call_args.args[2]["min_deployment_version"], "27.5")

    @mock.patch(_MOCK_BUILD, side_effect=_fake_run_coreai_build)
    def test_coreai_build_receives_the_numeric_form(self, build):
        """``v27`` reaches coreai-build as a version it accepts."""
        CoreAIBackend.preprocess(
            _edge_program(),
            [
                _aot_spec({"architectures": ["h15g"]}),
                CompileSpec(COMPILE_SPEC_KEYS.MIN_DEPLOYMENT_VERSION.value, b"v27"),
            ],
        )
        opts = build.call_args.args[2]
        self.assertEqual(opts["min_deployment_version"], "27.0")


class CompiledPreprocessTest(unittest.TestCase):
    """AOT-compiled ``.aimodelc`` delivery, with coreai-build mocked (ungated)."""

    @mock.patch(_MOCK_BUILD, side_effect=_fake_run_coreai_build)
    def test_inline_embeds_compiled_bundles(self, _build):
        result = CoreAIBackend.preprocess(
            _edge_program(),
            [_aot_spec({"platform": "iOS", "architectures": ["h15g", "h16"]})],
        )
        manifest = json.loads(result.processed_bytes)
        self.assertEqual(manifest["packaging"], "aot_compiled_inline")
        self.assertEqual(
            set(manifest),
            {
                "packaging",
                "hash",
                "files",
                "bundle_digests",
                "platform",
                "min_deployment_version",
                "archs",
                "version",
                "function",
                "input_names",
                "output_names",
            },
        )
        self.assertEqual(manifest["platform"], "iOS")
        self.assertEqual(sorted(manifest["archs"]), ["h15g", "h16"])
        self.assertEqual(manifest["version"], 2)
        self.assertEqual(manifest["function"], "main")
        self.assertEqual(manifest["input_names"], ["input_0"])
        self.assertEqual(manifest["output_names"], ["output_0"])
        self.assertIsNone(manifest["min_deployment_version"])
        self.assertEqual(
            manifest["archs"],
            {
                arch: f"{manifest['hash']}/model.{arch}.aimodelc"
                for arch in ("h15g", "h16")
            },
        )
        self.assertEqual(
            manifest["files"],
            {
                "model.h15g.aimodelc/model.mil": len(b"fake-compiled:iOS"),
                "model.h16.aimodelc/model.mil": len(b"fake-compiled:iOS"),
            },
        )
        store = result.data_store_output
        self.assertIsNotNone(store)
        self.assertEqual(
            set(store.pte_data),
            {f"coreai/{manifest['hash']}/{rel}" for rel in manifest["files"]},
        )
        self.assertEqual(store.external_data, {})
        _validate_asset_metadata(manifest, None, store)
        for entry in store.pte_data.values():
            self.assertEqual(entry.alignment, 16)
            self.assertEqual(store.buffers[entry.buffer_index], b"fake-compiled:iOS")

    @mock.patch(_MOCK_BUILD, side_effect=_fake_run_coreai_build)
    def test_sidecar_writes_compiled_bundles(self, _build):
        with tempfile.TemporaryDirectory() as d:
            with coreai_sidecar_dir(d):
                result = CoreAIBackend.preprocess(
                    _edge_program(),
                    [_aot_spec({"architectures": ["h15g"]}), _SIDECAR_SPEC],
                )
            manifest = json.loads(result.processed_bytes)
            self.assertEqual(
                manifest,
                {
                    "packaging": "aot_compiled_sidecar",
                    "hash": manifest["hash"],
                    "version": 2,
                    "files": {
                        "model.h15g.aimodelc/model.mil": len(b"fake-compiled:macOS")
                    },
                    "bundle_digests": manifest["bundle_digests"],
                    "function": "main",
                    "input_names": ["input_0"],
                    "output_names": ["output_0"],
                    "platform": "macOS",
                    "min_deployment_version": None,
                    "archs": {"h15g": f"{manifest['hash']}/model.h15g.aimodelc"},
                },
            )
            self.assertIsNone(result.data_store_output)  # bundles on disk only
            _validate_asset_metadata(manifest, Path(d), None)
            self.assertNotIn(d, result.processed_bytes.decode())
            for rel in manifest["archs"].values():
                bundle = Path(d) / rel
                self.assertTrue(bundle.is_dir())
                self.assertTrue(str(bundle).endswith(".aimodelc"))
                self.assertEqual(
                    (bundle / "model.mil").read_bytes(), b"fake-compiled:macOS"
                )

    @mock.patch(_MOCK_BUILD, side_effect=_fake_run_coreai_build)
    def test_archs_map_is_the_same_inline_and_sidecar(self, _build):
        """One shape for ``archs``, whatever the delivery.

        Inline could derive paths from the hash, but emitting the same map means
        a consumer never has to branch on ``packaging`` to read it. For inline
        the value is the key under ``coreai/``; for sidecar it is relative to
        the sidecar dir.
        """
        config = {"platform": "iOS", "architectures": ["h15g", "h16"]}
        inline_result = CoreAIBackend.preprocess(_edge_program(), [_aot_spec(config)])
        inline = json.loads(inline_result.processed_bytes)
        with tempfile.TemporaryDirectory() as d:
            with coreai_sidecar_dir(d):
                sidecar = json.loads(
                    CoreAIBackend.preprocess(
                        _edge_program(), [_aot_spec(config), _SIDECAR_SPEC]
                    ).processed_bytes
                )
        self.assertIsInstance(inline["archs"], dict)
        self.assertEqual(inline["archs"], sidecar["archs"])
        for arch, rel in inline["archs"].items():
            self.assertEqual(rel, f"{inline['hash']}/model.{arch}.aimodelc")
            self.assertIn(
                f"coreai/{rel}/model.mil",
                inline_result.data_store_output.pte_data,
            )

    @mock.patch(_MOCK_BUILD, side_effect=_fake_run_coreai_build)
    def test_defaults_to_all_architectures(self, _build):
        # Empty architectures => coreai-build decides (our fake yields one).
        result = CoreAIBackend.preprocess(_edge_program(), [_aot_spec({})])
        manifest = json.loads(result.processed_bytes)
        self.assertEqual(manifest["packaging"], "aot_compiled_inline")
        self.assertEqual(manifest["platform"], "macOS")  # default
        self.assertTrue(manifest["archs"])

    @mock.patch(_MOCK_BUILD, side_effect=_fake_run_coreai_build)
    def test_sidecar_rebuild_with_new_options_is_rejected(self, _build):
        """A second build must not quietly inherit the first one's bundles.

        The asset directory is keyed on the .aimodel hash, which does not cover
        platform, OS floor or compute preference, so a rebuild with different
        options lands on the same path.
        """
        with tempfile.TemporaryDirectory() as d:
            with coreai_sidecar_dir(d):
                CoreAIBackend.preprocess(
                    _edge_program(),
                    [
                        _aot_spec({"platform": "iOS", "architectures": ["h15g"]}),
                        _SIDECAR_SPEC,
                    ],
                )
                with self.assertRaisesRegex(RuntimeError, "already exists"):
                    CoreAIBackend.preprocess(
                        _edge_program(),
                        [
                            _aot_spec({"platform": "macOS", "architectures": ["h15g"]}),
                            _SIDECAR_SPEC,
                        ],
                    )

    @mock.patch(_MOCK_BUILD, side_effect=_fake_run_coreai_build)
    def test_failed_write_leaves_no_asset_directory(self, _build):
        """A half-written asset dir would block every later build.

        ``_reject_existing_asset_dir`` hard-fails on an existing directory, so
        the write is staged and renamed; a failure partway must clean up.
        """
        with tempfile.TemporaryDirectory() as d:
            with coreai_sidecar_dir(d):
                with mock.patch(
                    "executorch.backends.apple.coreai.compiler.preprocess"
                    ".shutil.move",
                    side_effect=OSError("no space left on device"),
                ):
                    with self.assertRaises(OSError):
                        CoreAIBackend.preprocess(
                            _edge_program(),
                            [_aot_spec({"architectures": ["h15g"]}), _SIDECAR_SPEC],
                        )
                # Nothing left behind, so the retry below is not blocked.
                self.assertEqual(list(Path(d).iterdir()), [])
                result = CoreAIBackend.preprocess(
                    _edge_program(),
                    [_aot_spec({"architectures": ["h15g"]}), _SIDECAR_SPEC],
                )
            manifest = json.loads(result.processed_bytes)
            bundle = Path(d) / next(iter(manifest["archs"].values()))
            self.assertTrue(bundle.is_dir())

    @mock.patch(_MOCK_BUILD, side_effect=_fake_run_coreai_build)
    def test_two_delegates_write_separate_asset_dirs(self, _build):
        """A graph break gives two delegates, which must not collide.

        They share one sidecar directory but differ in hash, so the
        already-exists guard has to stay quiet here.
        """
        with tempfile.TemporaryDirectory() as d:
            with coreai_sidecar_dir(d):
                lowered = _lower_with_break(
                    CoreAIPartitioner(
                        uses_sidecar=True,
                        aot_compile_config=AOTCompileConfig(
                            platform="iOS", architectures=["h15g"]
                        ),
                    )
                )
            lbms = get_lowered_backend_modules(lowered.exported_program().graph_module)
            self.assertEqual(len(lbms), 2)
            hashes = {json.loads(bytes(lbm.processed_bytes))["hash"] for lbm in lbms}
            self.assertEqual(len(hashes), 2)
            self.assertEqual(len(list(Path(d).iterdir())), 2)


class FixtureExportTest(unittest.TestCase):
    @mock.patch(_MOCK_BUILD, side_effect=_fake_run_coreai_build)
    def test_source_and_aot_fixtures_share_tensor_matrix(self, _build):
        from executorch.backends.apple.coreai.runtime.test.export_fixtures import (
            export_fixtures,
        )

        with tempfile.TemporaryDirectory() as d:
            roots = [Path(d) / "source", Path(d) / "aot"]
            config = AOTCompileConfig(platform="iOS", architectures=["h17p", "h15g"])
            source = export_fixtures(roots[0])
            compiled = export_fixtures(roots[1], aot_compile_config=config)
            self.assertEqual(len(source), 8)
            self.assertEqual(len(compiled), 8)
            self.assertEqual(
                {record["name"] for record in source},
                {
                    f"{dtype}_{shape}_{delivery}"
                    for dtype, shape, delivery in itertools.product(
                        ("fp32", "fp16"), ("static", "dynamic"), ("inline", "sidecar")
                    )
                },
            )
            for root, records in zip(roots, (source, compiled)):
                self.assertEqual(
                    json.loads((root / "fixtures.json").read_text()), records
                )
            for src, aot in zip(source, compiled):
                with self.subTest(fixture=src["name"]):
                    self.assertEqual(src["name"], aot["name"])
                    self.assertEqual(
                        aot["packaging"], f"aot_compiled_{src['packaging']}"
                    )
                    self.assertEqual(src["base_dir"], aot["base_dir"])
                    self.assertEqual(
                        (roots[0] / src["vectors"]).read_bytes(),
                        (roots[1] / aot["vectors"]).read_bytes(),
                    )
                    vectors = json.loads((roots[0] / src["vectors"]).read_text())
                    dynamic = "_dynamic_" in src["name"]
                    rows_per_case = [2, 4, 1, 3, 2] if dynamic else [2, 2]
                    dtype = "float16" if src["name"].startswith("fp16_") else "float32"
                    self.assertEqual(len(vectors["cases"]), len(rows_per_case))
                    for index, (case, rows) in enumerate(
                        zip(vectors["cases"], rows_per_case)
                    ):
                        with self.subTest(case=index):
                            self.assertEqual(len(case["inputs"]), 2)
                            self.assertEqual(len(case["outputs"]), 2)
                            for tensor in case["inputs"] + case["outputs"]:
                                self.assertEqual(tensor["shape"], [rows, 4])
                                self.assertEqual(tensor["dtype"], dtype)
                                self.assertEqual(len(tensor["values"]), rows * 4)
                            z, a = (tensor["values"] for tensor in case["inputs"])
                            self.assertEqual(
                                z, [i + 7 + index for i in range(rows * 4)]
                            )
                            self.assertEqual(
                                a, [2 - i - 2 * index for i in range(rows * 4)]
                            )
                            self.assertEqual(
                                case["outputs"][0]["values"],
                                [x + 2 * y for x, y in zip(z, a)],
                            )
                            self.assertEqual(
                                case["outputs"][1]["values"],
                                [x - y for x, y in zip(z, a)],
                            )
                    first, last = vectors["cases"][0], vectors["cases"][-1]
                    for kind in ("inputs", "outputs"):
                        for index in range(2):
                            with self.subTest(kind=kind, tensor=index):
                                self.assertEqual(
                                    first[kind][index]["shape"],
                                    last[kind][index]["shape"],
                                )
                                self.assertNotEqual(
                                    first[kind][index]["values"],
                                    last[kind][index]["values"],
                                )
                    self.assertGreater((roots[1] / aot["pte"]).stat().st_size, 0)
                    manifest = json.loads(
                        (roots[1] / aot["name"] / "delegate.json").read_text()
                    )
                    self.assertEqual(manifest["version"], 2)
                    self.assertIsInstance(manifest["files"], dict)
                    self.assertEqual(
                        set(manifest["bundle_digests"]),
                        {"model.h17p.aimodelc", "model.h15g.aimodelc"},
                    )
                    self.assertEqual(manifest["platform"], "iOS")
                    self.assertEqual(set(manifest["archs"]), {"h17p", "h15g"})
                    if aot["base_dir"]:
                        for bundle in manifest["archs"].values():
                            self.assertTrue(
                                (roots[1] / aot["base_dir"] / bundle).is_dir()
                            )

    def test_fixture_cli_passes_aot_platform_and_repeated_architectures(self):
        from executorch.backends.apple.coreai.runtime.test import export_fixtures

        with mock.patch(
            "sys.argv",
            [
                "export_fixtures",
                "--output-dir",
                "/tmp/coreai-cli-test",
                "--aot-platform",
                "iOS",
                "--aot-architecture",
                "h17p",
                "--aot-architecture",
                "h15g",
                "--static-only",
            ],
        ), mock.patch.object(
            export_fixtures, "export_fixtures", return_value=[]
        ) as export:
            export_fixtures.main()
        export.assert_called_once_with(
            Path("/tmp/coreai-cli-test"),
            include_dynamic=False,
            aot_compile_config=AOTCompileConfig(
                platform="iOS", architectures=["h17p", "h15g"]
            ),
        )

    def test_fixture_cli_defaults_to_source(self):
        from executorch.backends.apple.coreai.runtime.test import export_fixtures

        with mock.patch(
            "sys.argv", ["export_fixtures", "--output-dir", "/tmp/coreai-cli-test"]
        ), mock.patch.object(
            export_fixtures, "export_fixtures", return_value=[]
        ) as export:
            export_fixtures.main()
        export.assert_called_once_with(
            Path("/tmp/coreai-cli-test"), include_dynamic=True, aot_compile_config=None
        )

    def test_fixture_cli_architecture_requires_platform(self):
        from executorch.backends.apple.coreai.runtime.test import export_fixtures

        with mock.patch(
            "sys.argv",
            [
                "export_fixtures",
                "--output-dir",
                "/tmp/coreai-cli-test",
                "--aot-architecture",
                "h17p",
            ],
        ), mock.patch("sys.stderr"), mock.patch.object(
            export_fixtures, "export_fixtures"
        ) as export:
            with self.assertRaises(SystemExit) as error:
                export_fixtures.main()
        self.assertEqual(error.exception.code, 2)
        export.assert_not_called()


def _lower_linear(partitioner):
    model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32)).eval()
    ep = torch.export.export(model, (torch.randn(2, 32),))
    return to_edge_transform_and_lower(ep, partitioner=[partitioner])


def _lowered_manifest(lowered):
    lbms = get_lowered_backend_modules(lowered.exported_program().graph_module)
    assert len(lbms) == 1, lbms
    return lbms[0], json.loads(bytes(lbms[0].processed_bytes))


@unittest.skipUnless(
    _coreai_build_available(),
    "requires macOS with the Metal Toolchain (xcrun coreai-build)",
)
class CoreAIAOTCompileTest(unittest.TestCase):
    """Real coreai-build integration (only runs when the toolchain is present)."""

    def test_aot_inline_embeds_aimodelc(self):
        lowered = _lower_linear(
            CoreAIPartitioner(aot_compile_config=AOTCompileConfig(platform="macOS"))
        )
        lbm, manifest = _lowered_manifest(lowered)
        self.assertEqual(manifest["packaging"], "aot_compiled_inline")
        self.assertEqual(manifest["platform"], "macOS")
        self.assertGreaterEqual(len(manifest["archs"]), 1)
        self.assertTrue(any(".aimodelc/" in f for f in manifest["files"]))
        self.assertIsNotNone(lbm.named_data_store_output)
        _validate_asset_metadata(manifest, None, lbm.named_data_store_output)
        self.assertGreater(len(bytes(lowered.to_executorch().buffer)), 0)

    def test_aot_sidecar_writes_aimodelc_and_keeps_pte_small(self):
        with tempfile.TemporaryDirectory() as sidecar:
            with coreai_sidecar_dir(sidecar):
                lowered = _lower_linear(
                    CoreAIPartitioner(
                        aot_compile_config=AOTCompileConfig(platform="macOS"),
                        uses_sidecar=True,
                    )
                )
                lbm, manifest = _lowered_manifest(lowered)
                pte = bytes(lowered.to_executorch().buffer)

            self.assertEqual(manifest["packaging"], "aot_compiled_sidecar")
            self.assertGreaterEqual(len(manifest["archs"]), 1)
            # Sidecar: compiled bundle *contents* must not be embedded in the .pte.
            self.assertIsNone(lbm.named_data_store_output)
            # The manifest contains asset filenames, but not the compiled payloads.
            self.assertLess(len(pte), sum(manifest["files"].values()))
            _validate_asset_metadata(manifest, Path(sidecar), None)
            for rel in manifest["archs"].values():
                bundle = os.path.join(sidecar, rel)
                self.assertTrue(os.path.isdir(bundle), bundle)
                self.assertTrue(bundle.endswith(".aimodelc"))

    def test_delegates_present(self):
        lowered = _lower_linear(
            CoreAIPartitioner(aot_compile_config=AOTCompileConfig(platform="macOS"))
        )
        gm = lowered.exported_program().graph_module
        self.assertTrue(
            any(
                n.op == "call_function" and n.target is executorch_call_delegate
                for n in gm.graph.nodes
            )
        )


if __name__ == "__main__":
    unittest.main()
