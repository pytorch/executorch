# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ``CoreAIBackend.preprocess`` and its AOT-compile config.

Covers all four asset-delivery combinations: portable ``.aimodel`` vs
AOT-compiled ``.aimodelc``, each either inline (embedded in the .pte) or as a
sidecar, plus ``AOTCompileConfig`` parsing. The compiled-delivery cases mock
``coreai-build`` so they run without the Metal Toolchain; the real-toolchain
integration is in :class:`CoreAIAOTCompileTest` (gated on ``coreai-build``).
"""

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
    AOTCompileConfig,
    COMPILE_SPEC_KEYS,
    coreai_sidecar_dir,
    CoreAIBackend,
)
from executorch.backends.apple.coreai.partition.partitioner import CoreAIPartitioner
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


def _edge_program():
    ep = torch.export.export(_Elementwise().eval(), (torch.randn(2, 8),))
    return to_edge(ep).exported_program()


def _fake_run_coreai_build(aimodel_path, out_dir, opts):
    """Stand-in for ``xcrun coreai-build``: drop fake per-arch .aimodelc dirs."""
    for arch in opts["architectures"] or ["h15g"]:
        bundle = Path(out_dir) / f"model.{arch}.aimodelc"
        bundle.mkdir(parents=True, exist_ok=True)
        (bundle / "model.mil").write_bytes(b"fake-compiled")


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

    def test_empty_config_roundtrip(self):
        self.assertEqual(
            AOTCompileConfig.from_json(AOTCompileConfig().to_json()),
            AOTCompileConfig(),
        )


class PortablePreprocessTest(unittest.TestCase):
    """Portable ``.aimodel`` delivery (no coreai-build)."""

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

    def test_sidecar_writes_bundle_and_does_not_embed(self):
        with tempfile.TemporaryDirectory() as d:
            with coreai_sidecar_dir(d):
                result = CoreAIBackend.preprocess(_edge_program(), [_SIDECAR_SPEC])
            manifest = json.loads(result.processed_bytes)
            self.assertEqual(manifest["packaging"], "sidecar")
            # Only the relative, hash-named bundle is referenced (no build path).
            self.assertEqual(manifest["path"], f"{manifest['hash']}.aimodel")
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
            bundles = list(Path(d).glob("*.aimodel"))
            self.assertEqual(len(bundles), 1)
            self.assertTrue((bundles[0] / "main.mlirb").exists())


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
        self.assertEqual(manifest["platform"], "iOS")
        self.assertEqual(sorted(manifest["archs"]), ["h15g", "h16"])
        # Compiled bundle contents are embedded (files under a *.aimodelc dir).
        self.assertTrue(any(".aimodelc/" in f for f in manifest["files"]))
        self.assertIsNotNone(result.data_store_output)

    @mock.patch(_MOCK_BUILD, side_effect=_fake_run_coreai_build)
    def test_sidecar_writes_compiled_bundles(self, _build):
        with tempfile.TemporaryDirectory() as d:
            with coreai_sidecar_dir(d):
                result = CoreAIBackend.preprocess(
                    _edge_program(),
                    [_aot_spec({"architectures": ["h15g"]}), _SIDECAR_SPEC],
                )
            manifest = json.loads(result.processed_bytes)
            self.assertEqual(manifest["packaging"], "aot_compiled_sidecar")
            self.assertEqual(list(manifest["archs"]), ["h15g"])
            self.assertIsNone(result.data_store_output)  # bundles on disk only
            self.assertNotIn(d, result.processed_bytes.decode())
            for rel in manifest["archs"].values():
                bundle = Path(d) / rel
                self.assertTrue(bundle.is_dir())
                self.assertTrue(str(bundle).endswith(".aimodelc"))

    @mock.patch(_MOCK_BUILD, side_effect=_fake_run_coreai_build)
    def test_defaults_to_all_architectures(self, _build):
        # Empty architectures => coreai-build decides (our fake yields one).
        result = CoreAIBackend.preprocess(_edge_program(), [_aot_spec({})])
        manifest = json.loads(result.processed_bytes)
        self.assertEqual(manifest["packaging"], "aot_compiled_inline")
        self.assertEqual(manifest["platform"], "macOS")  # default
        self.assertTrue(manifest["archs"])


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
            self.assertNotIn(b"mpsExecutable", pte)
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
