# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ``CoreAIBackend.preprocess`` and its AOT-compile config.

Covers both asset formats, portable ``.aimodel`` and AOT-compiled
``.aimodelc``, embedded in the .pte, plus ``AOTCompileConfig`` parsing. The
compiled-delivery cases mock
``coreai-build`` so they run without the Metal Toolchain; the real-toolchain
integration is in :class:`CoreAIAOTCompileTest` (gated on ``coreai-build``).
"""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

from executorch.backends.apple.coreai.compiler.preprocess import (
    _aot_compile_options,
    _asset_metadata,
    AOTCompileConfig,
    COMPILE_SPEC_KEYS,
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

    def test_inline_keys_keep_the_bundle_directory(self):
        """The bundle name has to survive flattening into NamedDataStore keys.

        Otherwise unpacking cannot tell that these files belong to a ``.aimodel``.
        """
        manifest = json.loads(
            CoreAIBackend.preprocess(_edge_program(), []).processed_bytes
        )
        self.assertTrue(
            all(rel.startswith("model.aimodel/") for rel in manifest["files"]),
            manifest["files"],
        )


class ManifestBindingsTest(unittest.TestCase):
    def test_names_follow_boundary_order_past_single_digits(self):
        class ManyInputs(nn.Module):
            def forward(self, *args):
                return tuple(x - float(i) for i, x in enumerate(args))

        edge = to_edge(
            torch.export.export(ManyInputs(), tuple(torch.ones(2) for _ in range(12)))
        ).exported_program()
        manifest = json.loads(CoreAIBackend.preprocess(edge, []).processed_bytes)
        self.assertEqual(manifest["function"], "main")
        self.assertEqual(manifest["input_names"], [f"input_{i}" for i in range(12)])
        self.assertEqual(manifest["output_names"], [f"output_{i}" for i in range(12)])


class AssetMetadataTest(unittest.TestCase):
    @mock.patch(_MOCK_BUILD, side_effect=_fake_run_coreai_build)
    def test_sizes_and_digests_cover_every_delivered_file(self, _build):
        aot = _aot_spec({"platform": "iOS", "architectures": ["h15g", "h16"]})
        for specs, bundles in (
            ([], {"model.aimodel"}),
            ([aot], {"model.h15g.aimodelc", "model.h16.aimodelc"}),
        ):
            with self.subTest(aot=bool(specs)):
                result = CoreAIBackend.preprocess(_edge_program(), specs)
                manifest = json.loads(result.processed_bytes)
                out = result.data_store_output
                prefix = f"coreai/{manifest['hash']}/"
                embedded = {
                    key[len(prefix) :]: len(out.buffers[entry.buffer_index])
                    for key, entry in out.pte_data.items()
                }
                self.assertEqual(manifest["files"], embedded)
                self.assertEqual(set(manifest["bundle_digests"]), bundles)

    def test_digest_changes_with_file_contents(self):
        with tempfile.TemporaryDirectory() as d:
            weights = Path(d) / "model.aimodel" / "weights.bin"
            weights.parent.mkdir()
            weights.write_bytes(b"\x00\x01")
            before = _asset_metadata(Path(d))
            weights.write_bytes(b"\x01\x00")
            after = _asset_metadata(Path(d))
        self.assertEqual(before["files"], after["files"])
        self.assertNotEqual(before["bundle_digests"], after["bundle_digests"])


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

    def test_portable_reports_the_floor_it_can_apply(self):
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
        self.assertEqual(manifest["platform"], "iOS")
        self.assertEqual(sorted(manifest["archs"]), ["h15g", "h16"])
        # Compiled bundle contents are embedded (files under a *.aimodelc dir).
        self.assertTrue(any(".aimodelc/" in f for f in manifest["files"]))
        self.assertIsNotNone(result.data_store_output)

    @mock.patch(_MOCK_BUILD, side_effect=_fake_run_coreai_build)
    def test_archs_map_points_at_embedded_bundles(self, _build):
        """Each ``archs`` value is the bundle's key under ``coreai/``."""
        config = {"platform": "iOS", "architectures": ["h15g", "h16"]}
        result = CoreAIBackend.preprocess(_edge_program(), [_aot_spec(config)])
        manifest = json.loads(result.processed_bytes)
        for arch, rel in manifest["archs"].items():
            self.assertEqual(rel, f"{manifest['hash']}/model.{arch}.aimodelc")
            self.assertIn(f"coreai/{rel}/model.mil", result.data_store_output.pte_data)

    @mock.patch(_MOCK_BUILD, side_effect=_fake_run_coreai_build)
    def test_defaults_to_all_architectures(self, _build):
        # Empty architectures => coreai-build decides (our fake yields one).
        result = CoreAIBackend.preprocess(_edge_program(), [_aot_spec({})])
        manifest = json.loads(result.processed_bytes)
        self.assertEqual(manifest["packaging"], "aot_compiled_inline")
        self.assertEqual(manifest["platform"], "macOS")  # default
        self.assertTrue(manifest["archs"])

    @mock.patch(_MOCK_BUILD, side_effect=_fake_run_coreai_build)
    def test_two_delegates_get_separate_asset_keys(self, _build):
        """A graph break gives two delegates, whose assets must not collide."""
        lowered = _lower_with_break(
            CoreAIPartitioner(
                aot_compile_config=AOTCompileConfig(
                    platform="iOS", architectures=["h15g"]
                ),
            )
        )
        lbms = get_lowered_backend_modules(lowered.exported_program().graph_module)
        self.assertEqual(len(lbms), 2)
        hashes = {json.loads(bytes(lbm.processed_bytes))["hash"] for lbm in lbms}
        self.assertEqual(len(hashes), 2)


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
