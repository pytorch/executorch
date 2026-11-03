# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import subprocess
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn
from executorch.backends.apple.coreai import (
    AOTCompileConfig,
    get_default_compile_config,
    get_default_passes,
)
from executorch.backends.apple.coreai.compiler.preprocess import COMPILE_SPEC_KEYS
from executorch.backends.apple.coreai.partition.partitioner import (
    CoreAIPartitioner,
    do_not_delegate,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.lowered_backend_module import get_lowered_backend_modules
from executorch.exir.pass_base import PassResult

_MOCK_BUILD = "executorch.backends.apple.coreai.compiler.preprocess._run_coreai_build"
_WIDTH = 256


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


class _Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(_WIDTH, _WIDTH)
        self.down = nn.Linear(_WIDTH, _WIDTH)

    def forward(self, x):
        return self.down(torch.relu(self.up(x)) * 2.0)


class _BreakAtMul:
    def __call__(self, gm):
        for node in gm.graph.nodes:
            if node.op == "call_function" and "mul" in str(node.target):
                do_not_delegate(node)
        return PassResult(gm, True)


def _inputs():
    return {"forward": (torch.randn(1, _WIDTH),), "prefill": (torch.randn(8, _WIDTH),)}


def _lower(partitioner, *, graph_break=False, model=None, inputs=None):
    torch.manual_seed(0)
    model = model or _Stack().eval()
    inputs = inputs or _inputs()
    passes = list(get_default_passes()) + ([_BreakAtMul()] if graph_break else [])
    return to_edge_transform_and_lower(
        {name: torch.export.export(model, args) for name, args in inputs.items()},
        transform_passes=passes,
        partitioner=partitioner,
        compile_config=get_default_compile_config(),
    )


def _manifests(lowered, method):
    graph = lowered.exported_program(method).graph_module
    return [
        (json.loads(bytes(lbm.processed_bytes)), lbm)
        for lbm in get_lowered_backend_modules(graph)
    ]


def _bundle_bytes(lbm):
    return sum(len(blob) for blob in lbm.named_data_store_output.buffers)


class MultifunctionPartitionerTest(unittest.TestCase):
    def test_option_adds_compile_spec(self):
        keys = [
            spec.key
            for spec in CoreAIPartitioner(
                multifunction=True
            ).delegation_spec.compile_specs
        ]
        self.assertIn(COMPILE_SPEC_KEYS.MULTIFUNCTION.value, keys)
        default_keys = [
            spec.key for spec in CoreAIPartitioner().delegation_spec.compile_specs
        ]
        self.assertNotIn(COMPILE_SPEC_KEYS.MULTIFUNCTION.value, default_keys)

    def test_rejects_input_enumerations(self):
        with self.assertRaisesRegex(ValueError, "input_enumerations"):
            CoreAIPartitioner(multifunction=True, input_enumerations=[{0: [1, 8]}])


class MultifunctionPreprocessTest(unittest.TestCase):
    def test_methods_share_one_bundle(self):
        lowered = _lower([CoreAIPartitioner(multifunction=True)])
        ((forward, forward_lbm),) = _manifests(lowered, "forward")
        ((prefill, prefill_lbm),) = _manifests(lowered, "prefill")

        self.assertEqual(forward["function"], "forward_0")
        self.assertEqual(prefill["function"], "prefill_0")
        self.assertEqual(forward["packaging"], "aimodel")
        self.assertIs(forward["multifunction"], True)
        self.assertIs(prefill["multifunction"], True)
        for key in ("export_hash", "path", "files", "bundle_digests"):
            self.assertEqual(forward[key], prefill[key], key)
        self.assertEqual(forward["input_names"], ["input_0"])
        self.assertEqual(prefill["output_names"], ["output_0"])
        self.assertEqual(
            set(forward_lbm.named_data_store_output.pte_data),
            set(prefill_lbm.named_data_store_output.pte_data),
        )

        # Shared weights are stored once: two functions cost about one bundle.
        ((single, single_lbm),) = _manifests(
            _lower(
                [CoreAIPartitioner()],
                inputs={"forward": _inputs()["forward"]},
            ),
            "forward",
        )
        self.assertEqual(single["function"], "main")
        self.assertLess(_bundle_bytes(forward_lbm), 1.1 * _bundle_bytes(single_lbm))
        self.assertGreater(_bundle_bytes(single_lbm), 2 * _WIDTH * _WIDTH * 4)

    def test_graph_break_partitions_become_functions(self):
        lowered = _lower([CoreAIPartitioner(multifunction=True)], graph_break=True)
        functions = {
            method: [m["function"] for m, _ in _manifests(lowered, method)]
            for method in ("forward", "prefill")
        }
        self.assertEqual(
            functions,
            {
                "forward": ["forward_0", "forward_1"],
                "prefill": ["prefill_0", "prefill_1"],
            },
        )
        hashes = {
            m["export_hash"]
            for method in ("forward", "prefill")
            for m, _ in _manifests(lowered, method)
        }
        self.assertEqual(len(hashes), 1)

    def test_disabled_keeps_one_model_per_delegate(self):
        lowered = _lower([CoreAIPartitioner()])
        ((forward, _),) = _manifests(lowered, "forward")
        ((prefill, _),) = _manifests(lowered, "prefill")
        self.assertEqual((forward["function"], prefill["function"]), ("main", "main"))
        self.assertNotIn("multifunction", forward)
        self.assertNotEqual(forward["export_hash"], prefill["export_hash"])

    def test_mismatched_compile_specs_raise(self):
        partitioners = {
            "forward": [CoreAIPartitioner(multifunction=True)],
            "prefill": [
                CoreAIPartitioner(multifunction=True, min_deployment_version="27.0")
            ],
        }
        with self.assertRaisesRegex(ValueError, "identical Core AI compile specs"):
            _lower(partitioners)

    def test_option_on_only_some_methods_raises(self):
        partitioners = {
            "forward": [CoreAIPartitioner(multifunction=True)],
            "prefill": [CoreAIPartitioner()],
        }
        with self.assertRaisesRegex(ValueError, r"\['prefill'\]"):
            _lower(partitioners)

    @mock.patch(_MOCK_BUILD)
    def test_aot_compiled_methods_share_archs(self, build):
        def fake_build(aimodel_path, out_dir, opts):
            for arch in opts["architectures"]:
                bundle = out_dir / f"model.{arch}.aimodelc"
                bundle.mkdir(parents=True)
                (bundle / "model.mil").write_bytes(b"fake-compiled")

        build.side_effect = fake_build
        aot = AOTCompileConfig(platform="iOS", architectures=["h15g", "h16"])
        lowered = _lower(
            [CoreAIPartitioner(multifunction=True, aot_compile_config=aot)]
        )
        ((forward, _),) = _manifests(lowered, "forward")
        ((prefill, _),) = _manifests(lowered, "prefill")
        self.assertEqual(build.call_count, 1)
        self.assertEqual(forward["packaging"], "aimodelc")
        self.assertEqual(sorted(forward["archs"]), ["h15g", "h16"])
        self.assertEqual(forward["archs"], prefill["archs"])
        self.assertEqual(
            (forward["function"], prefill["function"]), ("forward_0", "prefill_0")
        )


def _runtime_available():
    try:
        from executorch.extension.pybindings.portable_lib import _is_available
    except ImportError:
        return False
    return _is_available("CoreAIBackend")


@unittest.skipUnless(
    _coreai_build_available(),
    "requires macOS with the Metal Toolchain (xcrun coreai-build)",
)
class MultifunctionAOTCompileTest(unittest.TestCase):
    def test_real_build_shares_compiled_bundles(self):
        aot = AOTCompileConfig(platform="macOS", architectures=["h16c"])
        lowered = _lower(
            [CoreAIPartitioner(multifunction=True, aot_compile_config=aot)],
            graph_break=True,
        )
        manifests = [
            m
            for method in ("forward", "prefill")
            for m, _ in _manifests(lowered, method)
        ]
        self.assertEqual(
            [m["function"] for m in manifests],
            ["forward_0", "forward_1", "prefill_0", "prefill_1"],
        )
        self.assertEqual(len({json.dumps(m["archs"]) for m in manifests}), 1)
        self.assertEqual(list(manifests[0]["archs"]), ["h16c"])


@unittest.skipUnless(_runtime_available(), "requires the Core AI Python runtime")
class MultifunctionRuntimeTest(unittest.TestCase):
    def _run_all(self, lowered, inputs):
        from executorch.extension.pybindings.portable_lib import (
            _load_program_from_buffer,
            Verification,
        )

        buffer = lowered.to_executorch().buffer
        outputs = {}
        with tempfile.TemporaryDirectory(prefix="coreai-multifunction-") as cache:
            program = _load_program_from_buffer(
                buffer,
                program_verification=Verification.Minimal,
                backend_options={"CoreAIBackend": {"coreai_cache_dir": cache}},
            )
            methods = {name: program.load_method(name) for name in inputs}
            for name, args in inputs.items():
                outputs[name] = methods[name].call(args, clone_outputs=True)[0]
            del methods, program
        return outputs

    def _check(self, graph_break):
        torch.manual_seed(0)
        model = _Stack().eval()
        inputs = _inputs()
        lowered = _lower(
            [CoreAIPartitioner(multifunction=True)],
            graph_break=graph_break,
            model=model,
            inputs=inputs,
        )
        outputs = self._run_all(lowered, inputs)
        with torch.no_grad():
            for name, args in inputs.items():
                torch.testing.assert_close(
                    outputs[name], model(*args), atol=1e-2, rtol=1e-2, msg=name
                )

    def test_each_method_runs_its_function(self):
        self._check(graph_break=False)

    def test_graph_break_methods_run(self):
        self._check(graph_break=True)


if __name__ == "__main__":
    unittest.main()
