# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import importlib
import sys
import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from executorch.backends.native.test.samples.cases import (
    Case,
    chain,
    draws,
    suite_cases,
)
from executorch.backends.native.test.samples.generate import (
    available_cpus,
    available_memory,
    BUDGET_ENV,
    memory_budget,
    WORKER_BYTES,
    worker_count,
)
from executorch.backends.native.test.samples.record import DTYPES, record
from executorch.backends.native.test.samples.schemes import (
    Gguf,
    GGUF_SUPER_SCALES,
    Scheme,
    schemes,
    Torchao,
)
from executorch.exir.native import read_samples
from executorch.extension.llm.export.gguf import ExportableGGUFTensor
from torch.export import Dim
from torchao.quantization import (
    IntxUnpackedToInt8Tensor,
    IntxWeightOnlyConfig,
    quantize_,
)
from torchao.quantization.granularity import PerGroup


# A host with 64 GiB available.
_HOST = {"proc/meminfo": "MemTotal: 134217728 kB\nMemAvailable: 67108864 kB\n"}
# /proc/self/mountinfo lines of the usual cgroup v2 and v1 memory mounts.
_V2_MOUNT = "30 23 0:26 / /sys/fs/cgroup rw shared:4 - cgroup2 cgroup2 rw\n"
_V1_MEMORY_MOUNT = (
    "35 25 0:31 / /sys/fs/cgroup/memory rw shared:9 - cgroup cgroup rw,memory\n"
)


def _available_memory(files: dict[str, str]) -> int | None:
    """available_memory() on a root holding only `files`."""
    with tempfile.TemporaryDirectory() as root:
        for name, text in files.items():
            path = Path(root, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return available_memory(Path(root))


def ops() -> dict:
    """Every op's samples; skips the test where FACTO, which the linear op's
    cases import, is missing (OSS puts it on the path only to build the op
    samples), or where transformers lacks what an op's cases import (OSS unit
    tests run an older transformers than Native needs)."""
    try:
        module = importlib.import_module("executorch.backends.native.test.samples.ops")
    except ImportError as e:
        package = (e.name or "").partition(".")[0]
        if package == "facto" and isinstance(e, ModuleNotFoundError):
            raise unittest.SkipTest("op samples need FACTO") from e
        if package == "transformers":
            raise unittest.SkipTest(f"op samples need a newer transformers: {e}") from e
        raise
    return module.ops()


class RecordTest(unittest.TestCase):
    def test_suite_cases_raises_on_a_model_eager_rejects(self) -> None:
        module = ModuleType("suite")
        module.test_mismatch = lambda runner: runner.lower_and_run_model(
            torch.nn.Linear(4, 2), (torch.randn(1, 3),)
        )
        with self.assertRaises(RuntimeError) as raised:
            suite_cases(module, samples=1)
        self.assertIn(
            "while recording suite case mismatch_0", raised.exception.__notes__
        )

    def test_suite_cases_raises_when_seeds_lower_different_shapes(self) -> None:
        module = ModuleType("suite")
        sizes = iter([2, 3])
        module.test_varying = lambda runner: runner.lower_and_run_model(
            torch.nn.Identity(), (torch.randn(next(sizes)),)
        )
        with self.assertRaises(ValueError):
            suite_cases(module, samples=2)

    def test_suite_cases_skips_skipped_tests(self) -> None:
        module = ModuleType("suite")
        module.test_lowered = lambda runner: runner.lower_and_run_model(
            torch.nn.Identity(), (torch.randn(2),)
        )

        def test_skipped(runner) -> None:
            raise unittest.SkipTest("unsupported")

        module.test_skipped = test_skipped
        cases = suite_cases(module, samples=1)
        self.assertEqual([c.name for c in cases], ["lowered_0"])

    def test_suite_cases_raises_when_no_test_lowers_a_model(self) -> None:
        with self.assertRaises(ValueError):
            suite_cases(ModuleType("suite"), samples=1)

    def test_record_raises_on_duplicate_case_names(self) -> None:
        case = Case("same", torch.nn.Linear(2, 2), draws(1, (1, 2)))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                record([case, case], Scheme(), Path(tmp) / "dup.ptn")

    def test_record_writes_no_package_when_the_scheme_fits_no_case(self) -> None:
        case = Case("relu", torch.nn.ReLU(), draws(1, (1, 32)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "relu.ptn"
            path.write_bytes(b"stale")
            record([case], schemes()["int4wo_g32"], path)
            self.assertFalse(path.exists())

    def test_record_golden_is_eager_and_precise_is_float64(self) -> None:
        model = torch.nn.Linear(16, 4)
        (x,) = draws(1, (3, 16))[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "linear.ptn"
            record([Case("linear", model, [(x,)])], Scheme(), path, dtypes=["bf16"])
            sets = read_samples(str(path))
        x, w, b = (t.detach().bfloat16() for t in (x, model.weight, model.bias))
        torch.testing.assert_close(
            sets["golden"].samples[0].calls[0].outputs[0],
            F.linear(x, w, b),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            sets["precise"].samples[0].calls[0].outputs[0],
            F.linear(x.double(), w.double(), b.double()).float(),
            rtol=0,
            atol=0,
        )

    def test_record_adds_the_8da4w_golden_to_weight_only_packages(self) -> None:
        case = Case("linear", torch.nn.Linear(64, 8), draws(2, (3, 64)))
        sets = {}
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("float", "int4wo_g32", "8da4w_g32"):
                path = Path(tmp) / f"{name}.ptn"
                record([case], schemes()[name], path, dtypes=["f32"])
                sets[name] = read_samples(str(path))
        self.assertNotIn("golden-int8_act", sets["float"])
        self.assertNotIn("golden-int8_act", sets["8da4w_g32"])
        for got, want in zip(
            sets["int4wo_g32"]["golden-int8_act"].samples,
            sets["8da4w_g32"]["golden"].samples,
            strict=True,
        ):
            torch.testing.assert_close(
                got.calls[0].outputs, want.calls[0].outputs, rtol=0, atol=0
            )

    def test_record_adds_no_8da4w_golden_without_a_linear(self) -> None:
        inputs = draws(1, (3,), sampler=partial(torch.randint, 8))
        case = Case("embedding", torch.nn.Embedding(8, 64), inputs)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "embedding.ptn"
            record([case], schemes()["int4wo_g32"], path, dtypes=["f32"])
            self.assertEqual(set(read_samples(str(path))), {"golden", "precise"})

    def test_tests_of_every_op_skip_only_without_facto_or_transformers(
        self,
    ) -> None:
        # Another package named facto on the path leaves facto.inputgen missing.
        # An older transformers lacks transformers.integrations.gguf, or a name
        # in a module it has.
        expected = {
            ModuleNotFoundError("", name="facto"): unittest.SkipTest,
            ModuleNotFoundError("", name="facto.inputgen"): unittest.SkipTest,
            ModuleNotFoundError(
                "", name="transformers.integrations.gguf"
            ): unittest.SkipTest,
            ImportError("", name="transformers.models.qwen3_5"): unittest.SkipTest,
            ModuleNotFoundError("", name="gguf"): ModuleNotFoundError,
            ImportError("", name="facto.inputgen"): ImportError,
        }
        for error, raised in expected.items():
            with self.subTest(error=repr(error), name=error.name), patch(
                "importlib.import_module", side_effect=error
            ), self.assertRaises((unittest.SkipTest, ImportError)) as caught:
                ops()
            self.assertIs(type(caught.exception), raised)

    def test_generate_imports_without_the_op_cases(self) -> None:
        # OSS puts FACTO, which the linear op's cases import, on the path only
        # to build the op samples; generate's memory helpers must import
        # without it.
        name = "executorch.backends.native.test.samples.generate"
        blocked = {"executorch.backends.native.test.samples.ops": None}
        with patch.dict(sys.modules, blocked):
            del sys.modules[name]
            generate = importlib.import_module(name)
        self.assertEqual(generate.WORKER_BYTES, WORKER_BYTES)

    def test_generate_runs_no_more_workers_than_memory_holds(self) -> None:
        self.assertEqual(worker_count(100, 64, 5 * WORKER_BYTES + 1), 5)
        self.assertEqual(worker_count(100, 64, WORKER_BYTES - 1), 1)
        self.assertEqual(worker_count(100, 3, 64 * WORKER_BYTES), 3)
        self.assertEqual(worker_count(2, 64, 64 * WORKER_BYTES), 2)
        self.assertEqual(worker_count(1, 1, None), 1)

    def test_generate_memory_budget_overrides_the_host(self) -> None:
        self.assertEqual(memory_budget(None, {BUDGET_ENV: "8"}), 8 << 30)
        self.assertEqual(memory_budget(4.0, {BUDGET_ENV: "8"}), 4 << 30)
        with patch(
            "executorch.backends.native.test.samples.generate.available_memory",
            return_value=123,
        ):
            self.assertEqual(memory_budget(None, {}), 123)

    def test_available_memory_is_the_host_s_without_a_cgroup_limit(self) -> None:
        self.assertEqual(_available_memory(_HOST), 64 << 30)
        files = {"proc/self/cgroup": "0::/\n", "proc/self/mountinfo": _V2_MOUNT}
        self.assertEqual(_available_memory({**_HOST, **files}), 64 << 30)
        self.assertIsNone(_available_memory({"proc/self/cgroup": "0::/\n"}))

    def test_available_memory_reads_the_limit_of_the_process_cgroup(self) -> None:
        # A systemd scope: its own memory.max, none at the cgroup root.
        scope = "sys/fs/cgroup/user.slice/run-1.scope"
        files = {
            "proc/self/cgroup": "0::/user.slice/run-1.scope\n",
            "proc/self/mountinfo": _V2_MOUNT,
            f"{scope}/memory.max": f"{4 << 30}\n",
            f"{scope}/memory.current": f"{1 << 29}\n",
            f"{scope}/memory.stat": "anon 536870912\ninactive_file 0\n",
        }
        self.assertEqual(_available_memory({**_HOST, **files}), 7 << 29)

    def test_available_memory_takes_the_tightest_ancestor_limit(self) -> None:
        slice_ = "sys/fs/cgroup/user.slice"
        scope = f"{slice_}/user-1.slice/run-1.scope"
        files = {
            "proc/self/cgroup": "0::/user.slice/user-1.slice/run-1.scope\n",
            "proc/self/mountinfo": _V2_MOUNT,
            f"{slice_}/memory.max": f"{2 << 30}\n",
            f"{slice_}/memory.current": f"{1 << 30}\n",
            f"{slice_}/memory.stat": "inactive_file 0\n",
            f"{slice_}/user-1.slice/memory.max": "max\n",
            f"{scope}/memory.max": f"{8 << 30}\n",
            f"{scope}/memory.current": f"{1 << 29}\n",
            f"{scope}/memory.stat": "inactive_file 0\n",
        }
        self.assertEqual(_available_memory({**_HOST, **files}), 1 << 30)

    def test_available_memory_reads_cgroup_v1_limits(self) -> None:
        root = "sys/fs/cgroup/memory"
        files = {
            "proc/self/cgroup": "12:memory:/user.slice\n"
            "3:cpu,cpuacct:/user.slice\n1:name=systemd:/user.slice\n",
            "proc/self/mountinfo": _V1_MEMORY_MOUNT,
            # v1's "unlimited" is a huge number, not "max".
            f"{root}/memory.limit_in_bytes": "9223372036854771712\n",
            f"{root}/memory.usage_in_bytes": f"{100 << 30}\n",
            f"{root}/memory.stat": "total_inactive_file 0\n",
            f"{root}/user.slice/memory.limit_in_bytes": f"{4 << 30}\n",
            f"{root}/user.slice/memory.usage_in_bytes": f"{3 << 30}\n",
            f"{root}/user.slice/memory.stat": "inactive_file 0\n"
            f"total_inactive_file {1 << 30}\n",
        }
        self.assertEqual(_available_memory({**_HOST, **files}), 2 << 30)

    def test_available_memory_counts_page_cache_as_available(self) -> None:
        # A container's own cgroup namespace, mostly filled with page cache.
        files = {
            "proc/self/cgroup": "0::/\n",
            "proc/self/mountinfo": _V2_MOUNT,
            "sys/fs/cgroup/memory.max": f"{4 << 30}\n",
            "sys/fs/cgroup/memory.current": f"{7 << 29}\n",
            "sys/fs/cgroup/memory.stat": f"anon {1 << 29}\ninactive_file {3 << 30}\n",
        }
        self.assertEqual(_available_memory({**_HOST, **files}), 7 << 29)

    def test_available_memory_finds_cgroup_mounts_in_mountinfo(self) -> None:
        def limit(directory: str, v1: bool = False) -> dict[str, str]:
            names = ("limit_in_bytes", "usage_in_bytes") if v1 else ("max", "current")
            return {
                f"{directory}/memory.{names[0]}": f"{4 << 30}\n",
                f"{directory}/memory.{names[1]}": f"{1 << 29}\n",
                f"{directory}/memory.stat": "inactive_file 0\ntotal_inactive_file 0\n",
            }

        fixtures = {
            # systemd's hybrid layout: cgroup2 at unified, under a tmpfs.
            "hybrid_unified": {
                "proc/self/cgroup": "0::/user.slice/run-1.scope\n",
                "proc/self/mountinfo": (
                    "25 1 0:23 / /sys/fs/cgroup ro - tmpfs tmpfs ro\n"
                    "30 25 0:26 / /sys/fs/cgroup/unified rw - cgroup2 cgroup2 rw\n"
                ),
                **limit("sys/fs/cgroup/unified/user.slice/run-1.scope"),
            },
            "v1_memory_mounted_elsewhere": {
                "proc/self/cgroup": "5:memory:/user.slice\n",
                "proc/self/mountinfo": (
                    "40 1 0:40 / /cgroup/memory rw - cgroup cgroup rw,memory\n"
                ),
                **limit("cgroup/memory/user.slice", v1=True),
            },
            "v1_combined_controllers": {
                "proc/self/cgroup": "4:cpu,memory:/user.slice\n",
                "proc/self/mountinfo": "36 25 0:32 / /sys/fs/cgroup/cpu,memory rw - "
                "cgroup cgroup rw,cpu,memory\n",
                **limit("sys/fs/cgroup/cpu,memory/user.slice", v1=True),
            },
            # A container without its own cgroup namespace: its cgroup is the
            # mount's root.
            "mount_root_below_the_hierarchy": {
                "proc/self/cgroup": "0::/docker/abc/worker\n",
                "proc/self/mountinfo": "30 23 0:26 /docker/abc /sys/fs/cgroup rw - "
                "cgroup2 cgroup2 rw\n",
                **limit("sys/fs/cgroup/worker"),
            },
        }
        for name, files in fixtures.items():
            with self.subTest(name):
                self.assertEqual(_available_memory({**_HOST, **files}), 7 << 29)

    def test_available_memory_warns_when_the_memory_cgroup_is_unmounted(self) -> None:
        fixtures = {
            "v2_without_a_cgroup2_mount": {
                "proc/self/cgroup": "0::/user.slice/run-1.scope\n",
                "proc/self/mountinfo": (
                    "25 1 0:23 / /sys/fs/cgroup ro - tmpfs tmpfs ro\n"
                ),
            },
            "v1_memory_without_its_mount": {
                "proc/self/cgroup": "5:memory:/user.slice\n0::/user.slice\n",
                "proc/self/mountinfo": _V2_MOUNT,
            },
        }
        for name, files in fixtures.items():
            with self.subTest(name):
                with self.assertLogs(level="WARNING") as logs:
                    self.assertEqual(_available_memory({**_HOST, **files}), 64 << 30)
                self.assertIn("--memory-budget-gb", logs.output[0])

    def test_generate_counts_the_cpus_the_process_may_use(self) -> None:
        for fake, want in (
            (
                SimpleNamespace(
                    process_cpu_count=lambda: 3,
                    sched_getaffinity=lambda _: {0, 1},
                    cpu_count=lambda: 64,
                ),
                3,
            ),
            (
                SimpleNamespace(
                    sched_getaffinity=lambda _: {0, 1}, cpu_count=lambda: 64
                ),
                2,
            ),
            (SimpleNamespace(cpu_count=lambda: 64), 64),
            (SimpleNamespace(cpu_count=lambda: None), 1),
        ):
            with (
                self.subTest(want=want),
                patch("executorch.backends.native.test.samples.generate.os", fake),
            ):
                self.assertEqual(available_cpus(), want)

    def test_generate_caps_workers_at_the_windows_process_pool_limit(self) -> None:
        with patch.object(sys, "platform", "win32"):
            self.assertEqual(worker_count(100, 128, None), 61)
        with patch.object(sys, "platform", "linux"):
            self.assertEqual(worker_count(100, 128, None), 100)

    def test_generate_rejects_a_budget_that_holds_no_worker(self) -> None:
        self.assertEqual(memory_budget(2.0, {}), WORKER_BYTES)
        for budget in (0.0, -4.0, float("nan"), float("inf"), 1.0):
            with self.subTest(budget=budget), self.assertRaisesRegex(
                ValueError, "--memory-budget-gb"
            ):
                memory_budget(budget, {})
        for text in ("0", "-1", "lots", "1.5"):
            with self.subTest(text=text), self.assertRaisesRegex(
                ValueError, BUDGET_ENV
            ):
                memory_budget(None, {BUDGET_ENV: text})

    def test_core_op_cases_fit_quantized_schemes_only_through_a_linear(self) -> None:
        for op_name, op in ops().items():
            if op_name in ("linear", "embedding"):
                continue
            for case in op.cases():
                quantized = case.name.startswith("quantized_")
                for name, scheme in schemes().items():
                    with self.subTest(op=op_name, case=case.name, scheme=name):
                        self.assertEqual(
                            scheme.fits(case.model), quantized or name == "float"
                        )

    def test_chain_visits_each_dims_bounds_starting_mid_range(self) -> None:
        a, b = Dim("a", min=1, max=8), Dim("b", min=2, max=4)
        case = chain("pair", torch.nn.Identity(), (a, 3), (b, a))
        self.assertEqual(
            [[x.shape for x in xs] for xs in case.inputs],
            [
                [(5, 3), (3, 5)],
                [(1, 3), (4, 1)],
                [(8, 3), (2, 8)],
                [(5, 3), (3, 5)],
            ],
        )
        self.assertEqual(case.dynamic_shapes, ({0: a}, {0: b, 1: a}))

    def test_chain_draws_from_its_sampler(self) -> None:
        case = chain(
            "ids",
            torch.nn.Embedding(8, 2),
            (Dim("tokens", min=1, max=4),),
            sampler=partial(torch.randint, 8),
        )
        for (x,) in case.inputs:
            self.assertEqual(x.dtype, torch.int64)
            self.assertTrue(((x >= 0) & (x < 8)).all())

    def test_record_keeps_integer_inputs_in_every_dtype(self) -> None:
        model = torch.nn.Embedding(8, 4)
        inputs = draws(1, (3,), sampler=partial(torch.randint, 8))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "embedding.ptn"
            record([Case("embedding", model, inputs)], Scheme(), path)
            samples = read_samples(str(path))["golden"].samples
        ((ids,),) = inputs
        self.assertEqual(len(samples), len(DTYPES))
        for sample in samples:
            (call,) = sample.calls
            dtype = DTYPES[call.method.rsplit("_", 1)[1]]
            self.assertEqual(call.inputs[0].dtype, torch.int64)
            torch.testing.assert_close(
                call.outputs[0], model.weight.detach().to(dtype)[ids], rtol=0, atol=0
            )

    def test_record_chains_a_dynamic_case_into_one_sample(self) -> None:
        case = chain("linear", torch.nn.Linear(32, 4), (Dim("batch", min=1, max=6), 32))
        scheme = schemes()["int4wo_g32"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "linear.ptn"
            record([case], scheme, path, dtypes=["f32"])
            sets = read_samples(str(path))
        self.assertEqual(set(sets), {"golden", "precise", "golden-int8_act"})
        for samples in sets.values():
            (sample,) = samples.samples
            self.assertEqual(
                [call.inputs[0].shape for call in sample.calls],
                [(4, 32), (1, 32), (6, 32), (4, 32)],
            )
        model = copy.deepcopy(case.model).eval()
        scheme.quantize(model)
        with torch.no_grad():
            for call, x in zip(
                sets["golden"].samples[0].calls, case.inputs, strict=True
            ):
                torch.testing.assert_close(call.outputs[0], model(*x), rtol=0, atol=0)

    def test_record_keeps_integer_outputs_precise(self) -> None:
        class Argmax(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.argmax(-1)

        case = Case("argmax", Argmax(), draws(1, (2, 8)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "argmax.ptn"
            record([case], Scheme(), path, dtypes=["f32"])
            precise = read_samples(str(path))["precise"].samples
        self.assertEqual(precise[0].calls[0].outputs[0].dtype, torch.int64)

    def test_precise_decodes_gguf_weights_whatever_the_method_dtype(self) -> None:
        # Halves are exact in every recorded dtype and the methods share their
        # GGUF blocks, so their precise outputs must match.
        x = torch.randint(-8, 9, (3, 256)) / 2
        model = torch.nn.Linear(256, 8, bias=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "linear.ptn"
            record([Case("linear", model, [(x,)])], Gguf("q6_k"), path)
            samples = read_samples(str(path))["precise"].samples
        outputs = {s.calls[0].method: s.calls[0].outputs[0] for s in samples}
        for dtype in ("f16", "bf16"):
            torch.testing.assert_close(
                outputs[f"linear_{dtype}"], outputs["linear_f32"], rtol=0, atol=0
            )

    def test_gguf_fits_no_more_linears_than_scale_families(self) -> None:
        def linears(count: int) -> torch.nn.Module:
            return torch.nn.Sequential(
                *(torch.nn.Linear(256, 256) for _ in range(count))
            )

        families = len(GGUF_SUPER_SCALES["q4_k"])
        self.assertTrue(Gguf("q4_k").fits(linears(families)))
        self.assertFalse(Gguf("q4_k").fits(linears(families + 1)))

    def test_weight_only_schemes_quantize_embeddings(self) -> None:
        model = torch.nn.Embedding(8, 256)
        fitting = {name for name, scheme in schemes().items() if scheme.fits(model)}
        self.assertEqual(
            fitting,
            {
                "float",
                "int4wo_g32",
                "int4wo_g128",
                "gguf_q4_0",
                "gguf_q5_0",
                "gguf_q8_0",
                "gguf_q2_k",
                "gguf_q3_k",
                "gguf_q4_k",
                "gguf_q5_k",
                "gguf_q6_k",
            },
        )
        ids = torch.tensor([[0, 7], [3, 3]])
        for name in sorted(fitting - {"float"}):
            scheme = schemes()[name]
            quantized = copy.deepcopy(model).eval()
            scheme.quantize(quantized)
            self.assertIsInstance(
                quantized.weight, (IntxUnpackedToInt8Tensor, ExportableGGUFTensor)
            )
            weight = scheme.decode(quantized.weight)
            torch.testing.assert_close(quantized(ids), weight[ids], rtol=0, atol=0)

    def test_precise_decodes_intx_weights_in_float32(self) -> None:
        # A code times its scale can need more bits than bf16 holds, so decoding
        # in the method's dtype would round the weights.
        x = torch.randint(-8, 9, (3, 64)) / 2
        model = torch.nn.Linear(64, 8, bias=False)
        config = IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(32))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "linear.ptn"
            record([Case("linear", model, [(x,)])], Torchao(config), path)
            samples = read_samples(str(path))["precise"].samples
        outputs = {s.calls[0].method: s.calls[0].outputs[0] for s in samples}
        bf16 = copy.deepcopy(model).to(torch.bfloat16)
        quantize_(bf16, config)
        w = bf16.weight
        codes = (
            w.qdata.double().unflatten(1, (-1, 32)) - w.zero_point.double()[..., None]
        )
        weight = (codes * w.scale.double()[..., None]).flatten(1)
        torch.testing.assert_close(
            outputs["linear_bf16"],
            torch.nn.functional.linear(x.double(), weight).float(),
            rtol=0,
            atol=0,
        )
