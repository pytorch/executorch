# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Measured routing (triton/measured_routing.py): the choice among the
implementations of a planned call.

Most tests use a toy family whose "replacement" of the edge ``add`` is the
aten ``add`` and whose measurements are given; the probe tests measure real
calls in the worker process."""

import dataclasses
import importlib.util
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

_TOY = torch.ops.aten.add.Tensor


class _Add(torch.nn.Module):
    def forward(self, x, y):
        return torch.relu(x + y)

    @staticmethod
    def inputs():
        return (
            torch.randn(64, 256, device="cuda"),
            torch.randn(64, 256, device="cuda"),
        )


class _Shared(_Add):
    """An add with a keyword argument, whose first input has another user and
    whose result has two, one of them another add."""

    def forward(self, x, y):
        r = torch.relu(x)
        s = torch.add(r, y, alpha=2.0)
        return s * 2, s + 1, r


class _SamePadded1d(torch.nn.Module):
    """A 1-D convolution Inductor has no template for: it lowers it to an
    extern convolution the runtime lacks."""

    def forward(self, x, w):
        return torch.nn.functional.conv1d(x, w, padding="same")

    @staticmethod
    def inputs():
        return (
            torch.randn(1, 32, 16, device="cuda"),
            torch.randn(32, 32, 4, device="cuda"),
        )


class _Sort(torch.nn.Module):
    """A stable sort too long for Inductor's own sort, which it lowers to the
    aten op (a fallback the runtime provides)."""

    def forward(self, s):
        return torch.sort(s, stable=True)[0]

    @staticmethod
    def inputs():
        return (torch.randn(2, 32768, device="cuda"),)


class _Convs(torch.nn.Module):
    """Calls whose regular lowering the probe classifies: a 3x3 convolution
    (Triton templates), a same-padded 1-D one (an extern convolution the
    runtime lacks), and a sort."""

    def forward(self, x, w3, x1, w4, s):
        return (
            torch.nn.functional.conv2d(x, w3, padding=1),
            torch.nn.functional.conv1d(x1, w4, padding="same"),
            torch.sort(s, stable=True)[0],
        )

    @staticmethod
    def inputs():
        return (
            torch.randn(1, 16, 8, 8, device="cuda"),
            torch.randn(16, 16, 3, 3, device="cuda") / 12,
            *_SamePadded1d.inputs(),
            *_Sort.inputs(),
        )


def _fast(ours, theirs):
    return {"fast": {"ours": ours, "inductor": theirs}, "interleaved": None}


def _entry(ours=(10.0,) * 5, theirs=(20.0,) * 5, interleaved=None, **statuses):
    """A worker result for the toy family: both available and timed unless a
    status says otherwise."""
    candidates = {"inductor": {"status": "available"}, "toy": {"status": "available"}}
    candidates.update(statuses)
    timings = None
    if all(c["status"] == "available" for c in candidates.values()):
        timings = {
            "fast": {"ours": list(ours), "inductor": list(theirs)},
            "interleaved": interleaved,
        }
    return {
        "schema_version": 1,
        "candidates": candidates,
        "timings": timings,
        "cacheable": all(
            c["status"] == "available" or c["cacheable"] for c in candidates.values()
        ),
    }


def _unavailable(cacheable, reason="no valid choices"):
    return {
        "status": "unavailable",
        "reason": reason,
        "error": "RuntimeError",
        "phase": "compile",
        "cacheable": cacheable,
    }


class VerdictTest(unittest.TestCase):
    """The timing protocol's decision, on given samples."""

    def verdict(self, timings, margin=0.15):
        from executorch.backends.cuda.triton.measured_routing import verdict

        return verdict(timings, margin)

    def test_clear_win_needs_no_interleaved_samples(self):
        self.assertEqual(self.verdict(_fast([10.0] * 5, [15.0] * 5)), (True, "faster"))

    def test_clear_loss_needs_no_interleaved_samples(self):
        self.assertEqual(self.verdict(_fast([10.0] * 5, [9.0] * 5)), (False, "slower"))

    def test_close_or_overlapping_fast_timings_escalate(self):
        for ours, theirs in (
            ([10.0] * 5, [13.0] * 5),
            ([10, 10, 10, 10, 20], [15] * 5),
        ):
            with self.subTest(ours=ours, theirs=theirs):
                self.assertIsNone(self.verdict(_fast(ours, theirs)))

    def test_interleaved_samples_decide(self):
        fast = _fast([10.0] * 5, [13.0] * 5)
        cases = (
            ([10.0, 10.2, 10.4] * 10, [13.0, 13.2, 13.4] * 10, (True, "faster")),
            ([10.0, 10.2, 10.4] * 10, [11.0, 11.2, 11.4] * 10, (False, "slower")),
            ([10.0, 10.0, 14.0] * 10, [11.6, 11.6, 16.0] * 10, (False, "inconclusive")),
        )
        for ours, theirs, expected in cases:
            with self.subTest(expected=expected):
                timings = {**fast, "interleaved": {"ours": ours, "inductor": theirs}}
                self.assertEqual(self.verdict(timings), expected)

    def test_margin_boundary_keeps_on_equality(self):
        timings = {
            **_fast([10.0] * 5, [11.5] * 5),
            "interleaved": {"ours": [10.0] * 10, "inductor": [11.5] * 10},
        }
        self.assertEqual(self.verdict(timings), (True, "faster"))
        self.assertEqual(self.verdict(timings, margin=0.16), (False, "slower"))


class WorkerWatchdogTest(unittest.TestCase):
    """A measurement worker is killed only when it stops making progress."""

    def wait(self, script):
        import subprocess
        import sys

        from executorch.backends.cuda.triton import measured_routing

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            measured_routing, "_WORKER_IDLE_TIMEOUT_S", 1.0
        ), mock.patch.object(measured_routing, "_WORKER_POLL_S", 0.1):
            proc = subprocess.Popen([sys.executable, "-c", script, directory])
            return measured_routing._wait_for_worker(proc, Path(directory))

    def test_steady_progress_runs_past_the_idle_limit(self):
        # A result every 0.3 s for 3 s: three times the idle limit in all.
        script = (
            "import sys, time, pathlib\n"
            "for i in range(10):\n"
            "    time.sleep(0.3)\n"
            "    (pathlib.Path(sys.argv[1]) / f'{i}.json').write_text('{}')\n"
        )
        self.assertEqual(self.wait(script), 0)

    def test_a_stalled_worker_is_killed(self):
        script = "import time\ntime.sleep(60)\n"
        self.assertIsNone(self.wait(script))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class MeasureTest(unittest.TestCase):
    """What the worker reports for one call, with the compiles given."""

    def measure(self, inductor=None, replacement=None):
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.backends.cuda.triton import measured_routing

        spec = {"replacement": {"name": "toy"}}
        with mock.patch.object(
            measured_routing,
            "_compile_replacement",
            **(replacement or {"return_value": lambda: "toy"}),
        ), mock.patch.object(
            measured_routing,
            "_compile_inductor",
            **(inductor or {"return_value": (lambda: "inductor", [])}),
        ), mock.patch.object(
            measured_routing,
            "time_with_cuda_graph_us",
            side_effect=lambda fn: {"toy": 10.0, "inductor": 20.0}[fn()],
        ):
            return measured_routing.measure(
                CudaBackend, {}, spec, torch.device("cuda"), 0.15
            )

    def test_both_available_are_timed(self):
        entry = self.measure()
        self.assertEqual(entry["candidates"]["inductor"], {"status": "available"})
        self.assertEqual(entry["timings"]["fast"]["ours"], [10.0] * 5)
        self.assertTrue(entry["cacheable"])

    def test_kernels_the_runtime_lacks_are_deterministic(self):
        entry = self.measure(
            inductor={"return_value": (lambda: None, ["aoti_torch_cuda_convolution"])}
        )
        status = entry["candidates"]["inductor"]
        self.assertEqual(status["status"], "unavailable")
        self.assertIn("aoti_torch_cuda_convolution", status["reason"])
        self.assertTrue(status["cacheable"])
        self.assertIsNone(entry["timings"])
        self.assertTrue(entry["cacheable"])

    def test_no_valid_choices_is_deterministic(self):
        from torch._inductor.select_algorithm import NoValidChoicesError

        wrapped = RuntimeError("backend compiler failed")
        wrapped.__cause__ = NoValidChoicesError("no choices")
        entry = self.measure(inductor={"side_effect": wrapped})
        self.assertTrue(entry["candidates"]["inductor"]["cacheable"])
        self.assertTrue(entry["cacheable"])

    def test_other_failures_are_transient(self):
        for side, name in (("inductor", "inductor"), ("replacement", "toy")):
            with self.subTest(side=side):
                failure = {"side_effect": torch.OutOfMemoryError("out of memory")}
                entry = self.measure(**{side: failure})
                status = entry["candidates"][name]
                self.assertEqual(status["status"], "unavailable")
                self.assertEqual(
                    (status["error"], status["phase"], status["cacheable"]),
                    ("OutOfMemoryError", "compile", False),
                )
                self.assertFalse(entry["cacheable"])


class _PassTest(unittest.TestCase):
    def setUp(self):
        if importlib.util.find_spec("triton") is None or torch.version.hip is not None:
            self.skipTest("Triton on CUDA required")
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def specs(self, **options):
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.exir.backend.compile_spec_schema import CompileSpec

        options.setdefault("measured_routing_cache_dir", str(self.directory))
        return [CudaBackend.generate_method_name_compile_spec("forward")] + [
            CompileSpec(key, value.encode()) for key, value in options.items()
        ]

    def run_passes(self, model=None, **options):
        """The edge graph of ``model`` after the backend's custom passes."""
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.exir import EdgeCompileConfig, to_edge

        model = model or _Add()
        program = torch.export.export(model, model.inputs(), strict=True)
        edge = to_edge(
            program, compile_config=EdgeCompileConfig(_check_ir_validity=False)
        )
        graph_module = edge.exported_program().graph_module
        for p in CudaBackend.get_custom_passes(self.specs(**options)):
            if not getattr(p, "requires_exported_program", False):
                p(graph_module)
        graph_module.graph.lint()
        return graph_module

    def log(self):
        (path,) = (self.directory / "logs").iterdir()
        log = json.loads(path.read_text())
        path.unlink()
        return log

    def export(self, api, model, run=True):
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from executorch.exir import (
            EdgeCompileConfig,
            to_edge,
            to_edge_transform_and_lower,
        )
        from executorch.export.utils import is_fbcode

        inputs = model.inputs()
        program = torch.export.export(model, inputs, strict=True)
        partitioner = CudaPartitioner(self.specs())
        config = EdgeCompileConfig(_check_ir_validity=False)
        if api == "to_backend":
            edge = to_edge(program, compile_config=config).to_backend(partitioner)
        else:
            edge = to_edge_transform_and_lower(
                program, partitioner=[partitioner], compile_config=config
            )
        program = edge.to_executorch()
        if not run or is_fbcode():
            # Running the program is skipped in fbcode.
            return
        from executorch.runtime import Runtime

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pte")
            with open(path, "wb") as f:
                program.write_to_file(f)
            program.write_tensor_data_to_file(directory)
            data = [os.path.join(directory, f) for f in os.listdir(directory)]
            data = [f for f in data if f.endswith(".ptd")]
            method = (
                Runtime.get()
                .load_program(path, data_path=data[0] if data else None)
                .load_method("forward")
            )
            outputs = method.execute([t.cpu() for t in inputs])
        expected = model(*inputs)
        expected = expected if isinstance(expected, tuple) else (expected,)
        for output, reference in zip(outputs, expected, strict=True):
            torch.testing.assert_close(output, reference.cpu(), atol=2e-2, rtol=2e-2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ChoiceTest(_PassTest):
    """The toy family through the backend's passes, with given worker
    results."""

    def setUp(self):
        super().setUp()
        from executorch.backends.cuda.triton import measured_routing, replacement_pass
        from executorch.exir.dialects._ops import ops as exir_ops

        self.add = exir_ops.edge.aten.add.Tensor
        self.legal = {"inductor", "toy"}
        self.static_choice = "toy"
        self.measure = True
        self.results = lambda spec: _entry()
        self.worker = mock.Mock(side_effect=self.run_worker)
        for patch in (
            mock.patch.dict(replacement_pass.EDGE_TO_TRITON_KERNELS, {self.add: _TOY}),
            mock.patch.dict(replacement_pass.PLANNERS, {self.add: self.plan}),
            mock.patch.object(measured_routing, "_run_worker", self.worker),
            mock.patch.object(measured_routing, "_capable_devices", set()),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def run_worker(self, backend, compile_specs, config, device, signatures):
        return None, {
            key: self.results(item["spec"]) for key, item in signatures.items()
        }

    def plan(self, node):
        from executorch.backends.cuda.triton import routing

        return routing.ReplacementPlan(
            "toy",
            (
                routing.Candidate(
                    routing.REGULAR_LOWERING,
                    None if "inductor" in self.legal else "known unavailable",
                ),
                routing.Candidate(
                    "toy",
                    None if "toy" in self.legal else "unsupported",
                    "toy",
                    _TOY,
                    node.args,
                ),
            ),
            self.static_choice,
            "outside the static range",
            "add(x, y)",
            self.measure,
        )

    def chose_toy(self, graph_module):
        return any(n.target is _TOY for n in graph_module.graph.nodes)

    def assertNoMeasurementMeta(self, graph_module):
        from executorch.backends.cuda.triton.measured_routing import META_KEY

        for node in graph_module.graph.nodes:
            self.assertNotIn(META_KEY, node.meta)

    def test_a_clearly_faster_replacement_is_chosen(self):
        graph_module = self.run_passes()
        self.assertTrue(self.chose_toy(graph_module))
        self.assertFalse(any(n.target is self.add for n in graph_module.graph.nodes))
        self.assertNoMeasurementMeta(graph_module)
        log = self.log()
        self.assertEqual(log["config"]["margin"], 0.15)
        node = log["nodes"][0]
        self.assertEqual((node["choice"], node["outcome"]), ("toy", "faster"))
        statuses = {c["name"]: c for c in node["candidates"]}
        self.assertEqual(statuses["inductor"]["status"], "available")
        self.assertEqual(statuses["toy"]["fast_us"]["median_us"], 10.0)

    def test_a_slower_replacement_keeps_the_node(self):
        self.results = lambda spec: _entry(theirs=[8.0] * 5)
        graph_module = self.run_passes()
        self.assertFalse(self.chose_toy(graph_module))
        self.assertNoMeasurementMeta(graph_module)
        self.assertEqual(self.log()["nodes"][0]["outcome"], "slower")

    def test_close_timings_follow_the_interleaved_samples(self):
        for theirs, outcome in (
            ([13.0, 13.2], "faster"),
            ([9.9, 14.0], "inconclusive"),
        ):
            with self.subTest(outcome=outcome):
                interleaved = {"ours": [10.0, 10.1] * 10, "inductor": theirs * 10}
                self.results = lambda spec, interleaved=interleaved: _entry(
                    theirs=[13.0] * 5, interleaved=interleaved
                )
                graph_module = self.run_passes(
                    measured_routing_cache_dir=tempfile.mkdtemp()
                )
                self.assertEqual(self.chose_toy(graph_module), outcome == "faster")

    def test_cached_timings_are_decided_again_at_the_current_margin(self):
        interleaved = {"ours": [10.0, 10.1] * 10, "inductor": [13.0, 13.1] * 10}
        self.results = lambda spec: _entry(theirs=[13.0] * 5, interleaved=interleaved)
        self.assertTrue(self.chose_toy(self.run_passes()))
        self.assertFalse(self.log()["nodes"][0]["cache_hit"])
        graph_module = self.run_passes(measured_routing_margin="0.4")
        self.assertFalse(self.chose_toy(graph_module))
        node = self.log()["nodes"][0]
        self.assertTrue(node["cache_hit"])
        self.assertEqual((node["margin"], node["outcome"]), (0.4, "slower"))
        self.assertEqual(self.worker.call_count, 1)

    def test_deterministic_unavailability_is_cached(self):
        self.results = lambda spec: _entry(inductor=_unavailable(True))
        for cache_hit in (False, True):
            self.assertTrue(self.chose_toy(self.run_passes()))
            node = self.log()["nodes"][0]
            self.assertEqual(
                (node["outcome"], node["cache_hit"]), ("only_available", cache_hit)
            )
        self.assertEqual(self.worker.call_count, 1)

    def test_transient_unavailability_is_measured_again(self):
        self.results = lambda spec: _entry(
            inductor=_unavailable(False, "out of memory")
        )
        for _ in range(2):
            self.assertTrue(self.chose_toy(self.run_passes()))
            statuses = {c["name"]: c for c in self.log()["nodes"][0]["candidates"]}
            self.assertFalse(statuses["inductor"]["cacheable"])
        self.assertEqual(self.worker.call_count, 2)
        self.assertFalse((self.directory / "cache").exists())

    def test_nothing_available_fails_the_export(self):
        from executorch.backends.cuda.triton.routing import ReplacementError

        self.results = lambda spec: _entry(
            inductor=_unavailable(False, "out of memory"),
            toy=_unavailable(False, "kernel failed"),
        )
        with self.assertRaisesRegex(
            ReplacementError, "nothing can run it.*out of memory.*kernel failed"
        ):
            self.run_passes()

    def test_unreported_and_unmeasurable_calls_take_the_static_choice(self):
        from executorch.backends.cuda.triton import measured_routing

        class Twice(_Add):
            def forward(self, x, y):
                return torch.relu(x + x)

        self.worker.side_effect = lambda *args: (None, {})
        for model, message in ((_Add(), "did not report"), (Twice(), "more than once")):
            for static_choice in ("toy", "inductor"):
                with self.subTest(message=message, static_choice=static_choice):
                    self.static_choice = static_choice
                    with self.assertLogs(measured_routing.logger, "ERROR") as logs:
                        graph_module = self.run_passes(model)
                    self.assertIn(message, "\n".join(logs.output))
                    self.assertEqual(
                        self.chose_toy(graph_module), static_choice == "toy"
                    )
                    self.assertEqual(self.log()["nodes"][0]["outcome"], "unmeasurable")
        self.assertFalse((self.directory / "cache").exists())

    def test_a_process_that_cannot_load_cpp_wrapper_modules_measures_nothing(self):
        """Whether the worker can measure is found out before any cached entry
        is used: with nothing, some or everything cached, every call takes its
        static choice, with one warning, and nothing is cached."""
        from executorch.backends.cuda.triton import measured_routing

        class TwoAdds(_Add):
            def forward(self, x, y):
                return x + y, torch.relu(x) * y + x

        self.static_choice = "inductor"
        self.run_passes(_Add())
        self.log()
        cached = sorted((self.directory / "cache").iterdir())
        reason = "ImportError('undefined symbol: aoti_torch_create_cuda_guard')"
        self.worker.side_effect = lambda *args: (reason, {})
        for model, hits in ((TwoAdds(), "some"), (_Add(), "all")):
            with self.subTest(cached=hits):
                measured_routing._capable_devices.clear()
                with self.assertLogs(measured_routing.logger, "WARNING") as logs:
                    graph_module = self.run_passes(model)
                warnings = [
                    r.getMessage() for r in logs.records if r.levelname != "INFO"
                ]
                self.assertEqual(len(warnings), 1, warnings)
                self.assertIn(
                    "cannot load modules compiled with Inductor's C++", warnings[0]
                )
                self.assertIn("aoti_torch_create_cuda_guard", warnings[0])
                self.assertFalse(self.chose_toy(graph_module))
                nodes = self.log()["nodes"]
                self.assertEqual({n["outcome"] for n in nodes}, {"unmeasurable"})
                self.assertEqual(sorted((self.directory / "cache").iterdir()), cached)

    def test_without_measurement_the_static_choice_is_taken(self):
        """With measured routing off, or (under the default, on) for a family
        that has not opted in."""
        for options, measure in (({"measured_routing": "OFF"}, True), ({}, False)):
            for static_choice in ("toy", "inductor"):
                with self.subTest(options=options, static_choice=static_choice):
                    self.static_choice, self.measure = static_choice, measure
                    graph_module = self.run_passes(**options)
                    self.assertEqual(
                        self.chose_toy(graph_module), static_choice == "toy"
                    )
                    self.assertNoMeasurementMeta(graph_module)
        self.assertEqual(self.worker.call_count, 0)
        self.assertFalse((self.directory / "logs").exists())

    def test_a_single_implementation_is_not_measured(self):
        for legal in ({"toy"}, {"inductor"}):
            with self.subTest(legal=legal):
                self.legal = legal
                (self.static_choice,) = legal
                self.assertEqual(self.chose_toy(self.run_passes()), legal == {"toy"})
        self.assertEqual(self.worker.call_count, 0)

    def test_chained_calls_with_keyword_arguments(self):
        graph_module = self.run_passes(_Shared())
        add, plus_one = [n for n in graph_module.graph.nodes if n.target is _TOY]
        self.assertIs(plus_one.args[0], add)
        self.assertEqual(len(add.users), 2)
        (relu,) = [n for n in graph_module.graph.nodes if "relu" in str(n.target)]
        self.assertEqual(len(relu.users), 2)
        self.assertEqual(self.worker.call_count, 1)

    def test_unreadable_cache_entries_are_measured_again(self):
        self.run_passes()
        self.log()
        (entry,) = (self.directory / "cache").iterdir()
        content = entry.read_text()
        for broken in (
            "",
            content[: len(content) // 2],
            content.replace('"schema_version": 1', '"schema_version": 0'),
            json.dumps({"schema_version": 1, "candidates": {"toy": {}}}),
            json.dumps({**_entry(), "candidates": {"toy": {"status": "available"}}}),
            json.dumps({**_entry(), "timings": None}),
            json.dumps(_entry(ours=[10.0, float("nan")] * 3)),
            json.dumps(_entry(ours=[10.0, float("inf")] * 3)),
            json.dumps(_entry(inductor=_unavailable(False))),
        ):
            with self.subTest(broken=broken[:40]):
                entry.write_text(broken)
                calls = self.worker.call_count
                self.run_passes()
                self.assertFalse(self.log()["nodes"][0]["cache_hit"])
                self.assertEqual(self.worker.call_count, calls + 1)

    def test_concurrent_writers_leave_one_complete_entry(self):
        from executorch.backends.cuda.triton import measured_routing, routing

        plan = routing.ReplacementPlan(
            "toy",
            (
                routing.Candidate(routing.REGULAR_LOWERING),
                routing.Candidate("toy", None, "toy", _TOY, (1,)),
            ),
            "toy",
        )
        path = self.directory / "cache" / "key.json"
        first, second = _entry(), _entry(theirs=[30.0] * 5)
        measured_routing._publish(path, first)
        self.assertEqual(measured_routing._read_entry(path, plan), first)
        measured_routing._publish(path, second)
        self.assertEqual(measured_routing._read_entry(path, plan), second)
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_equivalent_calls_share_one_measurement(self):
        class TwoAdds(_Add):
            def forward(self, x, y):
                return x + y, torch.relu(x) + y

        self.run_passes(TwoAdds())
        (call,) = self.worker.call_args_list
        self.assertEqual(len(call.args[4]), 1)
        self.assertEqual(len(self.log()["nodes"]), 2)

    def test_cache_key_covers_what_the_timings_depend_on(self):
        from executorch.backends.cuda.triton import measured_routing

        spec = {
            "original": "aten::add.Tensor",
            "args": [{"tensor": ["float32", [4], [1]]}],
        }
        device = torch.device("cuda")
        key = measured_routing.cache_key(spec, {}, device)
        self.assertEqual(measured_routing.cache_key(dict(spec), {}, device), key)
        strided = {**spec, "args": [{"tensor": ["float32", [4], [2]]}]}
        self.assertNotEqual(measured_routing.cache_key(strided, {}, device), key)
        self.assertNotEqual(
            measured_routing.cache_key({**spec, "alpha": 2}, {}, device), key
        )
        self.assertNotEqual(
            measured_routing.cache_key(spec, {"max_autotune": True}, device), key
        )
        precision = torch._C._get_fp32_precision_getter("cuda", "matmul")
        torch._C._set_fp32_precision_setter(
            "cuda", "matmul", "tf32" if precision != "tf32" else "ieee"
        )
        try:
            self.assertNotEqual(measured_routing.cache_key(spec, {}, device), key)
        finally:
            torch._C._set_fp32_precision_setter("cuda", "matmul", precision)
        properties = torch.cuda.get_device_properties(device)
        other_gpu = types.SimpleNamespace(
            **{
                attr: getattr(properties, attr)
                for attr in ("name", "major", "minor", "multi_processor_count")
            },
            total_memory=properties.total_memory + 1,
        )
        with mock.patch.object(
            torch.cuda, "get_device_properties", return_value=other_gpu
        ):
            self.assertNotEqual(measured_routing.cache_key(spec, {}, device), key)
        with mock.patch.object(measured_routing, "_sources_digest", "other"):
            self.assertNotEqual(measured_routing.cache_key(spec, {}, device), key)

    def test_source_hash_covers_the_measurement_kernels_and_autotune(self):
        from executorch.backends.cuda.triton import measured_routing

        root = Path(measured_routing.__file__).resolve().parents[1]
        files = {
            p.relative_to(root).as_posix() for p in measured_routing.source_files()
        }
        self.assertIn("triton/measured_routing.py", files)
        self.assertIn("autotune/launch_params.py", files)
        self.assertTrue(any(f.startswith("triton/kernels/") for f in files))
        self.assertFalse(any("tests/" in f for f in files))

    def test_invalid_settings_are_rejected(self):
        from executorch.backends.cuda.cuda_backend import CudaBackend

        for key, value in (
            ("measured_routing_margin", "-0.1"),
            ("measured_routing_margin", "1.5"),
            ("measured_routing_margin", "nan"),
            ("measured_routing_margin", "inf"),
            ("measured_routing_margin", "fast"),
            ("measured_routing", "MAYBE"),
        ):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(
                ValueError, key
            ):
                CudaBackend.get_custom_passes(self.specs(**{key: value}))

    def test_export_through_both_lowering_apis(self):
        for api, theirs, chosen in (
            ("to_edge_transform_and_lower", [20.0] * 5, "toy"),
            ("to_backend", [8.0] * 5, "inductor"),
        ):
            with self.subTest(api=api):
                self.directory = Path(tempfile.mkdtemp())
                self.results = lambda spec, theirs=theirs: _entry(theirs=theirs)
                self.export(api, _Add())
                self.assertEqual(self.log()["choices"], {chosen: 1})


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ProbeTest(_PassTest):
    """Real calls measured in the worker process: the conv family's plans with
    the regular lowering always a candidate, so the probe classifies it, and a
    stable sort whose "replacement" is the aten op."""

    def setUp(self):
        from executorch.export.utils import is_fbcode

        if is_fbcode():
            self.skipTest(
                "this build does not export torch's C shims, so the worker "
                "cannot load modules compiled with Inductor's C++ wrapper"
            )
        super().setUp()
        from executorch.backends.cuda.triton import replacement_pass, routing
        from executorch.backends.cuda.triton.kernels.conv import routing as conv_routing
        from executorch.exir.dialects._ops import ops as exir_ops

        def conv_plan(node):
            plan = conv_routing.plan(node)
            regular = routing.Candidate(routing.REGULAR_LOWERING)
            return dataclasses.replace(
                plan, candidates=(regular, *plan.candidates[1:]), measure=True
            )

        def sort_plan(node):
            return routing.ReplacementPlan(
                "sort",
                (
                    routing.Candidate(routing.REGULAR_LOWERING),
                    routing.Candidate(
                        "sort", None, "aten", torch.ops.aten.sort.default, node.args
                    ),
                ),
                measure=True,
            )

        sort = exir_ops.edge.aten.sort.stable
        for patch in (
            mock.patch.dict(
                replacement_pass.EDGE_TO_TRITON_KERNELS,
                {sort: torch.ops.aten.sort.stable},
            ),
            mock.patch.dict(
                replacement_pass.PLANNERS,
                {
                    **{t: conv_plan for t in replacement_pass.CONV_TARGETS},
                    sort: sort_plan,
                },
            ),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def test_real_measurements(self):
        """One export (and run, outside fbcode) through the AOTI compile: the
        3x3 convolution's regular lowering is timed; the probe finds that the
        same-padded 1-D one calls a kernel the runtime lacks, so our kernel
        is kept and the AOTI compile succeeds with it; the sort's supported
        fallback runs and is timed in the worker. Measuring the sort again
        afterwards in this process, where the program's runtime made its shims
        global symbols, still loads and times its regular lowering: the worker
        process is unaffected."""
        self.export("to_edge_transform_and_lower", _Convs())
        nodes = {n["spec"]["original"]: n for n in self.log()["nodes"]}
        self.assertEqual(
            sorted(nodes),
            ["aten::conv1d.padding", "aten::conv2d.default", "aten::sort.stable"],
        )
        regular = {
            name: {c["name"]: c for c in node["candidates"]}["inductor"]
            for name, node in nodes.items()
        }
        # The 3x3 convolution's templates and the supported fallback ran and
        # were timed in the worker.
        for name in ("aten::conv2d.default", "aten::sort.stable"):
            self.assertEqual(regular[name]["status"], "available", regular[name])
            self.assertGreater(regular[name]["fast_us"]["median_us"], 0)
        padded = regular["aten::conv1d.padding"]
        self.assertEqual(padded["status"], "unavailable")
        self.assertIn("aoti_torch_cuda_convolution", padded["reason"])
        self.assertTrue(padded["cacheable"])
        self.assertEqual(
            (
                nodes["aten::conv1d.padding"]["choice"],
                nodes["aten::conv1d.padding"]["outcome"],
            ),
            ("conv", "only_available"),
        )
        conv = {c["name"]: c for c in nodes["aten::conv2d.default"]["candidates"]}[
            "conv"
        ]
        self.assertTrue(conv["picks"])
        for path in (self.directory / "cache").iterdir():
            path.unlink()
        self.run_passes(_Sort())
        (node,) = self.log()["nodes"]
        self.assertFalse(node["cache_hit"])
        regular = {c["name"]: c for c in node["candidates"]}["inductor"]
        self.assertGreater(regular["fast_us"]["median_us"], 0)
