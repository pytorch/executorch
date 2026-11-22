# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ast
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "private_kv_rows_checker", Path(__file__).with_name("check_private_kv_rows.py")
)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class PrivateKVRowsPerformanceGateTest(unittest.TestCase):
    def test_native_tokens_distinguish_owners_positions_and_reuse(self):
        for profile in ("mini", "mg-ring"):
            requests = checker.commands(profile)
            steps = {
                request["tag"]: request
                for request in requests
                if request["op"] == "step"
            }
            with self.subTest(profile=profile):
                self.assertEqual(len(set(steps["decode64-first"]["tokens"])), 64)
                self.assertGreater(len(set(steps["long"]["tokens"])), 80)
                self.assertNotEqual(
                    steps["bench-c1-0"]["tokens"], steps["bench-c1-1"]["tokens"]
                )
                self.assertNotEqual(
                    steps["reuse-long"]["tokens"][0],
                    steps["reuse-before"]["tokens"][0],
                )
                self.assertTrue(steps["reuse-after"]["save_logits"])
                for request in steps.values():
                    self.assertEqual(len(request["tokens"]), len(request["seq_ids"]))
                    self.assertTrue(all(0 < token < 256 for token in request["tokens"]))
            self.assertEqual(requests, checker.commands(profile))

    def test_numerical_gate_rejects_matching_nonfinite_logits(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(RuntimeError, "ABI 2.*non-finite"):
                    checker.require_exact_logits([value], [value], "matching")
                with self.assertRaisesRegex(RuntimeError, "ABI 3.*non-finite"):
                    checker.require_exact_logits([value], [1.0], "candidate")
        checker.require_exact_logits([1.0, -2.0], [1.0, -2.0], "finite")
        with self.assertRaises(AssertionError):
            checker.require_exact_logits([1.0], [2.0], "mismatch")

    def test_cpu_phase_report_keeps_total_throughput_gate(self):
        steps, pairs = {}, []
        for c in (1, 2, 4, 8, 16, 32, 64):
            for i in range(20):
                steps[f"bench-c{c}-{i}"] = {
                    "elapsed_seconds": 0.002,
                    "cache_phase_seconds": {
                        "cache_prepare_time_ns": 0.0001,
                        "cache_commit_time_ns": 0.0003,
                        "archive_transfer_time_ns": 0.0002,
                        "archive_append_time_ns": 0.00001,
                    },
                }
            for phase in ("growth", "restore"):
                pairs.append(({"tag": f"{phase}-c{c}-0"}, {"elapsed_seconds": 0.001}))
        old_steps = {
            tag: {**response, "elapsed_seconds": 0.001}
            for tag, response in steps.items()
        }
        timings, lifecycle = checker.paired_timings([(old_steps, steps, pairs, pairs)])
        self.assertEqual(timings[1]["throughput_ratio"], 0.5)
        self.assertEqual(
            timings[1]["rounds"][0]["cache_cpu_wall_s"]["abi3"]["cache_commit_time_ns"],
            0.0003,
        )
        with self.assertRaisesRegex(RuntimeError, "d128/c1"):
            checker.require_performance(
                {128: {"same_work_timings": timings, "lifecycle_timings": lifecycle}}
            )

    def test_boundary_and_improvements_qualify(self):
        checker.require_performance(
            {
                128: {
                    "same_work_timings": {
                        1: {"throughput_ratio": 0.98},
                        64: {"throughput_ratio": 1.2},
                    }
                }
            }
        )

    def test_measured_regression_blocks_even_without_flag(self):
        with self.assertRaisesRegex(RuntimeError, "d256/c4"):
            checker.require_performance(
                {256: {"same_work_timings": {4: {"throughput_ratio": 0.979}}}}
            )

    def test_one_bad_concurrency_blocks_the_complete_gate(self):
        with self.assertRaisesRegex(RuntimeError, "d128/c64"):
            checker.require_performance(
                {
                    128: {
                        "same_work_timings": {
                            1: {"throughput_ratio": 1.01},
                            64: {"throughput_ratio": 0.7},
                        }
                    },
                    256: {"same_work_timings": {64: {"throughput_ratio": 1.1}}},
                }
            )

    def test_lifecycle_regression_is_not_hidden_by_fast_decode(self):
        with self.assertRaisesRegex(RuntimeError, "d128/growth-c64"):
            checker.require_performance(
                {
                    128: {
                        "same_work_timings": {64: {"throughput_ratio": 1.3}},
                        "lifecycle_timings": {"growth-c64": {"throughput_ratio": 0.5}},
                    }
                }
            )

    def test_native_workload_keeps_lifecycle_operations_in_bounds(self):
        for profile in ("mini", "mg-ring"):
            with self.subTest(profile=profile):
                self.check_workload(profile)

    def check_workload(self, profile):
        geometry = checker.qualification_config(profile, 128)
        lengths = {}
        written = {}
        counts = {"clone": 0, "rewind": 0, "growth": 0}
        for request in checker.commands(profile):
            op = request["op"]
            if op == "clear":
                lengths.clear()
                written.clear()
            elif op == "new":
                owner = next(i for i in range(64) if i not in lengths)
                lengths[owner] = 0
                written[owner] = 0
            elif op == "remove":
                del lengths[request["seq_id"]]
                del written[request["seq_id"]]
            elif op == "clone":
                owner = next(i for i in range(64) if i not in lengths)
                source = request["seq_id"]
                self.assertLessEqual(request["upto"], lengths[source])
                self.assertGreaterEqual(
                    request["upto"],
                    max(0, written[source] - geometry["max_step_tokens"]),
                )
                lengths[owner] = request["upto"]
                written[owner] = written[source]
                counts["clone"] += request.get("tag", "").startswith("restore-")
            elif op == "rewind":
                owner = request["seq_id"]
                old = lengths[owner]
                self.assertLessEqual(request["position"], old)
                self.assertGreaterEqual(
                    request["position"],
                    max(0, written[owner] - geometry["max_step_tokens"]),
                )
                lengths[owner] = request["position"]
                counts["rewind"] += request.get("tag", "").startswith("restore-")
            elif op == "step":
                self.assertLessEqual(
                    len(request["seq_ids"]), geometry["max_step_tokens"]
                )
                for owner in request["seq_ids"]:
                    lengths[owner] += 1
                    written[owner] = max(written[owner], lengths[owner])
                    self.assertLessEqual(lengths[owner], geometry["max_context"])
                counts["growth"] += request.get("tag", "").startswith("growth-c")
            self.assertLessEqual(len(lengths), 64)
        self.assertEqual(counts["clone"], 7 * 8)
        self.assertEqual(counts["rewind"], 7 * 8)
        self.assertEqual(counts["growth"], 2 * (1 + 2 + 4 + 8 + 16 + 32 + 64))

    def test_production_profile_matches_mg_attention_geometry(self):
        source = Path(__file__).parents[1] / "model/model.py"
        tree = ast.parse(source.read_text())
        config = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MuseGlimmerConfig"
        )
        defaults = {
            node.target.id: ast.literal_eval(node.value)
            for node in config.body
            if isinstance(node, ast.AnnAssign) and node.value is not None
        }
        geometry = checker.qualification_config("mg-ring", 128)
        for field in (
            "n_layers",
            "n_heads",
            "n_kv_heads",
            "head_dim",
            "global_attn_cfg",
        ):
            self.assertEqual(geometry["model"][field], defaults[field])
        pattern = ast.literal_eval(defaults["global_attn_cfg"])
        self.assertEqual(
            geometry["ring_layers"],
            sum(pattern[i % len(pattern)] > 0 for i in range(defaults["n_layers"])),
        )
        self.assertEqual(geometry["window"], 2048)
        self.assertEqual(geometry["max_step_tokens"], 4096)
        self.assertEqual(geometry["max_context"], 131072)
        self.assertLess(geometry["model"]["dim"], defaults["dim"])
        with self.assertRaises(ValueError):
            checker.qualification_config("mg-ring", 256)

    def test_production_ring_replays_evicted_history_at_rollback_floor(self):
        requests = checker.commands("mg-ring")
        seeds = [
            request
            for request in requests
            if request.get("tag", "").startswith("production-ring-seed-")
        ]
        self.assertEqual([len(r["tokens"]) for r in seeds], [4096, 4096])
        self.assertNotEqual(seeds[0]["tokens"], seeds[1]["tokens"])
        last_seed = requests.index(seeds[-1])
        self.assertEqual(
            requests[last_seed + 1 : last_seed + 3],
            [
                {"op": "clone", "seq_id": 0, "upto": 4096},
                {"op": "rewind", "seq_id": 0, "position": 4096},
            ],
        )
        replay = requests[last_seed + 3]
        self.assertEqual(replay["tag"], "production-ring-floor-replay")
        self.assertTrue(replay["save_logits"])
        self.assertEqual(replay["seq_ids"], [0, 1])

    def test_production_host_budget_counts_all_39_ring_layers(self):
        geometry = checker.qualification_config("mg-ring", 128)
        expected = 39 * (64 * 96 * (2 * 64 * 512 + 64 * 4) + 2 * 4096 * 512)
        self.assertEqual(checker.host_undo_budget(geometry), expected)
        self.assertGreater(expected, 14 * 1024**3)
        self.assertLess(expected, 15 * 1024**3)

    def test_driver_contraction_cannot_pass_on_live_bytes_alone(self):
        before = {
            "live_KV_bytes": 32 * 1024**2,
            "allocator_reserved_bytes": 64 * 1024**2,
            "device_used_bytes": 128 * 1024**2,
        }
        after = {
            "live_KV_bytes": 1024**2,
            "allocator_reserved_bytes": before["allocator_reserved_bytes"],
            "device_used_bytes": before["device_used_bytes"],
        }
        with self.assertRaisesRegex(RuntimeError, "return at least 8 MiB"):
            checker.require_driver_contraction(before, after)
        after["allocator_reserved_bytes"] -= 16 * 1024**2
        with self.assertRaisesRegex(RuntimeError, "return at least 8 MiB"):
            checker.require_driver_contraction(before, after)
        after["device_used_bytes"] -= 16 * 1024**2
        self.assertEqual(
            checker.require_driver_contraction(before, after)["driver_used_drop_bytes"],
            16 * 1024**2,
        )
        after["live_KV_bytes"] = 31 * 1024**2
        with self.assertRaisesRegex(RuntimeError, "less than 16 MiB"):
            checker.require_driver_contraction(before, after)

    def test_timings_count_lifecycle_work_and_exclude_verification(self):
        old_steps, new_steps, old_pairs, new_pairs = {}, {}, [], []
        for c in (1, 2, 4, 8, 16, 32, 64):
            for i in range(20):
                old_steps[f"bench-c{c}-{i}"] = {"elapsed_seconds": 10}
                new_steps[f"bench-c{c}-{i}"] = {"elapsed_seconds": 5}
            for phase, old, new in (("growth", 3, 2), ("restore", 7, 4)):
                for op in ("clone", "step"):
                    request = {"tag": f"{phase}-c{c}-0-{op}"}
                    old_pairs.append((request, {"elapsed_seconds": old}))
                    new_pairs.append((request, {"elapsed_seconds": new}))
            request = {"tag": f"growth-verify-c{c}"}
            old_pairs.append((request, {"elapsed_seconds": 1000}))
            new_pairs.append((request, {"elapsed_seconds": 1}))
        repeats = [(old_steps, new_steps, old_pairs, new_pairs)] * 4
        decode, lifecycle = checker.paired_timings(repeats)
        for c in decode:
            self.assertEqual(decode[c]["throughput_ratio"], 2)
            self.assertEqual(lifecycle[f"growth-c{c}"]["throughput_ratio"], 1.5)
            self.assertEqual(lifecycle[f"restore-c{c}"]["throughput_ratio"], 1.75)


if __name__ == "__main__":
    unittest.main()
