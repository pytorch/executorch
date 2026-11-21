# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "private_kv_rows_checker", Path(__file__).with_name("check_private_kv_rows.py")
)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class PrivateKVRowsPerformanceGateTest(unittest.TestCase):
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
        lengths = {}
        counts = {"clone": 0, "rewind": 0, "growth": 0}
        for request in checker.commands():
            op = request["op"]
            if op == "clear":
                lengths.clear()
            elif op == "new":
                owner = next(i for i in range(64) if i not in lengths)
                lengths[owner] = 0
            elif op == "remove":
                del lengths[request["seq_id"]]
            elif op == "clone":
                owner = next(i for i in range(64) if i not in lengths)
                source = lengths[request["seq_id"]]
                self.assertLessEqual(request["upto"], source)
                self.assertGreaterEqual(request["upto"], max(0, source - 128))
                lengths[owner] = request["upto"]
                counts["clone"] += request.get("tag", "").startswith("restore-")
            elif op == "rewind":
                old = lengths[request["seq_id"]]
                self.assertLessEqual(request["position"], old)
                self.assertGreaterEqual(request["position"], max(0, old - 128))
                lengths[request["seq_id"]] = request["position"]
                counts["rewind"] += request.get("tag", "").startswith("restore-")
            elif op == "step":
                self.assertLessEqual(len(request["seq_ids"]), 128)
                for owner in request["seq_ids"]:
                    lengths[owner] += 1
                    self.assertLessEqual(lengths[owner], 4096)
                counts["growth"] += request.get("tag", "").startswith("growth-c")
            self.assertLessEqual(len(lengths), 64)
        self.assertEqual(counts["clone"], 7 * 8)
        self.assertEqual(counts["rewind"], 7 * 8)
        self.assertEqual(counts["growth"], 2 * (1 + 2 + 4 + 8 + 16 + 32 + 64))

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
