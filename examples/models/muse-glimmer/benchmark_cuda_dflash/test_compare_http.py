# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import base64
import copy
import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import compare_http as client

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "requests.json"


def event(obj):
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def choice(delta=None, finish=None):
    return {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}


def stream(n=5):
    return [
        event(choice({"role": "assistant"})),
        event(choice({"reasoning_content": "think"})),
        event(choice({"content": "answer"})),
        event(choice(finish="stop")),
        event({"choices": [], "usage": {"completion_tokens": n}}),
        b"data: [DONE]\n\n",
    ]


def result():
    return {
        "raw_chunks": [],
        "events": [],
        "reasoning_content": "",
        "content": "",
        "first_visible_delta_ns": None,
        "first_reasoning_ns": None,
        "first_content_ns": None,
        "finish_ns": None,
        "finish_reason": None,
        "usage": None,
        "start_ns": 0,
    }


class ParserTests(unittest.TestCase):
    def test_fragmented_multiline_and_line_endings(self):
        for ending in (b"\n", b"\r\n", b"\r"):
            raw = ending.join(
                [
                    b":comment",
                    b"event: message",
                    b'data: {"choices":',
                    b'data: [], "usage": {"completion_tokens": 7}}',
                    b"",
                    b"",
                ]
            )
            parser = client.SSE()
            events = []
            for byte in raw:
                events.extend(parser.feed(bytes([byte])))
            self.assertEqual(len(events), 1)
            self.assertEqual(json.loads(events[0])["usage"]["completion_tokens"], 7)

    def test_reasoning_content_usage_and_exact_raw(self):
        r = result()
        chunks = stream()
        client.consume([(i * 10**9, chunk) for i, chunk in enumerate(chunks, 1)], r)
        self.assertEqual((r["reasoning_content"], r["content"]), ("think", "answer"))
        self.assertEqual(r["first_visible_delta_ns"], 2 * 10**9)
        self.assertEqual(r["first_reasoning_ns"], 2 * 10**9)
        self.assertEqual(r["first_content_ns"], 3 * 10**9)
        self.assertEqual(r["finish_reason"], "stop")
        self.assertEqual(r["usage"]["completion_tokens"], 5)
        self.assertEqual(
            b"".join(base64.b64decode(c["base64"]) for c in r["raw_chunks"]),
            b"".join(chunks),
        )
        self.assertEqual(r["events"][-1]["received_ns"], 6 * 10**9)
        m = client.metrics(r)
        self.assertEqual(m["client_ttft_s_not_gpu_first_token"], 2)
        self.assertEqual(m["client_avg_tpot_s"], 1)
        self.assertEqual(m["total_finish_s"], 6)

    def test_metric_denominator_zero_one_and_no_visible_text(self):
        for n in (0, 1, 2, 9):
            r = result()
            client.consume(enumerate(stream(n), 1), r)
            expected = None if n <= 1 else 4 / 1e9 / (n - 1)
            self.assertEqual(client.metrics(r)["client_avg_tpot_s"], expected)
        r["first_visible_delta_ns"] = None
        self.assertIsNone(client.metrics(r)["client_avg_tpot_s"])
        self.assertIsNone(client.metrics(r)["client_ttft_s_not_gpu_first_token"])

    def test_missing_usage_errors_and_malformed_streams(self):
        cases = [
            stream()[:-2] + [stream()[-1]],
            stream()[:-1],
            [event({"error": {"message": "server failed"}})],
            [b"data: not-json\n\n"],
            stream() + [event(choice())],
            stream()[:-1] + [b"data: [DONE]\n"],
            [b"data: \xff\n\n"],
            [event({"choices": {}})],
            [event({"choices": [], "usage": {"completion_tokens": True}})],
        ]
        for chunks in cases:
            with self.subTest(chunks=chunks):
                r = result()
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    client.consume(enumerate(chunks, 1), r)
                self.assertTrue(r["raw_chunks"])


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.scratch.cleanup)
        self.args = SimpleNamespace(
            base_url="http://unused.invalid",
            timeout=2,
            mode="cuda-batched-no-dflash",
            include_usage=False,
            concurrency=1,
            output_dir=Path(self.scratch.name),
        )
        self.tasks, self.sha = client.load_fixtures(FIXTURE)

    def test_frozen_hashes_and_bad_fourth_hash(self):
        self.assertEqual(len(self.tasks), 4)
        self.assertEqual(self.sha, hashlib.sha256(FIXTURE.read_bytes()).hexdigest())
        changed = {"schema_version": 1, "tasks": copy.deepcopy(self.tasks)}
        changed["tasks"][-1]["source_sha256"] = "0" * 64
        path = self.args.output_dir / "fixture.json"
        for task in self.tasks:
            source = FIXTURE.parent / task["source_file"]
            (path.parent / task["source_file"]).write_bytes(source.read_bytes())
        client.atomic_json(path, changed)
        with self.assertRaisesRegex(ValueError, "Source hash mismatch"):
            client.load_fixtures(path)

    def test_mock_transport_exact_post_body_and_instrumentation(self):
        for include in (False, True):
            self.args.include_usage = include
            original = copy.deepcopy(self.tasks[0])
            with patch.object(client.http.client, "HTTPConnection") as transport:
                conn = transport.return_value
                response = conn.getresponse.return_value
                response.status = 200
                response.getheaders.return_value = [
                    ("Content-Type", "text/event-stream")
                ]
                response.getheader.return_value = "text/event-stream"
                response.read1.side_effect = stream() + [b""]
                r = client.request_one(self.tasks[0], self.sha, self.args)
                self.assertTrue(r["success"], r["error"])
                call = conn.request.call_args
                self.assertEqual(call.args, ("POST", "/v1/chat/completions"))
                wire = json.loads(call.kwargs["body"])
                expected = copy.deepcopy(original["payload"])
                if include:
                    expected["stream_options"] = {"include_usage": True}
                self.assertEqual(wire, expected)
                self.assertEqual(r["wire_body_utf8"].encode(), call.kwargs["body"])
                self.assertEqual(self.tasks[0], original)
                conn.close.assert_called_once()

    def test_transport_failures_are_recorded_without_metrics(self):
        for status, chunks in (
            (500, [b"failure", b""]),
            (200, stream()[:-2] + [stream()[-1], b""]),
        ):
            with patch.object(client.http.client, "HTTPConnection") as transport:
                response = transport.return_value.getresponse.return_value
                response.status = status
                response.getheaders.return_value = []
                response.getheader.return_value = "text/event-stream"
                response.read1.side_effect = chunks
                r = client.request_one(self.tasks[0], self.sha, self.args)
                self.assertFalse(r["success"])
                self.assertIsNone(r["metrics"])
                self.assertTrue(r["error"]["message"])
                self.assertTrue(r["raw_chunks"])

    def test_sequential_and_four_barrier_launch_once(self):
        for concurrency in (1, 4):
            self.args.concurrency = concurrency
            active = 0
            peak = 0
            seen = []
            lock = threading.Lock()
            rendezvous = threading.Barrier(4) if concurrency == 4 else None

            def worker(task, sha, args, barrier=None):
                nonlocal active, peak
                if barrier is not None:
                    barrier.wait(timeout=2)
                with lock:
                    active += 1
                    peak = max(peak, active)
                    seen.append(task["id"])
                if rendezvous:
                    rendezvous.wait(timeout=2)
                with lock:
                    active -= 1
                return task["id"]

            with patch.object(client, "request_one", side_effect=worker):
                returned = client.run_group(self.tasks, self.sha, self.args)
            self.assertEqual(returned, [t["id"] for t in self.tasks])
            self.assertCountEqual(seen, returned)
            self.assertEqual(peak, concurrency)

    def test_makespan_all_four_and_no_failed_aggregate(self):
        for concurrent in (False, True):
            rows = [
                {
                    "start_ns": (0 if concurrent else i) * 10**9,
                    "finish_ns": (1 if concurrent else i + 1) * 10**9,
                    "success": True,
                    "usage": {"completion_tokens": 10},
                }
                for i in range(4)
            ]
            m = client.group_metrics(rows)
            self.assertEqual(m["makespan_s"], 1 if concurrent else 4)
            self.assertEqual(
                m["aggregate_completion_tokens_per_s"], 40 if concurrent else 10
            )
            rows[-1]["success"] = False
            self.assertIsNone(
                client.group_metrics(rows)["aggregate_completion_tokens_per_s"]
            )

    def test_atomic_json_roundtrip(self):
        path = self.args.output_dir / "result.json"
        client.atomic_json(path, {"text": 'quotes"\n', "metric": None})
        self.assertEqual(
            json.loads(path.read_bytes()), {"text": 'quotes"\n', "metric": None}
        )
        self.assertEqual(list(self.args.output_dir.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
