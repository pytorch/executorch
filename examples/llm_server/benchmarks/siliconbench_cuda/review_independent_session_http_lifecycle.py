"""Recompute mini HTTP workload and memory evidence from preserved raw streams."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def moments(values):
    return dict(mean=statistics.mean(values), sample_SD=statistics.stdev(values), values=values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-native-memory", action="store_true")
    parser.add_argument("--wait-minutes", type=int, default=0)
    args = parser.parse_args()
    assert not args.output.exists()
    until = time.monotonic() + args.wait_minutes * 60
    while not (args.job / "result.json").exists():
        state = json.loads((args.job / "progress.json").read_text())
        assert not (args.job / "failure.json").exists(), "Inspect preserved HTTP failure"
        handle = Path("/proc") / str(state["controller_pid"])
        assert handle.exists()
        assert b"run_independent_session_mini_http.py" in (handle / "cmdline").read_bytes()
        if time.monotonic() >= until:
            print(json.dumps(dict(status="observation-timeout", live_HTTP_pid=state["controller_pid"])))
            return
        time.sleep(20)
    document = json.loads((args.job / "result.json").read_text())
    assert document["actual_official_HTTP_server"] and document["mini_only"]
    assert document["context"] == 131072 and document["initial_KV_capacity"] == 512
    assert len(document["rounds"]) == 16
    native_enabled = document.get("native_allocator_sampling_enabled", False)
    if args.require_native_memory:
        assert native_enabled
    records, comparison = [], {}
    for row in document["rounds"]:
        number, prefix, arm = row["round"], row["prefix"], row["arm"]
        directory = args.job / ("round-" + str(number)) / prefix / arm
        requests_path, memory_path, log_path = [directory / name for name in ("requests.jsonl", "memory.jsonl", "server.log")]
        for path, key in ((requests_path, "requests_sha256"), (memory_path, "memory_samples_sha256"), (log_path, "server_log_sha256")):
            assert digest(path) == row[key]
        raw_requests = [json.loads(line) for line in requests_path.read_text().splitlines()]
        samples = [json.loads(line) for line in memory_path.read_text().splitlines()]
        assert len(raw_requests) == 32 and len(samples) > 100
        assert all(a["monotonic_ns"] <= b["monotonic_ns"] for a, b in zip(samples, samples[1:]))
        native = []
        for line in log_path.read_text(errors="replace").splitlines():
            if "KV_MEM_SAMPLE:" not in line:
                continue
            item, _ = json.JSONDecoder().raw_decode(line.split("KV_MEM_SAMPLE:", 1)[1])
            native.append(item)
        if native_enabled:
            assert len(native) > 10
            assert all(item["cuda_set_device_status"] == 0 for item in native)
            allocated = [item for item in native if item["allocator_pool_available"]]
            assert allocated and all(item["allocator_query_status"] == [0, 0, 0, 0] for item in allocated)
            first_allocation_ns = allocated[0]["monotonic_ns"]
            assert all(item["monotonic_ns"] < first_allocation_ns and item["live_KV_bytes"] == 0
                       for item in native if not item["allocator_pool_available"])
            assert len({item["pid"] for item in native}) == 1
        else:
            assert not native
        stages, cursor = [], 0
        for index, concurrency in enumerate((1, 2, 4, 8, 16, 1)):
            stage = row["stages"][index]
            assert stage["stage"] == index and stage["concurrency"] == concurrency
            expected_owners = [5] if index == 5 else list(range(concurrency))
            assert stage["active_owners"] == expected_owners
            raw = raw_requests[cursor:cursor + concurrency]
            cursor += concurrency
            assert sorted(item["owner"] for item in raw) == expected_owners
            raw_by_owner = {item["owner"]: item for item in raw}
            for item in stage["requests"]:
                original = raw_by_owner[item["owner"]]
                assert {k: v for k, v in original.items() if k not in ("body", "events")} == item
                assert original["body"]["max_tokens"] == 64 and original["body"]["temperature"] == 0
                assert original["body"]["session_id"] == "mini-" + str(item["owner"])
                events = original["events"]
                assert not any("error" in event for event in events)
                usages = [event["usage"] for event in events if event.get("usage")]
                assert usages == [item["usage"]]
                finishes = [choice["finish_reason"] for event in events for choice in event.get("choices", [])
                            if choice.get("finish_reason") is not None]
                assert len(finishes) == 1
                usage = item["usage"]
                assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
                assert item["start_ns"] < item["first_content_ns"] <= item["end_ns"]
                assert item["seconds"] == (item["end_ns"] - item["start_ns"]) / 1e9
                assert item["TTFT_seconds"] == (item["first_content_ns"] - item["start_ns"]) / 1e9
                if prefix == "off":
                    assert usage["prompt_tokens_details"]["cached_tokens"] == 0
                text = "".join(choice.get("delta", {}).get("content", "") for event in events for choice in event.get("choices", []))
                key = (number, prefix, index, item["owner"])
                value = dict(body=original["body"], prompt_tokens=usage["prompt_tokens"],
                             completion_tokens=usage["completion_tokens"], finish_reason=finishes[0],
                             output_text_sha256=hashlib.sha256(text.encode()).hexdigest())
                comparison.setdefault(key, {})[arm] = value
            tokens = sum(item["usage"]["completion_tokens"] for item in raw)
            elapsed = (stage["end_ns"] - stage["start_ns"]) / 1e9
            assert tokens == stage["output_tokens"]
            assert tokens / elapsed == stage["aggregate_TPS"]
            timed = [item for item in samples if stage["start_ns"] <= item["monotonic_ns"] <= stage["end_ns"]]
            assert timed
            lifecycle = {}
            labels = ["stage-" + str(index) + "-c" + str(concurrency) + suffix for suffix in ("-active", "-idle")]
            for label in labels:
                selected = [item for item in samples if item["stage"] == label]
                assert selected
                lower, upper = selected[0]["monotonic_ns"], selected[-1]["monotonic_ns"]
                readings = [item for item in native if lower <= item["monotonic_ns"] <= upper]
                value = dict(NVML_peak_bytes=max(item["GPU_used_bytes"] for item in selected),
                             NVML_last_bytes=selected[-1]["GPU_used_bytes"], NVML_samples=len(selected))
                if native_enabled:
                    value["native_sample_coverage"] = bool(readings)
                    value["native_samples"] = len(readings)
                    if readings:
                        value["native_last"] = readings[-1]
                        value["native_peak_KV_bytes"] = max(item["live_KV_bytes"] for item in readings)
                        pooled = [item for item in readings if item["allocator_pool_available"]]
                        assert pooled and readings[-1]["allocator_pool_available"], label
                        value["allocator_initial_unavailable_samples"] = len(readings) - len(pooled)
                        value["allocator_peak_reserved_bytes"] = max(item["allocator_reserved_bytes"] for item in pooled)
                    else:
                        assert label.endswith("-active"), "Missing required idle lifecycle evidence: " + label
                        value["native_limitation"] = "No250ms native sample within this short active interval; no active native peak is claimed"
                lifecycle[label] = value
            stages.append(dict(stage=index, concurrency=concurrency, seconds=elapsed,
                               output_tokens=tokens, TPS=tokens / elapsed,
                               mean_TTFT_seconds=statistics.mean(item["TTFT_seconds"] for item in raw),
                               lifecycle=lifecycle))
        assert cursor == len(raw_requests)
        after = stages[-1]["lifecycle"]["stage-5-c1-idle"]
        if native_enabled and prefix == "off" and arm == "independent":
            final = after["native_last"]
            assert final["live_sequences"] == 1 and final["physical_columns"] == 2, final
            assert final["max_full_capacity"] <= 2048, final
        records.append(dict(round=number, prefix=prefix, arm=arm, stages=stages,
                            native_samples=len(native),
                            native_initial_pool_unavailable_samples=sum(not item["allocator_pool_available"] for item in native),
                            native_query_duration_ns=moments([item["query_duration_ns"] for item in native]) if len(native) > 1 else None,
                            raw_file_SHA256={path.name: digest(path) for path in (requests_path, memory_path, log_path)}))
    assert len(comparison) == 256 and all(set(value) == {"global", "independent"} for value in comparison.values())
    differences = []
    for key, values in comparison.items():
        left, right = values["global"], values["independent"]
        assert left["body"] == right["body"] and left["prompt_tokens"] == right["prompt_tokens"]
        if left != right:
            differences.append(dict(round=key[0], prefix=key[1], stage=key[2], owner=key[3],
                                    global_output=left, independent_output=right))
    lengths_equal = all(values["global"]["completion_tokens"] == values["independent"]["completion_tokens"]
                        for values in comparison.values())
    summaries = []
    for prefix in ("off", "on"):
        for arm in ("global", "independent"):
            rows = sorted((row for row in records if row["prefix"] == prefix and row["arm"] == arm), key=lambda row: row["round"])
            assert [row["round"] for row in rows] == [1, 2, 3, 4]
            for index in range(6):
                summaries.append(dict(prefix=prefix, arm=arm, stage=index, concurrency=rows[0]["stages"][index]["concurrency"],
                                      TPS=moments([row["stages"][index]["TPS"] for row in rows[1:]]),
                                      mean_TTFT_seconds=moments([row["stages"][index]["mean_TTFT_seconds"] for row in rows[1:]]),
                                      NVML_idle_last_bytes=moments([row["stages"][index]["lifecycle"]["stage-" + str(index) + "-c" + str(row["stages"][index]["concurrency"]) + "-idle"]["NVML_last_bytes"] for row in rows[1:]])))
    result = dict(raw_streams_usage_timestamps_and_hashes_verified=True, matched_requests=len(comparison),
                  completion_lengths_equal_between_arms=lengths_equal,
                  all_outputs_reached_limit=all(value[arm]["completion_tokens"] == 64 for value in comparison.values() for arm in value),
                  output_differences=differences, natural_output_texts_identical=not differences,
                  native_allocator_memory_verified=native_enabled,
                  automatic_OFF_c16_to_c1_reclaim_verified=native_enabled,
                  source_result_SHA256=digest(args.job / "result.json"), rounds=records, summaries=summaries,
                  native_sampler_overhead_not_qualified=True, model_performance_qualified=False, candidate_promoted=False,
                  limitations="Mini HTTP diagnostic. NVML includes all device allocations; the ET allocator pool is not a measure of graph-private pools. Natural output text differences need model correctness review.")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("matched_requests", "completion_lengths_equal_between_arms", "all_outputs_reached_limit", "native_allocator_memory_verified", "automatic_OFF_c16_to_c1_reclaim_verified")}))


if __name__ == "__main__":
    main()
