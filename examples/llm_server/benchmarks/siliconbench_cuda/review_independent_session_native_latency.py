"""Reconstruct synchronized native executor latency from preserved mini runs.

The probe ends elapsed_seconds before snapshot(), protocol serialization and
IPC. Its interval includes host step preparation, sampling and CUDA completion;
it is not GPU kernel time or HTTP throughput. No candidate is promoted here.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_round(root, number, expected):
    path = root / f"round-{number}" / "protocol.jsonl"
    assert digest(path) == expected
    operations, pending = [], None
    for line in path.read_text().splitlines():
        entry = json.loads(line)
        if "request" in entry:
            assert pending is None
            pending = entry["request"]
        elif entry["response"].get("ready"):
            assert pending is None
        else:
            assert pending is not None
            response = entry["response"]
            assert response["op"] == pending["op"] and not response.get("failed")
            assert math.isfinite(response["elapsed_seconds"])
            assert response["elapsed_seconds"] >= 0
            operations.append(dict(request=pending, response=response))
            pending = None
    assert pending is None
    return operations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-job", type=Path, required=True)
    parser.add_argument("--independent-job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    documents, protocols = {}, {}
    for arm, root in [("global", args.global_job), ("independent", args.independent_job)]:
        document = json.loads((root / "result.json").read_text())
        assert document["arm"] == arm and document["role"] == "mini"
        assert len(document["rounds"]) == 4
        documents[arm] = document
        protocols[arm] = [load_round(root, number, row["protocol_sha256"])
                          for number, row in enumerate(document["rounds"], 1)]
    assert documents["global"]["probe_sha256"] == documents["independent"]["probe_sha256"]
    for number in range(4):
        clean = {}
        for arm in documents:
            clean[arm] = [{key: value for key, value in op["request"].items() if key != "tag"}
                          for op in protocols[arm][number]
                          if op["request"]["op"] not in ("compact", "metrics")]
        assert clean["global"] == clean["independent"]

    stages = []
    for index, concurrency in enumerate([1, 2, 4, 8, 16, 1]):
        arms = {}
        for arm, document in documents.items():
            rows = []
            for number, round_record in enumerate(document["rounds"]):
                stage = round_record["stages"][index]
                assert stage["concurrency"] == concurrency
                ops = protocols[arm][number]
                boundaries = [i for i, op in enumerate(ops)
                              if op["request"]["op"] == "metrics"
                              and op["response"]["snapshot"] == stage["before"]]
                assert len(boundaries) == 1
                steps = ops[boundaries[0] + 1:boundaries[0] + 65]
                assert len(steps) == 64
                assert all(op["request"]["op"] == "step"
                           and op["request"]["seq_ids"] == stage["active_owners"]
                           and len(op["request"]["tokens"]) == concurrency for op in steps)
                if number > 0:
                    assert not any(op["request"].get("save_logits") for op in steps)
                elapsed = [op["response"]["elapsed_seconds"] for op in steps]
                previous = stage["before"]
                changed = []
                for i, op in enumerate(steps):
                    current = op["response"]["snapshot"]
                    if any(current[key] != previous[key] for key in
                           ("row_growth_count", "column_resize_count", "live_KV_bytes")):
                        changed.append(i)
                    previous = current
                rows.append(dict(round=number + 1,
                                 mean_native_step_ms=1000 * statistics.mean(elapsed),
                                 first_three_native_ms=1000 * sum(elapsed[:3]),
                                 after_first_three_native_step_ms=1000 * statistics.mean(elapsed[3:]),
                                 after_first_three_native_TPS=61 * concurrency / sum(elapsed[3:]),
                                 storage_change_call_indices=changed))
            arms[arm] = dict(rounds=rows, statistics={
                key: dict(mean=statistics.mean(row[key] for row in rows[1:]),
                          sample_stddev=statistics.stdev(row[key] for row in rows[1:]),
                          first_discarded=rows[0][key])
                for key in ("mean_native_step_ms", "first_three_native_ms",
                            "after_first_three_native_step_ms", "after_first_three_native_TPS")})
        key = "after_first_three_native_step_ms"
        baseline, candidate = [arms[arm]["statistics"][key] for arm in ("global", "independent")]
        standard_error = math.sqrt((baseline["sample_stddev"] ** 2 +
                                    candidate["sample_stddev"] ** 2) / 3)
        delta = candidate["mean"] - baseline["mean"]
        stages.append(dict(stage=index, concurrency=concurrency, arms=arms,
                           latency_ratio=candidate["mean"] / baseline["mean"],
                           slowdown_ms=delta, three_standard_errors_ms=3 * standard_error,
                           slowdown_exceeds_three_standard_errors=delta > 3 * standard_error,
                           statistical_signal_is_not_acceptance=True))

    result = dict(native_interval_excludes_IPC_and_post_interval_snapshots=True,
                  includes_host_preparation_argmax_and_CUDA_synchronization=True,
                  GPU_kernel_timing=False, HTTP_throughput=False,
                  four_rounds_first_discarded=True, stages=stages,
                  model_performance_qualified=False, candidate_promoted=False,
                  remaining_gates=["prefill and mixed model latency", "HTTP workload equivalence",
                                   "HTTP performance and memory", "automatic reclaim implementation"],
                  source_results_sha256={arm: digest(root / "result.json") for arm, root in
                                         [("global", args.global_job), ("independent", args.independent_job)]})
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stages": len(stages), "candidate_promoted": False,
                      "latency_ratios": [stage["latency_ratio"] for stage in stages]}))


if __name__ == "__main__":
    main()
