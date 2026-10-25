"""Independently compare matched raw tokens, logits and live-owner memory.

Graph acceptance requires a separate actual CUDA trace. Nothing is promoted.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        while data := file.read(16 * 1024 * 1024):
            h.update(data)
    return h.hexdigest()


def protocol(root, number):
    path = root / f"round-{number}" / "protocol.jsonl"
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    operations = []
    request = None
    for entry in entries:
        if "request" in entry:
            assert request is None
            request = entry["request"]
        else:
            response = entry["response"]
            if response.get("ready"):
                assert request is None
                continue
            assert request is not None and response["op"] == request["op"]
            if request["op"] not in ["compact", "metrics"]:
                clean = {key: value for key, value in request.items() if key != "tag"}
                operations.append(dict(request=clean, response=response))
            request = None
    assert request is None
    return operations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixed", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--source-spec", type=Path, required=True)
    p.add_argument("--graph-review", type=Path)
    p.add_argument("--weight-payload-review", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    import numpy as np

    fixed = json.loads((a.fixed / "result.json").read_text())
    runtime = json.loads((a.runtime / "result.json").read_text())
    assert fixed["arm"] == "fixed" and runtime["arm"] == "runtime"
    assert fixed["role"] == runtime["role"]
    assert len(fixed["rounds"]) == len(runtime["rounds"]) == 4
    assert fixed["probe_sha256"] == runtime["probe_sha256"]
    # PTD can include generated SO payloads as well as tensor storages.
    # A differing package hash requires a separate per-storage weight review.
    weight_review = None
    if fixed["weights_blob_sha256"] != runtime["weights_blob_sha256"]:
        assert a.weight_payload_review is not None
        weight_review = json.loads(a.weight_payload_review.read_text())
        assert weight_review["all_quantized_and_nonquantized_storage_sha256_equal"]
        assert weight_review["fixed_PTD_sha256"] == fixed["weights_blob_sha256"]
        assert weight_review["runtime_PTD_sha256"] == runtime["weights_blob_sha256"]
    comparisons = []
    exact = True
    for number in range(1, 5):
        f = protocol(a.fixed, number)
        r = protocol(a.runtime, number)
        assert len(f) == len(r)
        selected_logits = 0
        max_abs = 0.0
        for left, right in zip(f, r):
            assert left["request"] == right["request"]
            if left["request"]["op"] in ["new", "clone"]:
                assert left["response"]["seq_id"] == right["response"]["seq_id"]
            if left["request"]["op"] != "step":
                continue
            assert left["response"]["argmax"] == right["response"]["argmax"]
            ls = left["response"]["logits"]
            rs = right["response"]["logits"]
            assert len(ls) == len(rs)
            for lpart, rpart in zip(ls, rs):
                assert (
                    lpart["shape"] == rpart["shape"]
                    and lpart["selected_indices"] == rpart["selected_indices"]
                )
                lp = Path(lpart["file"])
                rp = Path(rpart["file"])
                assert lp.resolve().is_relative_to(a.fixed.resolve())
                assert rp.resolve().is_relative_to(a.runtime.resolve())
                lv = np.fromfile(lp, dtype="<f4").reshape(lpart["shape"])
                rv = np.fromfile(rp, dtype="<f4").reshape(rpart["shape"])
                assert np.isfinite(lv).all() and np.isfinite(rv).all()
                difference = float(np.max(np.abs(lv - rv)))
                max_abs = max(max_abs, difference)
                exact = exact and np.array_equal(lv, rv)
                assert np.allclose(lv, rv, rtol=1e-5, atol=1e-4), difference
                selected_logits += lv.shape[0]
        assert (selected_logits > 0) == (number == 1)
        comparisons.append(
            dict(
                round=number,
                matched_operations=len(f),
                saved_logit_rows=selected_logits,
                max_abs_logit_error=max_abs,
                argmax_identical=True,
                fixed_protocol_sha256=digest(
                    a.fixed / f"round-{number}" / "protocol.jsonl"
                ),
                runtime_protocol_sha256=digest(
                    a.runtime / f"round-{number}" / "protocol.jsonl"
                ),
            )
        )
    memory = []
    for round_number in range(1, 5):
        r = runtime["rounds"][round_number - 1]["stages"]
        assert [x["concurrency"] for x in r] == [1, 2, 4, 8, 16, 1]
        assert [x["after"]["physical_columns"] for x in r] == [2, 3, 5, 9, 17, 2]
        assert len({x["after"]["pid"] for x in r}) == 1
        assert r[-1]["active_owners"] == [r[-2]["active_owners"][5]]
        assert r[-1]["after"]["live_KV_bytes"] < r[-2]["after"]["live_KV_bytes"]
        assert r[-1]["after"]["device_used_bytes"] < r[-2]["after"]["device_used_bytes"]
        memory.append(
            dict(
                round=round_number,
                live_KV_by_stage=[x["after"]["live_KV_bytes"] for x in r],
                device_used_by_stage=[x["after"]["device_used_bytes"] for x in r],
                shrink_live_KV_bytes=r[-2]["after"]["live_KV_bytes"]
                - r[-1]["after"]["live_KV_bytes"],
                shrink_device_used_bytes=r[-2]["after"]["device_used_bytes"]
                - r[-1]["after"]["device_used_bytes"],
            )
        )
    graph_pass = False
    graph = None
    if a.graph_review:
        graph = json.loads(a.graph_review.read_text())
        assert (
            graph["actual_CUDA_trace"]
            and graph["recapture_after_growth_and_shrink_verified"]
        )
        assert graph["replay_after_growth_and_shrink_verified"]
        assert graph["probe_sha256"] == runtime["probe_sha256"]
        assert graph["model_PTE_sha256"] == runtime["model_PTE_sha256"]
        graph_pass = True
    result = dict(
        role=runtime["role"],
        source_spec_sha256=digest(a.source_spec),
        numerical_checks_pass=True,
        weight_payload_review=weight_review,
        all_saved_logits_bitwise_equal=exact,
        numerical_rounds=comparisons,
        physical_growth_and_reclamation_pass=True,
        memory_rounds=memory,
        mean_last3_device_shrink_bytes=statistics.mean(
            x["shrink_device_used_bytes"] for x in memory[1:]
        ),
        graph_recapture_replay_pass=graph_pass,
        graph_review=graph,
        graph_evidence_required_before_full_model=not graph_pass,
        candidate_promoted=False,
        whole_optimization_accepted=False,
        primary_SiliconBench_results_replaced=False,
        fixed_result_sha256=digest(a.fixed / "result.json"),
        runtime_result_sha256=digest(a.runtime / "result.json"),
        limitations="Matched teacher-forced actual executor diagnostics. Exact request order and logits are checked, not HTTP task throughput or full GMRID F1. Device snapshots and quarter-second samples may miss short peaks.",
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "numerical_checks_pass",
                    "physical_growth_and_reclamation_pass",
                    "graph_recapture_replay_pass",
                    "candidate_promoted",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
