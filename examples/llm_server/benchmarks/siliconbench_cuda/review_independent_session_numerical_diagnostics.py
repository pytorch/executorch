"""Retain numerical diagnostics from every matched mini operation, without promotion."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def operations(root, number, expected):
    path = root / f"round-{number}/protocol.jsonl"
    assert digest(path) == expected
    result, pending = [], None
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if "request" in row:
            assert pending is None
            pending = row["request"]
        elif row["response"].get("ready"):
            assert pending is None
        else:
            response = row["response"]
            assert pending is not None and response["op"] == pending["op"]
            assert not response.get("failed")
            if pending["op"] not in ("compact", "metrics"):
                result.append(dict(request={k: v for k, v in pending.items() if k != "tag"},
                                   response=response))
            pending = None
    assert pending is None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("global-job", "independent-job", "weight-review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    import numpy as np

    documents = [json.loads((root / "result.json").read_text())
                 for root in (args.global_job, args.independent_job)]
    assert documents[0]["probe_sha256"] == documents[1]["probe_sha256"]
    assert all(len(d["rounds"]) == 4 for d in documents)
    weights = json.loads(args.weight_review.read_text())
    assert weights["all_quantized_and_nonquantized_storage_sha256_equal"]
    assert weights["fixed_PTD_sha256"] == documents[0]["weights_blob_sha256"]
    assert weights["runtime_PTD_sha256"] == documents[1]["weights_blob_sha256"]
    rounds = []
    for number in range(1, 5):
        paired = [operations(root, number, d["rounds"][number - 1]["protocol_sha256"])
                  for root, d in zip((args.global_job, args.independent_job), documents)]
        assert len(paired[0]) == len(paired[1])
        mismatches, errors, steps = [], [], 0
        for index, (left, right) in enumerate(zip(*paired)):
            assert left["request"] == right["request"]
            if left["request"]["op"] != "step":
                continue
            l, r = left["response"], right["response"]
            if l["argmax"] != r["argmax"]:
                mismatches.append(dict(operation=index, step=steps, global_argmax=l["argmax"],
                                       independent_argmax=r["argmax"],
                                       histories=l["snapshot"]["logical_length"]))
            steps += 1
            assert len(l["logits"]) == len(r["logits"])
            for lp, rp in zip(l["logits"], r["logits"]):
                assert lp["shape"] == rp["shape"]
                assert lp["selected_indices"] == rp["selected_indices"]
                arrays, paths = [], []
                for part, root in zip((lp, rp), (args.global_job, args.independent_job)):
                    path = Path(part["file"])
                    assert path.resolve().is_relative_to(root.resolve())
                    a = np.fromfile(path, dtype="<f4").reshape(part["shape"])
                    assert np.isfinite(a).all()
                    arrays.append(a.astype(np.float64))
                    paths.append(dict(file=str(path), SHA256=digest(path)))
                a, b = arrays
                delta = a - b
                errors.append(dict(operation=index, step=steps - 1, selected=lp["selected_indices"],
                                   rows=a.shape[0], max_abs=float(np.abs(delta).max()),
                                   relative_L2=float(np.linalg.norm(delta) / max(np.linalg.norm(a), 1e-12)),
                                   RMSE=float(np.sqrt(np.mean(delta * delta))),
                                   selected_argmax_mismatches=int(np.count_nonzero(np.argmax(a, -1) != np.argmax(b, -1))),
                                   files=paths))
        rounds.append(dict(round=number, matched_operations=len(paired[0]), step_count=steps,
                           argmax_mismatch_count=len(mismatches), argmax_mismatches=mismatches,
                           selected_logit_diagnostics=errors,
                           max_abs=max((row["max_abs"] for row in errors), default=None),
                           max_relative_L2=max((row["relative_L2"] for row in errors), default=None)))
    result = dict(same_teacher_forced_tokens=True, same_original_weight_storage_bytes=True,
                  candidate_accepted=False, full_model_qualified=False, rounds=rounds,
                  numerical_cause_not_established=True,
                  remaining_required_evidence=["Compare deployed compiled math and autotune choices",
                                              "FP32 or equivalent same-math reference for the numerical difference",
                                              "Saved logits/top-token margins at mismatched argmax steps"],
                  source_results_SHA256=[digest(root / "result.json")
                                         for root in (args.global_job, args.independent_job)],
                  weight_review_SHA256=digest(args.weight_review))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(rounds=[{k: row[k] for k in
                                  ("round", "step_count", "argmax_mismatch_count", "max_abs", "max_relative_L2")}
                                 for row in rounds], candidate_accepted=False)))


if __name__ == "__main__":
    main()
