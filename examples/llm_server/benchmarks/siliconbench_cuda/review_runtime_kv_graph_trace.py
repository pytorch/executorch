"""Verify actual CUDA graph APIs inside the diagnostic executor's NVTX ranges.

This requires the raw first-round protocol, Nsight SQLite trace, and actual
result hashes. It never infers graph replay from a configuration flag.
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sqlite3


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-job", type=Path, required=True)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    result = json.loads((args.runtime_job / "result.json").read_text())
    assert result["arm"] == "runtime" and len(result["rounds"]) == 4
    protocol_path = args.runtime_job / "round-1/protocol.jsonl"
    assert digest(protocol_path) == result["rounds"][0]["protocol_sha256"]
    operations, pending = [], None
    for line in protocol_path.read_text().splitlines():
        entry = json.loads(line)
        if "request" in entry:
            assert pending is None
            pending = entry["request"]
        elif entry["response"].get("ready"):
            assert pending is None
        else:
            assert pending is not None
            operations.append(dict(request=pending, response=entry["response"]))
            pending = None
    assert pending is None
    db = sqlite3.connect("file:" + str(args.sqlite.resolve()) + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    tables = {
        r[0] for r in db.execute('select name from sqlite_master where type="table"')
    }
    assert {"NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_RUNTIME", "StringIds"} <= tables
    strings = {r[0]: r[1] for r in db.execute("select id,value from StringIds")}
    ranges = {}
    for row in db.execute("select * from NVTX_EVENTS"):
        row = dict(row)
        label = row.get("text") or strings.get(row.get("textId"))
        if not label or not label.startswith("command-"):
            continue
        assert row["end"] is not None and row["start"] < row["end"]
        assert (
            label not in ranges
        ), "Each serialized request must have one real executor range"
        ranges[label] = row
    assert set(ranges) == {op["request"]["tag"] for op in operations}
    graph_events = [
        dict(row)
        for row in db.execute(
            "select r.start,r.end,r.globalTid,r.returnValue,s.value as api "
            "from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id "
            'where s.value like "%Graph%"'
        )
    ]
    counts = {}
    for label, interval in ranges.items():
        events = [
            e
            for e in graph_events
            if e["globalTid"] == interval["globalTid"]
            and interval["start"] <= e["start"]
            and e["end"] <= interval["end"]
        ]
        relevant = [
            e
            for e in events
            if any(
                word in e["api"]
                for word in [
                    "GraphInstantiate",
                    "GraphLaunch",
                    "GraphDestroy",
                    "GraphExecDestroy",
                ]
            )
        ]
        assert all(e["returnValue"] == 0 for e in relevant), relevant
        counts[label] = collections.Counter(e["api"] for e in relevant)

    def total(ops, fragment):
        return sum(
            n
            for op in ops
            for name, n in counts[op["request"]["tag"]].items()
            if fragment in name
        )

    stage_evidence = []
    for stage in result["rounds"][0]["stages"]:
        indices = [
            i
            for i, op in enumerate(operations)
            if op["request"]["op"] == "metrics"
            and op["response"]["snapshot"] == stage["before"]
        ]
        assert (
            len(indices) == 1
        ), "Recorded stage must map to one actual protocol boundary"
        start = indices[0]
        steps = operations[start + 1 : start + 65]
        assert len(steps) == 64 and all(op["request"]["op"] == "step" for op in steps)
        assert all(op["request"]["seq_ids"] == stage["active_owners"] for op in steps)
        instantiated, launches = total(steps, "GraphInstantiate"), total(
            steps, "GraphLaunch"
        )
        assert instantiated > 0 and launches > 0, (
            stage["stage"],
            instantiated,
            launches,
        )
        stage_evidence.append(
            dict(
                stage=stage["stage"],
                concurrency=stage["concurrency"],
                physical_columns=stage["after"]["physical_columns"],
                actual_instantiations=instantiated,
                actual_launches=launches,
            )
        )
    assert [s["physical_columns"] for s in stage_evidence] == [2, 3, 5, 9, 17, 2]
    shrink_ops = [
        op
        for op in operations
        if op["request"]["op"] == "compact"
        and op["response"]["snapshot"]["physical_columns"] == 2
    ]
    assert (
        shrink_ops
        and total(shrink_ops, "GraphExecDestroy") + total(shrink_ops, "GraphDestroy")
        > 0
    )
    final_steps = [op for op in operations if op["request"]["op"] == "step"][-21:]
    assert len(final_steps) == 21
    method_evidence = []
    for index, (tokens, width) in enumerate(
        [(1, 1), (2, 2), (3, 4), (7, 8), (13, 16), (32, 32), (63, 64)]
    ):
        steps = final_steps[index * 3 : index * 3 + 3]
        assert all(len(op["request"]["tokens"]) == tokens for op in steps)
        launches = total(steps, "GraphLaunch")
        instantiated = total(steps, "GraphInstantiate")
        assert launches > 0 and (width == 1 or instantiated > 0)
        method_evidence.append(
            dict(
                actual_tokens=tokens,
                static_width=width,
                actual_instantiations=instantiated,
                actual_launches=launches,
            )
        )
    output = dict(
        actual_CUDA_trace=True,
        recapture_after_growth_and_shrink_verified=True,
        replay_after_growth_and_shrink_verified=True,
        all_seven_static_methods_after_shrink_verified=True,
        probe_sha256=result["probe_sha256"],
        model_PTE_sha256=result["model_PTE_sha256"],
        protocol_sha256=digest(protocol_path),
        sqlite_sha256=digest(args.sqlite),
        first_discarded_round_only=True,
        stage_evidence=stage_evidence,
        post_shrink_method_evidence=method_evidence,
        acceptance_scope="Actual graph instantiate/launch/destroy API events inside owned executor NVTX command ranges; numerical and memory acceptance are separate",
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(dict(actual_graph_replay_verified=True, stages=6, static_methods=7))
    )


if __name__ == "__main__":
    main()
