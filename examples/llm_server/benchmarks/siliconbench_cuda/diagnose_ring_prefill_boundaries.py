"""Compare paired mini artifacts before and across a sliding-window boundary."""

import argparse
from array import array
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import traceback


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(config, output):
    cfg = json.loads(config.read_text())
    assert not output.exists(), "Use a new directory and preserve previous results"
    output.mkdir()
    (output / "configuration.json").write_text(json.dumps(cfg, indent=2))
    bindings = {}
    for name in ("probe", "baseline_PTE", "PTE", "PTD"):
        bindings[name] = {"path": cfg[name], "SHA256": digest(cfg[name])}
        assert bindings[name]["SHA256"] == cfg["expected_SHA256"][name], name
    (output / "bindings.json").write_text(json.dumps(bindings, indent=2))
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = cfg["native_lib"] + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env["ET_CUDA_RING_JOURNAL_BUFFERS"] = "0"
    env["ET_CUDA_RING_ARCHIVE_BATCH_ROWS"] = "64"
    cases = [
        ("fresh-1", [], 1),
        ("fresh-64", [], 64),
        ("fresh-512", [], 512),
        ("fresh-1024", [], 1024),
        ("fresh-2048", [], 2048),
        ("history576-64", [64, 512], 64),
        ("history576-512", [64, 512], 512),
        ("history576-1024", [64, 512], 1024),
        ("history576-2048", [64, 512], 2048),
        ("history1024-64", [1024], 64),
        ("history1024-2048", [1024], 2048),
    ]
    completed = []
    with Path(cfg["lock"]).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip(), "Existing GPU work must remain untouched"
        for name, seed_sizes, size in cases:
            selected = sorted({0, min(63, size - 1), min(511, size - 1), size - 1})
            for arm in ("baseline", "ring"):
                (output / "progress.json").write_text(json.dumps({
                    "status": "diagnosing-mini", "case": name, "arm": arm,
                    "completed": completed, "mini_only": True,
                }))
                directory = output / name / arm
                directory.mkdir(parents=True)
                pte = cfg["baseline_PTE"] if arm == "baseline" else cfg["PTE"]
                command = [
                    cfg["probe"], "--pte=" + pte, "--data_path=" + cfg["PTD"],
                    "--max_sessions=81", "--max_context=131072",
                    "--initial_kv_capacity=512", "--output_dir=" + str(directory),
                ]
                with (directory / "stderr.log").open("x") as log:
                    proc = subprocess.Popen(command, stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=log,
                                            env=env, text=True, bufsize=1)
                    def receive(request=None):
                        line = proc.stdout.readline()
                        assert line, (name, arm, "probe ended", proc.poll())
                        response = json.loads(line)
                        with (directory / "trace.jsonl").open("a") as stream:
                            stream.write(json.dumps({"request": request, "response": response}) + "\n")
                        assert not response.get("failed"), response
                        return response
                    def send(request):
                        proc.stdin.write(json.dumps(request) + "\n")
                        proc.stdin.flush()
                        return receive(request)
                    try:
                        assert receive()["ready"]
                        sid = send({"op": "new"})["seq_id"]
                        for index, seed_size in enumerate(seed_sizes):
                            send({"op": "step", "tokens": [7 + i % 1000 for i in range(seed_size)],
                                  "seq_ids": [sid] * seed_size, "select": [seed_size - 1],
                                  "tag": "seed-" + str(index)})
                        send({"op": "step", "tokens": [7 + i % 1000 for i in range(size)],
                              "seq_ids": [sid] * size, "select": selected,
                              "save_logits": True, "tag": "prefill"})
                        send({"op": "step", "tokens": [111], "seq_ids": [sid],
                              "select": [0], "save_logits": True, "tag": "decode"})
                        proc.stdin.close()
                        assert proc.wait(timeout=120) == 0
                    finally:
                        if proc.poll() is None:
                            proc.terminate()
                            proc.wait(timeout=120)
            comparisons = []
            for tag, positions in (("prefill", selected), ("decode", [0])):
                for index, position in enumerate(positions):
                    filename = f"{tag}-{index}.f32"
                    left_path = output / name / "baseline" / filename
                    right_path = output / name / "ring" / filename
                    left, right = array("f"), array("f")
                    left.frombytes(left_path.read_bytes())
                    right.frombytes(right_path.read_bytes())
                    assert len(left) == len(right) and left
                    assert all(math.isfinite(value) for value in left)
                    assert all(math.isfinite(value) for value in right)
                    difference = sum((a - b) ** 2 for a, b in zip(left, right))
                    scale = sum(a * a for a in left)
                    nrms = math.sqrt(difference / max(scale, 1e-30))
                    absolute = max(abs(a - b) for a, b in zip(left, right))
                    comparisons.append({
                        "tag": tag, "selected_position": position, "values": len(left),
                        "normalized_RMS": nrms, "max_abs": absolute,
                        "byte_exact": left_path.read_bytes() == right_path.read_bytes(),
                        "baseline_argmax": max(range(len(left)), key=left.__getitem__),
                        "ring_argmax": max(range(len(right)), key=right.__getitem__),
                        "original_threshold_pass": nrms <= 0.005 and absolute <= 0.5,
                        "baseline_SHA256": digest(left_path), "ring_SHA256": digest(right_path),
                    })
            completed.append({"case": name, "seed_steps": seed_sizes,
                              "history": sum(seed_sizes), "step_tokens": size,
                              "comparisons": comparisons})
            (output / "interim-results.json").write_text(json.dumps(completed, indent=2))
    result = {"cases": completed, "mini_only": True, "diagnostic_only": True,
              "full_model_qualified": False, "original_qualification_suite_passed": False,
              "all_diagnostic_points_pass": all(
                  row["original_threshold_pass"] for case in completed for row in case["comparisons"]),
              "thresholds": {"normalized_RMS": 0.005, "max_abs": 0.5},
              "UTC": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (output / "result.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rc = 1
    try:
        main(args.config, args.output)
        rc = 0
    except BaseException:
        if args.output.exists():
            (args.output / "failure.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        if args.output.exists():
            (args.output / "exit-receipt.json").write_text(json.dumps({
                "returncode": rc, "mini_only": True, "diagnostic_only": True,
                "UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }))
    raise SystemExit(rc)
