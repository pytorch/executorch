"""Capture real single-owner mini Gemma Q/K/V through unmodified CUDA launches."""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback


def main(config, output):
    cfg = json.loads(config.read_text())
    output.mkdir(exist_ok=False)
    (output / "configuration.json").write_text(json.dumps(cfg, indent=2))
    for key, expected in cfg["SHA256"].items():
        assert hashlib.sha256(Path(cfg[key]).read_bytes()).hexdigest() == expected, key
    library = output / "trace.so"
    command = ["/usr/bin/c++", "-O2", "-std=c++17", "-fPIC", "-shared",
               cfg["interposer_source"], "-I" + cfg["cuda_include"],
               "-L" + cfg["cuda_lib"], "-Wl,-rpath," + cfg["cuda_lib"],
               "-lcuda", "-lcudart", "-ldl", "-o", str(library)]
    (output / "build-command.json").write_text(json.dumps(command, indent=2))
    with (output / "build.log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    records = []
    with Path(cfg["lock"]).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip()
        for size in [64, 512, 1024, 2048]:
            for arm in ["baseline", "ring"]:
                folder = output / str(size) / arm
                folder.mkdir(parents=True)
                trace = folder / "trace"
                trace.mkdir()
                env = dict(os.environ, LD_PRELOAD=str(library),
                           LD_LIBRARY_PATH=cfg["native_lib"],
                           ET_DIAGNOSTIC_ATTENTION_OUTPUT=str(trace),
                           ET_CUDA_RING_JOURNAL_BUFFERS="0", ET_CUDA_RING_ARCHIVE_BATCH_ROWS="64")
                model = cfg["baseline_PTE"] if arm == "baseline" else cfg["PTE"]
                command = [cfg["probe"], "--pte=" + model, "--data_path=" + cfg["PTD"],
                           "--max_sessions=81", "--max_context=131072", "--initial_kv_capacity=512",
                           "--output_dir=" + str(folder)]
                with (folder / "stderr.log").open("x") as log:
                    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=log, env=env, text=True, bufsize=1)
                    def receive(request=None):
                        line = child.stdout.readline()
                        assert line, (size, arm, "probe ended", child.poll())
                        response = json.loads(line)
                        with (folder / "trace.jsonl").open("a") as stream:
                            stream.write(json.dumps(dict(request=request, response=response)) + "\n")
                        assert not response.get("failed"), response
                        return response
                    def send(request):
                        child.stdin.write(json.dumps(request) + "\n")
                        child.stdin.flush()
                        return receive(request)
                    try:
                        assert receive()["ready"]
                        owner = send(dict(op="new"))["seq_id"]
                        send(dict(op="step", tokens=[7 + i % 1000 for i in range(size)],
                                  seq_ids=[owner] * size, select=[size - 1],
                                  save_logits=True, tag="prefill"))
                        send(dict(op="step", tokens=[111], seq_ids=[owner], select=[0],
                                  save_logits=True, tag="decode"))
                        child.stdin.close()
                        assert child.wait(timeout=120) == 0
                    finally:
                        if child.poll() is None:
                            child.terminate()
                            child.wait(timeout=120)
                metadata = list(trace.glob("*-metadata.json"))
                assert len(metadata) == 5, (size, arm, "expected five real sliding-attention captures", len(metadata))
                for i in range(5):
                    assert (trace / (str(i) + "-attention-out.bf16")).stat().st_size == 8192
                records.append(dict(size=size, arm=arm, directory=str(folder), sliding_layers=5))
                (output / "progress.json").write_text(json.dumps(dict(completed=records, diagnostic_only=True)))
    (output / "result.json").write_text(json.dumps(dict(
        cases=records, graphs_disabled=True, no_kernel_arguments_or_math_changed=True,
        diagnostic_only=True, full_model_qualified=False,
        library_SHA256=hashlib.sha256(library.read_bytes()).hexdigest()), indent=2))


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
            (args.output / "exit-receipt.json").write_text(json.dumps(dict(
                returncode=rc, diagnostic_only=True,
                UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())))
    raise SystemExit(rc)
