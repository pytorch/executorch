"""Profile seven MG mixes after the independent four-round API experiment.

Only diagnostic harnesses are rebuilt with NVTX labels. Framework sources,
weights, activation precision, and accepted performance records remain inputs.
The profiler captures forward plus greedy after histories are initialized.
Reported profiler latencies are never accepted as ordinary performance data.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-job", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    args.job.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4")

    def state(status, **fields):
        pending = args.job / "progress.pending.json"
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
                                           **fields), indent=2) + "\n")
        pending.replace(args.job / "progress.json")

    try:
        state("waiting-official-API-build")
        deadline = time.monotonic() + 21600
        provenance_path = args.api_job / "build-and-input-provenance.json"
        while not provenance_path.is_file():
            assert time.monotonic() < deadline, "API driver build remains incomplete"
            time.sleep(30)
        original = json.loads(provenance_path.read_text())
        assert digest(args.gguf) == original["GGUF_sha256"]
        for path, expected in original["preserved_SDK_and_source_hashes"].items():
            assert digest(path) == expected, f"Original SDK/source input changed: {path}"
        state("building-diagnostic-harnesses")
        commands = []
        for original_command in original["compile_and_link_commands"]:
            command = list(original_command)
            for index, argument in enumerate(command):
                if argument.endswith(("/roofline_et.cpp", "/roofline_cpp.cpp")):
                    command[index] = str(source / Path(argument).name)
                elif argument in [str(args.api_job / name) for name in
                                   ["roofline_et.o", "roofline_et", "roofline_cpp"]]:
                    command[index] = str(args.job / Path(argument).name)
            if any(argument.endswith(".cpp") for argument in command):
                command[1:1] = ["-DROOFLINE_PROFILE=1", "-I/usr/local/cuda/include"]
            if "-o" in command and command[command.index("-o") + 1].endswith(
                    ("roofline_et", "roofline_cpp")):
                command.append("-ldl")
            commands.append(command)
            with (args.job / "build.log").open("a") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=1200)
        inputs = dict(
            compile_commands=commands,
            diagnostic_source_SHA256={name: digest(source / name) for name in
                                      ["roofline_common.h", "roofline_et.cpp",
                                       "roofline_cpp.cpp", Path(__file__).name]},
            original_API_provenance_SHA256=digest(provenance_path),
            original_performance_results_unchanged=True,
            profiles_are_diagnostics_not_performance_rounds=True,
            framework_sources_not_modified=True,
            CUDA_capture_range="cudaProfilerApi",
            NVTX_forward_ranges="mix=<name>;phase=warmup|timed;iteration=<index>",
            initialization_and_history_construction_outside_capture=True)
        (args.job / "source-provenance.json").write_text(json.dumps(inputs, indent=2) + "\n")
        state("waiting-four-round-API-completion")
        while True:
            progress = json.loads((args.api_job / "progress.json").read_text())
            assert progress["status"] != "failed", "API comparison failed; preserve it for review"
            if progress["status"] == "completed":
                break
            assert time.monotonic() < deadline, "API comparison remains incomplete"
            time.sleep(30)
        final_inputs = json.loads(provenance_path.read_text())
        assert {key: value for key, value in final_inputs.items()
                if key != "artifact_sha256"} == {
                    key: value for key, value in original.items()
                    if key != "artifact_sha256"}, "Original API build provenance changed"
        final_provenance_SHA256 = digest(provenance_path)
        inputs["completed_API_provenance_SHA256"] = final_provenance_SHA256
        inputs["artifact_sha256"] = final_inputs["artifact_sha256"]
        (args.job / "source-provenance.json").write_text(json.dumps(inputs, indent=2) + "\n")
        for name, expected in final_inputs["artifact_sha256"].items():
            assert digest(args.artifact / name) == expected
        with args.lock.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert not subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                text=True).strip(), "Existing GPU owner must remain untouched"
            for framework in ["ET", "CPP"]:
                state("profiling", framework=framework,
                      current_profiling_latencies_not_main_performance=True)
                binary = args.job / ("roofline_et" if framework == "ET" else "roofline_cpp")
                report = args.job / framework
                result = args.job / (framework + "-profile-driver.json")
                child = ([str(binary), str(args.artifact / "model.pte"),
                          str(args.artifact / "aoti_cuda_blob.ptd"), str(result), "3"]
                         if framework == "ET" else
                         [str(binary), str(args.gguf), str(result), "3"])
                command = ["nsys", "profile", "--trace=cuda,nvtx", "--sample=none",
                           "--cpuctxsw=none", "--capture-range=cudaProfilerApi",
                           "--capture-range-end=stop", "--cuda-graph-trace=node",
                           "--force-overwrite=false", "--output=" + str(report), *child]
                with (args.job / (framework + ".log")).open("x") as log:
                    subprocess.run(command, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, check=True, timeout=3600)
                assert json.loads(result.read_text())["completed"]
                subprocess.run(["nsys", "export", "--type=sqlite",
                                "--output=" + str(report.with_suffix(".sqlite")),
                                str(report.with_suffix(".nsys-rep"))],
                               check=True, timeout=900, stdout=subprocess.DEVNULL)
                wait_deadline = time.monotonic() + 60
                while subprocess.check_output(
                        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                        text=True).strip():
                    assert time.monotonic() < wait_deadline, "Owned CUDA context not torn down"
                    time.sleep(1)
        assert digest(provenance_path) == final_provenance_SHA256, \
            "Completed API provenance changed during profiling"
        state("completed", profiles=2, actual_per_mix_kernel_analysis_pending=True,
              diagnostic_latencies_accepted_as_main_performance=False)
    except Exception as error:
        state("failed", error_type=type(error).__name__, detail=str(error)[-1600:])
        raise


if __name__ == "__main__":
    main()
