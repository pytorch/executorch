"""Build optional memory instrumentation over the qualified native SDK.

The original worker, probe, source and SDK libraries remain unchanged. The
adapter preserves the official server transport/scheduler and only samples
mutex-protected KV metrics and CUDA allocator attributes when enabled.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-job", "python", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    package = Path(__file__).resolve().parent

    def state(status, **fields):
        pending = args.output / "progress.pending.json"
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
                                           UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                           CPU_only=True, candidate_promoted=False, **fields), indent=2) + "\n")
        pending.replace(args.output / "progress.json")

    try:
        state("verifying-qualified-SDK")
        gate_path = args.native_job / "native-gate.json"
        gate = json.loads(gate_path.read_text())
        assert gate["native_GPU_tests_pass"] and gate["test_count"] >= 29
        progress = json.loads((args.native_job / "progress.json").read_text())
        assert progress["status"] == "native-GPU-tests-passed"
        assert not (Path("/proc") / str(progress["controller_pid"])).exists()
        source = Path(gate["source"])
        sdk = source / "cmake-out"
        library_paths = list((sdk / "lib").glob("*.a"))
        assert library_paths, "Installed SDK libraries unavailable"
        libraries = {str(path): digest(path) for path in library_paths}
        for name, expected in gate["binaries"].items():
            assert digest(Path(name)) == expected, name
        for name, expected in gate["unchanged_controls"].items():
            assert digest(source / name) == expected, name
        spec = json.loads((package / "kv_memory_worker_source.json").read_text())
        official = source / spec["official_worker_path"]
        assert digest(official) == spec["official_worker_SHA256"]
        assert digest(package / "kv_memory_worker.cpp") == spec["memory_worker_SHA256"]
        reconstructed = official.read_text().replace('#include <unistd.h>',
                                                     '#include "kv_memory_sampler.h"\n#include <unistd.h>')
        reconstructed = reconstructed.replace('#include <iostream>',
            '#include <iostream>\n\nDECLARE_string(backend);\nDECLARE_bool(dflash);')
        needle = '    auto batching_runtime = llm::create_muse_glimmer_batching_runtime();'
        assert reconstructed.count(needle) == 1
        reconstructed = reconstructed.replace(needle, needle +
            '\n    siliconbench::KVMemorySampler memory_sampler(\n'
            '        *batching_runtime->backend.executor, FLAGS_backend, FLAGS_dflash);')
        assert reconstructed == (package / "kv_memory_worker.cpp").read_text()
        sampler = (package / "kv_memory_sampler.h").read_text()
        assert 'cudaDeviceSynchronize' not in sampler and 'cudaStreamSynchronize' not in sampler
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "CUDA_HOME": "/usr/local/cuda",
               "CUDAARCHS": "120", "PYTHONDONTWRITEBYTECODE": "1"}
        env["PATH"] = str(args.python.parent) + os.pathsep + env["PATH"]
        cmake = shutil.which("cmake", path=env["PATH"])
        assert cmake is not None
        prefix = subprocess.check_output([str(args.python), "-B", "-c",
                                           "import torch; print(torch.utils.cmake_prefix_path)"],
                                          env=env, text=True).strip()
        build = args.output / "build"
        commands = [[cmake, "-S", str(package / "kv_memory_worker_project"), "-B", str(build),
                     "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                     "-DET_SOURCE=" + str(source), "-DCMAKE_PREFIX_PATH=" + str(sdk) + ";" + prefix],
                    [cmake, "--build", str(build), "--parallel", "2", "--target", "kv_memory_worker"]]
        state("building-memory-worker")
        with (args.output / "build.log").open("x") as log:
            for argv in commands:
                log.write(json.dumps(argv) + "\n")
                log.flush()
                subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=3600)
        for name, expected in libraries.items():
            assert digest(Path(name)) == expected, name
        for name, expected in gate["binaries"].items():
            assert digest(Path(name)) == expected, name
        assert digest(official) == spec["official_worker_SHA256"]
        binary = build / "kv_memory_worker"
        original_worker = args.native_job / "model-build/muse_glimmer_batching_worker"
        proof = dict(native_gate_SHA256=digest(gate_path), source=str(source), SDK=str(sdk),
                     unchanged_SDK_libraries=libraries,
                     original_worker_SHA256=digest(original_worker),
                     instrumented_worker=str(binary), instrumented_worker_SHA256=digest(binary),
                     official_worker_source_SHA256=digest(official),
                     optional_sampler_header_SHA256=digest(package / "kv_memory_sampler.h"),
                     official_control_flow_preserved=True, no_inference_kernel_change=True,
                     sampler_does_not_synchronize_GPU=True, sample_interval_ms=250,
                     CUDA_visibility_during_build="hidden", HTTP_validation_pending=True,
                     sampling_overhead_validation_pending=True,
                     compile_commands_SHA256=digest(build / "compile_commands.json"))
        (args.output / "proof.json").write_text(json.dumps(proof, indent=2) + "\n")
        state("memory-worker-built-pending-HTTP-validation", binary=str(binary))
    except Exception as error:
        state("failed", error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
