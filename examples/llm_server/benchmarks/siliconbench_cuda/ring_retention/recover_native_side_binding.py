# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license in LICENSE.
"""Rebuild an isolated native binding repair and qualify real mini artifacts."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from repair_optional_side_bindings import patch
from wait_then_run_qualified_ring_recovery import process_start


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["native-preparation", "baseline-audit", "gguf", "output", "lock"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--model", choices=["MG", "Gemma"], required=True)
    p.add_argument("--export-job", type=Path)
    p.add_argument("--executable-root", type=Path)
    p.add_argument("--wait-process", action="append", default=[])
    p.add_argument("--jobs", type=int, default=8)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    executable_root = a.executable_root or a.output
    if a.executable_root:
        executable_root.mkdir(parents=True, exist_ok=False)
    pkg = Path(__file__).resolve().parent
    def state(status, **extra):
        value = dict(status=status, PID=os.getpid(), UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            model=a.model, real_mini_qualified=False, full_models_paused=True, **extra)
        temp = a.output / "progress.pending.json"
        temp.write_text(json.dumps(value, indent=2) + "\n")
        temp.replace(a.output / "progress.json")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
        CUDA_HOME="/usr/local/cuda", CUDAARCHS="120", TORCH_CUDA_ARCH_LIST="12.0")
    env.pop("CUDA_LAUNCH_BLOCKING", None)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    def run(argv, log, **kwargs):
        with (a.output / log).open("a") as stream:
            subprocess.run([str(v) for v in argv], env=kwargs.pop("env", env), stdout=stream,
                stderr=subprocess.STDOUT, check=True, timeout=kwargs.pop("timeout", 10800), **kwargs)
    try:
        watched = {int(s.split(":")[0]): s.split(":")[1] for s in a.wait_process}
        deadline = time.monotonic() + 10800
        state("waiting-existing-diagnostic", watched_processes=watched)
        while any(process_start(pid) == start for pid, start in watched.items()):
            if time.monotonic() >= deadline:
                state("observation-budget-exhausted-existing-diagnostic-not-restarted")
                return
            time.sleep(15)
        prior = json.loads((a.native_preparation / "native-gate.json").read_text())
        assert prior["native_tests_passed"] and prior["allocator_policy_test_count"] == 4
        oldsource = Path(prior["source"])
        paths = ["backends/cuda/runtime/cuda_kv_pool.h", "backends/cuda/runtime/cuda_kv_pool.cpp",
                 "backends/cuda/runtime/cuda_kv_cache.cpp", "backends/cuda/runtime/test/test_cuda_kv_cache.cpp"]
        before = {name: sha(oldsource / name) for name in paths}
        assert before[paths[2]] == prior["actual_native_Cpp_SHA256"]
        source = executable_root / "native/executorch"
        state("copying-isolated-native-source")
        shutil.copytree(oldsource, source, symlinks=True, ignore=shutil.ignore_patterns(
            ".git", "cmake-out", "__pycache__", "*.pte", "*.ptd", "*.gguf", "*.log"))
        filelists = "shim_et/xplat/executorch/build/build_variables.bzl"
        assert sha(source / filelists) == sha(oldsource / filelists)
        env["PYTHONPATH"] = str(source.parent)
        patch(source)
        testdir = source / "backends/cuda/runtime/test"
        shutil.copy2(pkg / "ring_optional_side_binding_tests.inc", testdir / "ring_optional_side_binding_tests.inc")
        testsource = testdir / "test_cuda_kv_cache.cpp"
        text = testsource.read_text()
        assert '#include "ring_retention_native_tests.inc"' in text
        assert '#include "ring_optional_side_binding_tests.inc"' not in text
        testsource.write_text(text + '\n#include "ring_optional_side_binding_tests.inc"\n')
        assert before == {name: sha(oldsource / name) for name in paths}
        audit = dict(prior_native_gate_SHA256=sha(a.native_preparation / "native-gate.json"),
            baseline_source=str(oldsource), candidate_source=str(source), actual_before_SHA256=before,
            actual_after_SHA256={name: sha(source / name) for name in paths}, old_source_unchanged=True,
            ring_kernels_unchanged=True, change="Only ABI4 seq_cols is optional; required buffers and present-buffer shape/dtype validation remain enforced")
        auditfile = a.output / "native-side-binding-source-audit.json"
        auditfile.write_text(json.dumps(audit, indent=2) + "\n")
        cmake = shutil.which("cmake", path=env["PATH"])
        assert cmake
        prefix = subprocess.check_output([sys.executable, "-B", "-c", "import torch; print(torch.utils.cmake_prefix_path)"], env=env, text=True).strip()
        build = source / "cmake-out"
        state("building-isolated-release-native")
        run([cmake, "-S", source, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
            "-DEXECUTORCH_BUILD_PRESET_FILE=" + str(source / "tools/cmake/preset/llm.cmake"),
            "-DEXECUTORCH_BUILD_CUDA=ON", "-DEXECUTORCH_BUILD_PYBIND=OFF",
            "-DEXECUTORCH_BUILD_TESTS=ON", "-DBUILD_TESTING=ON", "-DCMAKE_INSTALL_PREFIX=" + str(build),
            "-DCMAKE_PREFIX_PATH=" + prefix, "-DPYTHON_EXECUTABLE=" + sys.executable], "build.log", cwd=source)
        run([cmake, "--build", build, "--parallel", a.jobs, "--target", "test_cuda_kv_cache"], "build.log", cwd=source)
        binary = build / "backends/cuda/test_cuda_kv_cache"
        with a.lock.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
            state("running-native-storage-and-binding-gates")
            xml = a.output / "native-all-tests.xml"
            run([binary, "--gtest_output=xml:" + str(xml)], "native-tests.log", timeout=1800)
            tests = list(ET.parse(xml).getroot().iter("testcase"))
            assert len(tests) >= 81 and all(t.find("failure") is None and t.find("error") is None and t.find("skipped") is None for t in tests)
            ring = [t for t in tests if t.attrib["classname"] == "CudaSplitRingCacheTest"]
            binding = [t for t in tests if t.attrib["classname"] == "CudaSplitRingBindingTest"]
            assert len(ring) == 4 and len(binding) == 3
            trim_xml = a.output / "native-trim-tests.xml"
            run([binary, "--gtest_filter=CudaSplitRingCacheTest.*:CudaSplitRingBindingTest.*",
                "--gtest_output=xml:" + str(trim_xml)], "native-trim-tests.log", timeout=1800,
                env=dict(env, ET_CUDA_IDLE_SPARE_BYTES=str(256 * 2**20)))
            trimmed = list(ET.parse(trim_xml).getroot().iter("testcase"))
            assert len(trimmed) == 7 and all(t.find("failure") is None and t.find("error") is None and t.find("skipped") is None for t in trimmed)
        state("installing-qualified-native-SDK")
        run([cmake, "-S", source, "-B", build, "-DEXECUTORCH_BUILD_TESTS=OFF", "-DBUILD_TESTING=OFF"], "build.log", cwd=source)
        run([cmake, "--build", build, "--parallel", a.jobs, "--target", "install"], "build.log", cwd=source)
        probe_source = Path(prior["probe"]).parent.parent / "mini_probe.cpp"
        assert probe_source.is_file()
        local_probe = executable_root / "mini_probe.cpp"
        shutil.copy2(probe_source, local_probe)
        probe_build = executable_root / "probe-build"
        run([cmake, "-S", pkg.parent / "rewind_kv_probe_project", "-B", probe_build,
            "-DCMAKE_BUILD_TYPE=Release", "-DET_SOURCE=" + str(source),
            "-DREWIND_PROBE_SOURCE=" + str(local_probe), "-DCMAKE_PREFIX_PATH=" + str(build) + ";" + prefix], "probe-build.log", cwd=source)
        run([cmake, "--build", probe_build, "--parallel", a.jobs, "--target", "rewind_kv_probe"], "probe-build.log", cwd=source)
        prep = a.output / "native-preparation"
        prep.mkdir()
        gate = dict(prior, source=str(source), probe=str(probe_build / "rewind_kv_probe"),
            probe_SHA256=sha(probe_build / "rewind_kv_probe"), legacy_test_count=len(tests)-7,
            new_ring_test_names=[t.attrib["name"] for t in ring], allocator_policy_test_count=4,
            binding_test_count=3, allocator_binding_test_count=3,
            actual_native_Cpp_SHA256=sha(source / paths[2]),
            native_source_audit_SHA256=sha(auditfile), optional_side_binding_repaired=True,
            new_native_tests_XML_SHA256=sha(xml), new_allocator_tests_XML_SHA256=sha(trim_xml),
            real_mini_model_qualified=False)
        (prep / "native-gate.json").write_text(json.dumps(gate, indent=2) + "\n")
        state("qualifying-real-mini-artifacts", model_reexport_required=a.export_job is None)
        if a.export_job:
            argv = [sys.executable, "-B", "-u", pkg / "run_real_mini_four_arms.py",
                "--native-preparation", prep, "--export-job", a.export_job,
                "--output", a.output / "four-arms", "--lock", a.lock]
        else:
            argv = [sys.executable, "-B", "-u", pkg / "run_qualified_ring_recovery.py",
                "--native-preparation", prep, "--baseline-audit", a.baseline_audit, "--gguf", a.gguf,
                "--model", a.model, "--output", a.output / "qualification", "--lock", a.lock]
            if a.executable_root:
                argv += ["--cache-root", executable_root / "qualification-cache"]
        run(argv, "qualification.log", timeout=25200)
        state("real-mini-data-complete-review-required", performance_acceptance_pending=True)
    except BaseException as error:
        state("failed-preserve-evidence", error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
