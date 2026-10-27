"""Build an isolated allocator experiment while preserving a measured working set.

This does not change the installed SDK, model artifacts, attention, or scheduler.
The runtime option remains experimental until both real mini models pass.
"""

import argparse
import datetime
import difflib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import traceback
import xml.etree.ElementTree as ET


ORIGINAL_SHA256 = "fcfcce93a9b7a0f92a1760eeea2a2d8f4978737e1df9c65d2612a70fe9e6e524"
GUARDED_SHA256 = "506b5d240ef05898e7102b1a944dcd6971c5bd47148b8f31a00f7b0b8e7d3fb5"
OPTION = "ET_CUDA_IDLE_KEEP_PREFILL_WORKING_SET"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patch_source(before):
    start = before.index("runtime::Error CudaKVPool::trim_idle_memory(")
    end = before.index("\nvoid CudaKVPool::trim_sequence_memory_pool", start)
    replacement = """runtime::Error CudaKVPool::trim_idle_memory(uint64_t unused_allowance) {
  ET_CHECK_OK_OR_RETURN_ERROR(synchronize());
  const auto device = CUDAGuard::create(device_);
  ET_CHECK_OK_OR_RETURN_ERROR(device.error());
  const char* mode = std::getenv("ET_CUDA_IDLE_KEEP_PREFILL_WORKING_SET");
  const bool preserve_prefill = mode && mode[0] == '1' && mode[1] == '\\0';
  auto trim = [&](cudaMemPool_t p, bool preserve_working_set) -> runtime::Error {
    if (!p) return runtime::Error::Ok;
    uint64_t used = 0, reserved = 0;
    ET_CHECK_OR_RETURN_ERROR(
        cudaMemPoolGetAttribute(p, cudaMemPoolAttrUsedMemCurrent, &used) == cudaSuccess &&
        cudaMemPoolGetAttribute(p, cudaMemPoolAttrReservedMemCurrent, &reserved) == cudaSuccess,
        Internal, "cannot read idle pool usage");
    uint64_t working_set = used;
    if (preserve_working_set) {
      uint64_t peak = 0;
      ET_CHECK_OR_RETURN_ERROR(
          cudaMemPoolGetAttribute(p, cudaMemPoolAttrUsedMemHigh, &peak) == cudaSuccess,
          Internal, "cannot read prefill working-set high water");
      working_set = std::max(used, peak);
    }
    const uint64_t keep = working_set > UINT64_MAX - unused_allowance
        ? UINT64_MAX : working_set + unused_allowance;
    if (reserved <= keep) return runtime::Error::Ok;
    ET_CHECK_OR_RETURN_ERROR(cudaMemPoolTrimTo(p, keep) == cudaSuccess,
        Internal, "cannot trim idle pool");
    return runtime::Error::Ok;
  };
  // KV migration has its own pool and existing capacity/contraction policy.
  ET_CHECK_OK_OR_RETURN_ERROR(trim(sequence_memory_pool_, false));
  return trim(::executorch::extension::cuda::CudaAllocator::pool_for_device(device_),
      preserve_prefill);
}
"""
    after = before[:start] + replacement + before[end:]
    if "#include <cstdlib>" not in after:
        anchor = "#include <executorch/backends/cuda/runtime/cuda_kv_pool.h>"
        assert anchor in after
        after = after.replace(anchor, "#include <cstdlib>\n\n" + anchor, 1)
    return after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True,
                        help="Existing qualified native preparation directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.parent / "native/executorch"
    build = source / "cmake-out"
    commands = []

    def state(status, **extra):
        value = dict(status=status, PID=os.getpid(),
                     UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     full_models_tested=False, candidate_accepted=False, **extra)
        pending = args.output / "progress.pending.json"
        pending.write_text(json.dumps(value, indent=2) + "\n")
        pending.replace(args.output / "progress.json")

    def run(argv, cwd=None, env=None):
        commands.append(dict(argv=[str(x) for x in argv],
                             cwd=str(cwd) if cwd else None))
        (args.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        with (args.output / "build-and-tests.log").open("a") as log:
            subprocess.run(argv, cwd=cwd, env=env, check=True,
                           stdout=log, stderr=subprocess.STDOUT)

    try:
        state("building-isolated-working-set-retention")
        original = source / "backends/cuda/runtime/cuda_kv_pool.cpp"
        assert sha(original) in (ORIGINAL_SHA256, GUARDED_SHA256)
        before = original.read_text()
        after = patch_source(before)
        cpp = args.output / "cuda_kv_pool.cpp"
        cpp.write_text(after)
        patch = args.output / "prefill-workset-retention.patch"
        patch.write_text("".join(difflib.unified_diff(
            before.splitlines(True), after.splitlines(True),
            fromfile="a/backends/cuda/runtime/cuda_kv_pool.cpp",
            tofile="b/backends/cuda/runtime/cuda_kv_pool.cpp")))
        flag_lines = (build / "backends/cuda/CMakeFiles/aoti_cuda_backend.dir/flags.make").read_text().splitlines()
        flags = []
        for name in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
            flags += shlex.split(next(x.split(" = ", 1)[1]
                                      for x in flag_lines if x.startswith(name + " = ")))
        obj = args.output / "cuda_kv_pool.cpp.o"
        run(["/usr/bin/c++", *flags, "-iquote", str(source / "backends/cuda/runtime"),
             "-o", str(obj), "-c", str(cpp)])
        archive = args.output / "libaoti_cuda_backend.a"
        shutil.copy2(build / "lib/libaoti_cuda_backend.a", archive)
        run(["ar", "r", str(archive), str(obj)])

        def link(linkfile, cwd, name):
            old = shlex.split(linkfile.read_text())
            argv = []
            for value in old:
                if value.startswith("-Wl,--dependency-file="):
                    continue
                value = value.replace(str(build / "lib/libaoti_cuda_backend.a"), str(archive))
                value = value.replace(str(build / "backends/cuda/libaoti_cuda_backend.a"), str(archive))
                if value == "libaoti_cuda_backend.a":
                    value = str(archive)
                elif not value.startswith("-") and value.endswith((".o", ".a", ".so")) and not Path(value).is_absolute():
                    value = str((cwd / value).resolve())
                argv.append(value)
            argv[argv.index("-o") + 1] = str(args.output / name)
            assert str(archive) in argv
            run(argv, cwd=cwd)

        link(args.parent / "probe-build/CMakeFiles/rewind_kv_probe.dir/link.txt",
             args.parent / "probe-build", "rewind_kv_probe")
        link(build / "backends/cuda/CMakeFiles/test_cuda_kv_cache.dir/link.txt",
             build / "backends/cuda", "test_cuda_kv_cache")
        state("waiting-native-GPU-gates")
        with args.lock.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            active = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
            assert not active.strip(), "Another GPU process is active; no experiment started"
            for enabled, label in ((False, "baseline"), (True, "working-set")):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
                env.pop("ET_CUDA_IDLE_SPARE_BYTES", None)
                env.pop(OPTION, None)
                if enabled:
                    env["ET_CUDA_IDLE_SPARE_BYTES"] = str(256 * 2**20)
                    env[OPTION] = "1"
                xml = args.output / (label + ".xml")
                argv = [str(args.output / "test_cuda_kv_cache"), "--gtest_output=xml:" + str(xml)]
                if enabled:
                    argv.append("--gtest_filter=CudaSplitRingCacheTest.*:CudaSplitRingBindingTest.*")
                run(argv, env=env)
                tests = list(ET.parse(xml).getroot().iter("testcase"))
                assert len(tests) == (7 if enabled else 81)
                assert all(t.find("failure") is None and t.find("error") is None
                           and t.find("skipped") is None for t in tests)
        gate = dict(native_tests_passed=True, native_test_count=81,
                    allocator_test_count=7, probe=str(args.output / "rewind_kv_probe"),
                    probe_SHA256=sha(args.output / "rewind_kv_probe"), source=str(source),
                    before_cpp_SHA256=sha(original), after_cpp_SHA256=sha(cpp),
                    archive_SHA256=sha(archive), option=OPTION,
                    working_set_policy="shared keep=max(current used, used high)+allowance; private unchanged",
                    model_qualification=False, experimental_patch=str(patch))
        (args.output / "native-gate.json").write_text(json.dumps(gate, indent=2) + "\n")
        state("native-ready-real-model-gates-pending", gate=gate)
    except BaseException as error:
        state("failed-preserve-evidence", error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
