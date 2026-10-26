"""Export two mini arms in one process to retain identical launch-parameter picks.

This diagnostic reuses an already qualified native SDK. GEMM source and weight
storage are unchanged. Actual compiled configurations and numerical/performance
comparisons remain required after export; cache retention is not acceptance.
"""

import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(16 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preparation-job", "exporter-package", "gguf", "output", "lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument('--share-inductor-configs', action='store_true',
        help='Diagnostic reuse of legal generated reduction/pointwise configuration winners')
    parser.add_argument('--diagnostic-generated-gemm-backends',choices=['ATEN'],
        help='Only this numerical control: use ATen for Inductor-generated GEMM in both arms')
    parser.add_argument('--diagnostic-deterministic-generated-gemm',action='store_true',
        help='Only this numerical control: deterministic generated Triton GEMM algorithm selection')
    parser.add_argument('--reuse-failed-job', type=Path,
        help='Copy compiler caches from a confirmed terminal attempt; preserve its records')
    parser.add_argument("--owned-GPU-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    assert args.diagnostic_generated_gemm_backends is None or args.share_inductor_configs
    assert not args.diagnostic_deterministic_generated_gemm or args.share_inductor_configs
    assert not (args.diagnostic_generated_gemm_backends is not None and args.diagnostic_deterministic_generated_gemm)
    if args.reuse_failed_job:
        from failed_export_cache_reuse import verify_failed_export
        verify_failed_export(args.reuse_failed_job,args.output,
            ('prepare_independent_session_shared_autotune_mini.py',))

    def state(status, **fields):
        pending = args.output / "progress.pending.json"
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
                                           UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                           candidate_accepted=False, full_model_export_started=False,
                                           **fields), indent=2) + "\n")
        pending.replace(args.output / "progress.json")

    previous = json.loads((args.preparation_job / "progress.json").read_text())
    assert previous["status"] == "mini-artifacts-ready"
    gate_path = args.preparation_job / "native-gate.json"
    gate = json.loads(gate_path.read_text())
    if args.reuse_failed_job:
        previous_gate=json.loads((args.reuse_failed_job/'native-gate.json').read_text())
        assert previous_gate['imported_native_gate_SHA256']==digest(gate_path)
    assert gate["native_GPU_tests_pass"] and gate["test_count"] == 74
    source = Path(gate["source"])
    package = args.exporter_package
    manifest = json.loads((package / "public-source-manifest.json").read_text())
    assert manifest["public_only_source"]
    for name in ("export_independent_session_mini.py", "independent_session_capacity_source.json"):
        assert digest(package / name) == manifest["files"][name]["sha256"]
    assert digest(package / "independent_session_capacity_source.json") == gate["source_spec_SHA256"]
    for name, expected in gate["binaries"].items():
        assert digest(Path(name)) == expected
    assert digest(args.gguf) == "4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e"
    controls = {name: digest(source / name) for name in (
        "backends/cuda/triton/kernels/int4_quantized_gemm.py",
        "backends/cuda/triton/kernels/int5_quantized_gemm.py",
        "backends/cuda/triton/kernels/int6_quantized_gemm.py",
        "backends/cuda/triton/kernels/quantized_gemm_utils.py",
        "backends/cuda/autotune/launch_params.py",
        "extension/cuda/cuda_allocator.cpp")}
    if not args.owned_GPU_child:
        assert not args.output.exists()
        args.output.mkdir(parents=True)
        build = args.output / "model-build"
        build.mkdir()
        aliases = dict(gate)
        aliases["binaries"] = dict(gate["binaries"])
        aliases["imported_native_gate_SHA256"] = digest(gate_path)
        for name in ("muse_glimmer_runtime_kv_probe", "muse_glimmer_batching_worker"):
            original = args.preparation_job / "model-build" / name
            target = build / name
            target.symlink_to(original)
            aliases["binaries"][str(target)] = digest(original)
        (args.output / "native-gate.json").write_text(json.dumps(aliases, indent=2) + "\n")
        state("waiting-GPU-lock", source=str(source), original_native_gate_SHA256=digest(gate_path))
        with args.lock.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
            for role in ("mini-global", "mini-independent"):
                job = args.output / (role + "-export")
                job.mkdir()
                (job / "GPU-lock-owner.json").write_text(json.dumps(dict(pid=os.getpid())) + "\n")
            state("running-shared-autotune-mini-export", source=str(source))
            with (args.output / "export.log").open("x") as log:
                subprocess.run([sys.executable, "-B", "-u", str(Path(__file__).resolve()), *sys.argv[1:],
                                "--owned-GPU-child"], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=21600)
        proof = json.loads((args.output / "shared-autotune-cache-review.json").read_text())
        assert proof["existing_launch_picks_unchanged"] and proof["existing_module_autotune_picks_unchanged"]
        state("mini-artifacts-ready", source=str(source), native_gate=str(args.output / "native-gate.json"),
              source_spec_SHA256=gate["source_spec_SHA256"], shared_autotune_cache_review=str(args.output / "shared-autotune-cache-review.json"),
              actual_compiled_GEMM_choice_review_pending=True)
        return
    assert json.loads((args.output / "mini-global-export/GPU-lock-owner.json").read_text())["pid"] == os.getppid()
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("qualified_mini_exporter", package / "export_independent_session_mini.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    snapshots = []
    restore_inductor = None
    if args.share_inductor_configs:
        from shared_inductor_config_control import install_shared_inductor_config_control
        restore_inductor = install_shared_inductor_config_control(args.output/'shared-inductor-config-control.json',
            generated_gemm_backends=args.diagnostic_generated_gemm_backends,
            deterministic_generated_gemm=args.diagnostic_deterministic_generated_gemm)
    for role in ("mini-global", "mini-independent"):
        job = args.output / (role + "-export")
        sys.argv = [str(package / "export_independent_session_mini.py"), "--source", str(source),
                    "--native-gate", str(args.output / "native-gate.json"), "--gguf", str(args.gguf),
                    "--role", role, "--output", str(args.output / (role + "-artifact")),
                    "--job-dir", str(job), "--lock", str(args.lock), "--owned-GPU-child"]
        (job / "cache").mkdir(parents=True)
        if args.reuse_failed_job:
            from failed_export_cache_reuse import copy_compiler_caches
            copy_compiler_caches(args.reuse_failed_job/(role+'-export/cache'),
                job/'cache',job/'reused-compiler-cache.json')
        os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(job / "cache/inductor"), TRITON_CACHE_DIR=str(job / "cache/triton"))
        import torch
        import triton
        from executorch.backends.cuda.cuda_backend import CudaBackend
        import torch._inductor.cpp_builder as cpp
        original_builder = cpp.CppBuilder.build
        original_options = CudaBackend.__dict__["get_aoti_compile_options"]
        try:
            exporter.main()
        finally:
            cpp.CppBuilder.build = original_builder
            CudaBackend.get_aoti_compile_options = original_options
        from executorch.backends.cuda.autotune import launch_params
        launch_picks = {repr(key): value for key, value in launch_params._CACHE.items()}
        tuner_picks = {}
        for module_name in ("int4_quantized_gemm", "int5_quantized_gemm", "int6_quantized_gemm"):
            module = sys.modules["executorch.backends.cuda.triton.kernels." + module_name]
            for variable, value in vars(module).items():
                if not isinstance(value, dict):
                    continue
                for bucket, tuner in value.items():
                    if isinstance(tuner, triton.runtime.Autotuner):
                        tuner_picks[module_name + "." + variable + ":" + repr(bucket)] = {
                            repr(key): dict(kwargs=config.kwargs, num_warps=config.num_warps, num_stages=config.num_stages)
                            for key, config in tuner.cache.items()}
        snapshots.append(dict(role=role, launch_picks=launch_picks, module_tuner_picks=tuner_picks))
        (args.output / "shared-autotune-cache-snapshots.json").write_text(json.dumps(snapshots, indent=2) + "\n")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    first, second = snapshots
    assert first["launch_picks"]
    assert all(second["launch_picks"][key] == value for key, value in first["launch_picks"].items())
    for kernel, choices in first["module_tuner_picks"].items():
        assert all(second["module_tuner_picks"][kernel][key] == value for key, value in choices.items())
    assert all(digest(source / name) == expected for name, expected in controls.items())
    if restore_inductor is not None:
        restore_inductor()
    proof = dict(existing_launch_picks_unchanged=True, existing_module_autotune_picks_unchanged=True,
                 original_launch_pick_count=len(first["launch_picks"]), final_launch_pick_count=len(second["launch_picks"]),
                 unchanged_source_controls=controls, source_spec_SHA256=gate["source_spec_SHA256"],
                 snapshots_SHA256=digest(args.output / "shared-autotune-cache-snapshots.json"),
                 actual_compiled_choices_not_yet_verified=True, candidate_accepted=False,
                 full_model_export_started=False, native_gate_SHA256=digest(args.output / "native-gate.json"))
    if args.share_inductor_configs:
        control_path = args.output/'shared-inductor-config-control.json'
        control = json.loads(control_path.read_text())
        assert any(event['action']=='reuse' for event in control['events'])
        proof['shared_inductor_config_control_SHA256'] = digest(control_path)
        proof['generated_kernel_choices_reused_diagnostic_only'] = True
    (args.output / "shared-autotune-cache-review.json").write_text(json.dumps(proof, indent=2) + "\n")


if __name__ == "__main__":
    output = Path(sys.argv[sys.argv.index("--output") + 1]) if "--output" in sys.argv else None
    try:
        main()
    except Exception as error:
        if output is None:
            raise
        output.mkdir(parents=True, exist_ok=True)
        failure = dict(status="failed", controller_pid=os.getpid(), UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       candidate_accepted=False, full_model_export_started=False,
                       error=str(error), traceback=traceback.format_exc())
        (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        (output / "progress.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
