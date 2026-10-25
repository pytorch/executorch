"""Prepare and compile an isolated runtime-column candidate without GPU work.

Production source, environments, models, binaries and prior logs stay untouched.
CUDA devices are hidden for the entire build. CPU toy fixtures may be generated
by CMake; this script never runs GPU tests, exports MG, starts an inference
server or accepts the candidate.
"""

import argparse
import datetime
import hashlib
import json
import os
import io
import tarfile
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--public-base-cache",
        type=Path,
        help="Read-only public checkout used for Git archives; no network or changes to its files",
    )
    parser.add_argument(
        "--prepared-source",
        type=Path,
        help="Resume an owned isolated source whose prior build is terminal; preserve its prior logs",
    )
    parser.add_argument("--jobs", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    spec = json.loads((package / "runtime_kv_columns_source.json").read_text())
    patch = package / "runtime_kv_columns.patch"
    assert digest(patch) == spec["patch_sha256"]
    assert args.baseline_source.is_dir() and args.python.is_file()
    assert not args.output.exists(), "Preserve every prior build and attempt"
    args.output.mkdir(parents=True)
    source = (
        args.prepared_source if args.prepared_source else args.output / "executorch"
    )
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="",
        TORCH_CUDA_ARCH_LIST="12.0",
        CUDAARCHS="120",
        CUDA_HOME="/usr/local/cuda",
        PYTHONDONTWRITEBYTECODE="1",
        MAX_JOBS=str(args.jobs),
        OMP_NUM_THREADS=str(args.jobs),
    )
    env["PATH"] = str(args.python.parent) + os.pathsep + env["PATH"]
    progress = args.output / "progress.json"

    def state(status, **fields):
        value = dict(
            status=status,
            controller_pid=os.getpid(),
            updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            CUDA_visible_devices="",
            GPU_tests_started=False,
            GPU_model_export_started=False,
            candidate_accepted=False,
            baseline_modified=False,
            **fields
        )
        temp = progress.with_suffix(".pending.json")
        temp.write_text(json.dumps(value, indent=2) + "\n")
        temp.replace(progress)
        with (args.output / "progress-history.jsonl").open("a") as history:
            history.write(json.dumps(value) + "\n")

    def run(argv, *, cwd=None, timeout=7200):
        with (args.output / "build.log").open("a") as log:
            log.write(
                json.dumps(dict(argv=[str(a) for a in argv], cwd=str(cwd))) + "\n"
            )
            log.flush()
            subprocess.run(
                [str(a) for a in argv],
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout,
            )

    def git(root, *argv):
        return subprocess.check_output(
            ["git", "-C", str(root), *argv], text=True, env=env
        ).strip()

    try:
        state("isolating-source")
        assert git(args.baseline_source, "rev-parse", "HEAD") == spec["base_commit"]
        baseline_status = git(args.baseline_source, "status", "--porcelain")
        dependency_archives = {}

        def archive_checkout(root, revision, dest, *, dependency=False):
            assert git(root, "rev-parse", "HEAD") == revision, str(root)
            if dependency:
                assert not git(
                    root, "status", "--porcelain", "--untracked-files=no"
                ), str(root)
            raw = subprocess.check_output(
                ["git", "-C", str(root), "archive", revision], env=env
            )
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
                for member in archive.getmembers():
                    target = dest / member.name
                    assert target.resolve().is_relative_to(dest.resolve())
                    assert not member.islnk()
                    if member.issym():
                        assert (
                            (target.parent / member.linkname)
                            .resolve()
                            .is_relative_to(dest.resolve())
                        )
                archive.extractall(dest, filter="data")
            if dependency:
                dependency_archives[str(dest.relative_to(source))] = dict(
                    commit=revision,
                    archive_sha256=hashlib.sha256(raw).hexdigest(),
                    baseline_dependency_unchanged=True,
                )
            for entry in git(root, "ls-tree", "-r", revision).splitlines():
                metadata, name = entry.split("\t", 1)
                mode, kind, commit = metadata.split()
                if mode == "160000":
                    assert kind == "commit"
                    dependency_root = args.baseline_source / (dest / name).relative_to(
                        source
                    )
                    archive_checkout(
                        dependency_root, commit, dest / name, dependency=True
                    )
            if dependency:
                assert not git(
                    root, "status", "--porcelain", "--untracked-files=no"
                ), str(root)

        prior_preparation = None
        if args.prepared_source:
            assert source.is_dir() and not source.resolve().is_relative_to(
                args.baseline_source.resolve()
            )
            prior = json.loads((source.parent / "progress.json").read_text())
            assert prior["status"] == "failed"
            assert not Path("/proc", str(prior["controller_pid"])).exists()
            assert (source / "cmake-out/CMakeCache.txt").is_file()
            for name, record in spec["changed_files"].items():
                assert digest(source / name) == record["after_sha256"], name
            prior_preparation = dict(
                source=str(source),
                terminal_prior_progress_sha256=digest(source.parent / "progress.json"),
                prior_build_logs_preserved=True,
            )
        elif args.public_base_cache:
            assert args.public_base_cache.is_dir()
            archive_checkout(args.public_base_cache, spec["base_commit"], source)
        else:
            # Fetch the public base independently of the baseline partial clone.
            run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--depth=1",
                    "--no-checkout",
                    "https://github.com/pytorch/executorch.git",
                    source,
                ]
            )
            run(
                [
                    "git",
                    "-C",
                    source,
                    "fetch",
                    "--depth=1",
                    "origin",
                    spec["base_commit"],
                ]
            )
            run(["git", "-C", source, "checkout", "--detach", spec["base_commit"]])
            # Archive exact clean baseline dependencies instead of relying on
            # fresh recursive downloads of unrelated upstream default branches.
            for entry in git(source, "ls-tree", "-r", spec["base_commit"]).splitlines():
                metadata, name = entry.split("\t", 1)
                mode, kind, commit = metadata.split()
                if mode == "160000":
                    assert kind == "commit"
                    archive_checkout(
                        args.baseline_source / name,
                        commit,
                        source / name,
                        dependency=True,
                    )
        # Preserve current task fixes outside the candidate diff, e.g. the
        # validated allocator policy. Copy tracked source only, never artifacts.
        inherited = {}
        for name in git(args.baseline_source, "diff", "--name-only").splitlines():
            if name in spec["changed_files"]:
                expected = spec["selected_baseline_sha256"].get(
                    name, spec["changed_files"][name]["before_sha256"]
                )
                assert digest(args.baseline_source / name) == expected, name
                continue
            assert name.startswith(
                (
                    "backends/cuda/",
                    "extension/cuda/",
                    "extension/llm/",
                    "examples/models/muse-glimmer/",
                    "examples/llm_server/",
                )
            ), name
            path = Path(name)
            assert (
                path.suffix in (".py", ".cpp", ".h", ".cu", ".cmake")
                or path.name == "CMakeLists.txt"
            ), name
            assert (args.baseline_source / name).is_file()
            shutil.copy2(args.baseline_source / name, source / name)
            inherited[name] = digest(source / name)
        if not args.prepared_source:
            for name, record in spec["changed_files"].items():
                if record["before_sha256"] is not None:
                    assert digest(source / name) == record["before_sha256"], name
            run(["git", "-C", source, "apply", "--check", patch])
            run(["git", "-C", source, "apply", patch])
        for name, record in spec["changed_files"].items():
            assert digest(source / name) == record["after_sha256"], name
        controls = {}
        for name in spec["unchanged_quant_files"]:
            assert digest(source / name) == digest(args.baseline_source / name), name
            controls[name] = digest(source / name)
        assert git(args.baseline_source, "status", "--porcelain") == baseline_status
        # This workflow exports with pure Python and runs the native C++
        # executor. The baseline itself has no _portable_lib Python extension;
        # requiring it here would add an unrelated build prerequisite.
        alias = source / "examples/models/muse_glimmer"
        if alias.is_symlink():
            assert alias.readlink() == Path("muse-glimmer") and alias.exists()
        else:
            assert not alias.exists()
            alias.symlink_to("muse-glimmer")
        # Match the source package resource staging in setup.py build_py.
        resources = {}
        for name in ["program.fbs", "scalar_type.fbs"]:
            original = source / "schema" / name
            dest = source / "exir/_serialize" / name
            if dest.exists():
                assert digest(dest) == digest(original)
            else:
                shutil.copy2(original, dest)
            resources[name] = digest(dest)
        torch_prefix = subprocess.check_output(
            [
                str(args.python),
                "-B",
                "-c",
                "import torch; print(torch.utils.cmake_prefix_path)",
            ],
            env=env,
            text=True,
        ).strip()
        cmake = Path(shutil.which("cmake", path=env["PATH"]))
        build = source / "cmake-out"
        env["PYTHONPATH"] = str(source.parent)
        state(
            "building-core",
            source=str(source),
            dependency_archives=dependency_archives,
            reused_preparation=prior_preparation,
            package_resources=resources,
            inherited_source=inherited,
            unchanged_quant_source=controls,
        )
        run(
            [
                cmake,
                "-S",
                source,
                "-B",
                build,
                "-DCMAKE_BUILD_TYPE=Release",
                "-DEXECUTORCH_BUILD_PRESET_FILE="
                + str(source / "tools/cmake/preset/llm.cmake"),
                "-DEXECUTORCH_BUILD_CUDA=ON",
                "-DEXECUTORCH_BUILD_PYBIND=OFF",
                "-DEXECUTORCH_BUILD_TESTS=ON",
                "-DBUILD_TESTING=ON",
                "-DCMAKE_INSTALL_PREFIX=" + str(build),
                "-DCMAKE_PREFIX_PATH=" + torch_prefix,
                "-DPYTHON_EXECUTABLE=" + str(args.python),
            ]
        )
        run(
            [
                cmake,
                "--build",
                build,
                "--parallel",
                args.jobs,
                "--target",
                "test_cuda_kv_cache",
                "install",
            ]
        )
        model_build = args.output / "muse-glimmer-build"
        state(
            "building-probe",
            source=str(source),
            dependency_archives=dependency_archives,
            reused_preparation=prior_preparation,
            package_resources=resources,
            inherited_source=inherited,
            unchanged_quant_source=controls,
        )
        run(
            [
                cmake,
                "-S",
                source / "examples/models/muse-glimmer",
                "-B",
                model_build,
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_PREFIX_PATH=" + str(build),
                "-DEXECUTORCH_BUILD_CUDA=ON",
                "-DMUSE_GLIMMER_BUILD_BATCHING=ON",
                "-DMUSE_GLIMMER_BUILD_RUNTIME_KV_PROBE=ON",
            ]
        )
        run(
            [
                cmake,
                "--build",
                model_build,
                "--parallel",
                args.jobs,
                "--target",
                "muse_glimmer_runtime_kv_probe",
                "muse_glimmer_batching_worker",
            ]
        )
        binaries = [
            build / "backends/cuda/test_cuda_kv_cache",
            model_build / "muse_glimmer_runtime_kv_probe",
            model_build / "muse_glimmer_batching_worker",
        ]
        assert all(path.is_file() for path in binaries)
        assert git(args.baseline_source, "status", "--porcelain") == baseline_status
        state(
            "CPU-build-completed",
            source=str(source),
            dependency_archives=dependency_archives,
            reused_preparation=prior_preparation,
            package_resources=resources,
            inherited_source=inherited,
            unchanged_quant_source=controls,
            patch_sha256=spec["patch_sha256"],
            binaries={str(path): digest(path) for path in binaries},
            next_gate="All 100 main performance cells, 40 full and 60 smoke correctness cells independently accepted before any GPU test/export",
        )
    except Exception as error:
        state(
            "failed",
            error_type=type(error).__name__,
            detail=str(error)[-1600:],
            traceback=traceback.format_exc()[-2500:],
        )
        raise


if __name__ == "__main__":
    main()
