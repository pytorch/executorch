# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Preflight checks for the Arm VGF backend environment.

Examples:

    python -m executorch.backends.arm.vgf.check_env --aot
    python -m executorch.backends.arm.vgf.check_env --runtime
    python -m executorch.backends.arm.vgf.check_env --host-emulator
    python -m executorch.backends.arm.vgf.check_env --source-build --build-dir cmake-out-vkml

The default mode is --aot. It checks export/AoT prerequisites only.
Runtime, host-emulator, and source-build checks are explicit because pip-based
setup should cover most Python/package dependencies.

"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess  # nosec B404 - invoked only for trusted local tools
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from executorch.backends.arm.vgf.model_converter import (
    find_model_converter_binary,
    model_converter_env,
)


STATUS_OK = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"

VGF_BACKEND_NAME = "VgfBackend"

_REQUIRED_VKML_INSTANCE_LAYERS = {
    "VK_LAYER_ML_Graph_Emulation",
    "VK_LAYER_ML_Tensor_Emulation",
}

_VGF_LIBRARY_NAMES = ("libvgf.a", "libvgf.so", "libvgf.dylib")


@dataclass(frozen=True)
class VgfEnvironmentCheck:
    """One VGF environment preflight result."""

    name: str
    status: str
    detail: str
    action: str | None = None

    @property
    def ok(self) -> bool:
        return self.status != STATUS_FAIL

    def to_dict(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "action": self.action,
        }


@dataclass(frozen=True)
class VgfEnvironmentReport:
    """Structured VGF preflight report."""

    checks: list[VgfEnvironmentCheck]
    mode: str = "custom"

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    @property
    def failures(self) -> list[VgfEnvironmentCheck]:
        return [check for check in self.checks if check.status == STATUS_FAIL]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "ok": self.ok,
            "checks": [check.to_dict() for check in self.checks],
        }

    def raise_for_errors(self) -> None:
        if self.ok:
            return

        formatted_failures = "\n".join(_format_check(check) for check in self.failures)
        raise RuntimeError(
            "VGF environment validation failed:\n\n" + formatted_failures
        )

    def format(self) -> str:
        title = f"VGF environment preflight ({self.mode}): " + (
            "OK" if self.ok else "FAILED"
        )
        return "\n\n".join([title, *(_format_check(check) for check in self.checks)])


def check_vgf_aot_environment() -> VgfEnvironmentReport:
    """Check VGF AoT/export prerequisites.

    This is the default check. It intentionally avoids runtime, Vulkan, VKML,
    and source-build checks.

    """

    return VgfEnvironmentReport(
        mode="aot",
        checks=[
            _check_tosa_serializer(),
            _check_model_converter(),
            _check_model_converter_lib_dir(),
        ],
    )


def is_vgf_aot_available() -> bool:
    """Return True when VGF AoT/export prerequisites are available."""

    return check_vgf_aot_environment().ok


def check_vgf_runtime_environment() -> VgfEnvironmentReport:
    """Check whether the installed/runtime pybinding exposes VGF runtime
    support.
    """

    return VgfEnvironmentReport(
        mode="runtime",
        checks=[
            _check_runtime_vgf_backend(),
        ],
    )


def is_vgf_runtime_available() -> bool:
    """Return True when VGF runtime support is available."""

    return check_vgf_runtime_environment().ok


def check_vgf_host_emulator_environment() -> VgfEnvironmentReport:
    """Check host-emulator runtime prerequisites.

    This checks runtime backend registration plus Vulkan/VKML environment setup.

    """

    checks = [
        *_checks_from(check_vgf_runtime_environment()),
        _check_vulkan_sdk(),
        _check_emulation_layer(),
    ]
    return VgfEnvironmentReport(mode="host-emulator", checks=checks)


def check_vgf_source_build_environment(
    build_dir: str | os.PathLike[str] | None = None,
) -> VgfEnvironmentReport:
    """Check source-build diagnostics for the VGF runtime backend."""

    return VgfEnvironmentReport(
        mode="source-build",
        checks=[
            _check_vgf_library_path(),
            _check_cmake_build_flags(
                build_dir=build_dir,
                require_runtime_build=True,
            ),
        ],
    )


def check_environment(
    build_dir: str | os.PathLike[str] | None = None,
    *,
    require_runtime_build: bool = False,
) -> VgfEnvironmentReport:
    """Backward-compatible entry point.

    Existing callers get the AoT check by default. Callers that pass build_dir
    or require_runtime_build get the source-build diagnostic check.

    """

    if build_dir is not None or require_runtime_build:
        return check_vgf_source_build_environment(build_dir=build_dir)
    return check_vgf_aot_environment()


def _checks_from(report: VgfEnvironmentReport) -> list[VgfEnvironmentCheck]:
    return list(report.checks)


def _format_check(check: VgfEnvironmentCheck) -> str:
    lines = [f"[{check.status}] {check.name}", f"  {check.detail}"]
    if check.action:
        lines.append(f"  Action: {check.action}")
    return "\n".join(lines)


def _repo_root() -> Path:
    resolved = Path(__file__).resolve()
    for parent in resolved.parents:
        if (parent / "setup.py").is_file() and (parent / "backends" / "arm").is_dir():
            return parent

    # Normal source-tree fallback:
    # backends/arm/vgf/check_env.py -> repo root is parents[3].
    if len(resolved.parents) > 3:
        return resolved.parents[3]
    return resolved.parent


def _safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    seen: set[str] = set()
    deduped: list[Path] = []
    for path in paths:
        key = str(path.expanduser().resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path.expanduser())
    return deduped


def _split_env_paths(value: str | None) -> list[Path]:
    if not value:
        return []
    return [Path(part).expanduser() for part in value.split(os.pathsep) if part]


def _existing_env_paths(names: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for name in names:
        paths.extend(_split_env_paths(os.environ.get(name)))
    return [path for path in _dedupe_paths(paths) if _safe_is_dir(path)]


def _check_tosa_serializer() -> VgfEnvironmentCheck:
    try:
        serializer = importlib.import_module("tosa_serializer")
    except Exception as exc:
        return VgfEnvironmentCheck(
            "TOSA serializer",
            STATUS_FAIL,
            f"Could not import tosa_serializer: {exc}",
            "Install VGF AoT dependencies with "
            "python -m pip install 'executorch[vgf]' or, in a source checkout, "
            "python -m pip install --no-dependencies "
            "-r backends/arm/requirements-arm-tosa.txt.",
        )

    major = getattr(serializer, "TOSA_VERSION_MAJOR", None)
    minor = getattr(serializer, "TOSA_VERSION_MINOR", None)
    if major is not None and minor is not None:
        version = f"{major}.{minor}"
    else:
        version = getattr(serializer, "__version__", "<version unavailable>")

    return VgfEnvironmentCheck(
        "TOSA serializer",
        STATUS_OK,
        f"Imported tosa_serializer from {getattr(serializer, '__file__', '<unknown>')} "
        f"(version={version}).",
    )


def _resolve_executable(binary: str) -> Path | None:
    path = Path(binary)
    if path.is_absolute() or path.parent != Path("."):
        if _safe_is_file(path) and os.access(path, os.X_OK):
            return path
        return None

    resolved = shutil.which(binary)
    if resolved:
        return Path(resolved)
    return None


def _command_output(result: subprocess.CompletedProcess[str]) -> str:
    text = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    lines = text.splitlines()
    if not lines:
        return "<no output>"
    return "\n".join(lines[:4])


def _check_model_converter() -> VgfEnvironmentCheck:
    binary = find_model_converter_binary()
    if binary is None:
        return VgfEnvironmentCheck(
            "MLSDK model converter",
            STATUS_FAIL,
            "Could not find model-converter on PATH and MODEL_CONVERTER_PATH "
            "does not point to an executable file.",
            "Install VGF AoT dependencies with "
            "python -m pip install 'executorch[vgf]' or, in a source checkout, "
            "python -m pip install -r backends/arm/requirements-arm-vgf.txt. "
            "Alternatively set MODEL_CONVERTER_PATH to the converter executable.",
        )

    executable = _resolve_executable(binary)
    if executable is None:
        return VgfEnvironmentCheck(
            "MLSDK model converter",
            STATUS_FAIL,
            f"Resolved converter candidate {binary!r}, but it is not executable.",
            "Fix MODEL_CONVERTER_PATH or place model-converter on PATH.",
        )

    try:
        result = subprocess.run(  # nosec B603 - local converter executable
            [str(executable), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
            env=model_converter_env(),
        )
    except Exception as exc:
        return VgfEnvironmentCheck(
            "MLSDK model converter",
            STATUS_FAIL,
            f"Found {executable}, but running '--version' failed: {exc}",
            "Check MODEL_CONVERTER_LIB_DIR and the process loader paths. "
            "For source setup, source examples/arm/arm-scratch/setup_path.sh.",
        )

    if result.returncode != 0:
        return VgfEnvironmentCheck(
            "MLSDK model converter",
            STATUS_FAIL,
            f"{executable} --version exited with {result.returncode}:\n"
            f"{_command_output(result)}",
            "Check that the model-converter binary and its shared libraries are "
            "from the same MLSDK install.",
        )

    return VgfEnvironmentCheck(
        "MLSDK model converter",
        STATUS_OK,
        f"{executable} --version succeeded:\n{_command_output(result)}",
    )


def _check_model_converter_lib_dir() -> VgfEnvironmentCheck:
    lib_dir = os.environ.get("MODEL_CONVERTER_LIB_DIR")
    if not lib_dir:
        return VgfEnvironmentCheck(
            "MODEL_CONVERTER_LIB_DIR",
            STATUS_OK,
            "MODEL_CONVERTER_LIB_DIR is not set; relying on the process loader "
            "paths. This is OK when model-converter --version succeeds.",
        )

    path = Path(lib_dir).expanduser()
    if _safe_is_dir(path):
        return VgfEnvironmentCheck(
            "MODEL_CONVERTER_LIB_DIR",
            STATUS_OK,
            f"MODEL_CONVERTER_LIB_DIR points to existing directory: {path}",
        )

    return VgfEnvironmentCheck(
        "MODEL_CONVERTER_LIB_DIR",
        STATUS_FAIL,
        f"MODEL_CONVERTER_LIB_DIR={lib_dir!r} does not exist or is not a directory.",
        "Unset MODEL_CONVERTER_LIB_DIR or set it to the converter library directory.",
    )


def _load_runtime() -> Any:
    from executorch.runtime import Runtime

    return Runtime.get()


def _check_runtime_vgf_backend() -> VgfEnvironmentCheck:
    try:
        runtime = _load_runtime()
    except Exception as exc:
        return VgfEnvironmentCheck(
            "VGF runtime backend",
            STATUS_FAIL,
            f"Could not initialize executorch.runtime.Runtime: {exc}",
            "Install or rebuild ExecuTorch with runtime pybindings. For source "
            "builds, enable the VGF runtime backend and reinstall the package.",
        )

    try:
        registered_backend_names = list(
            runtime.backend_registry.registered_backend_names
        )
        is_available = runtime.backend_registry.is_available(
            backend_name=VGF_BACKEND_NAME
        )
    except Exception as exc:
        return VgfEnvironmentCheck(
            "VGF runtime backend",
            STATUS_FAIL,
            f"Runtime backend registry query failed: {exc}",
            "Reinstall or rebuild ExecuTorch with backend registry pybindings.",
        )

    if is_available:
        return VgfEnvironmentCheck(
            "VGF runtime backend",
            STATUS_OK,
            f"{VGF_BACKEND_NAME} is available in the runtime backend registry.",
        )

    rendered = ", ".join(registered_backend_names[:20])
    if len(registered_backend_names) > 20:
        rendered += ", ..."

    return VgfEnvironmentCheck(
        "VGF runtime backend",
        STATUS_FAIL,
        f"{VGF_BACKEND_NAME} is not available. Registered backends: "
        f"{rendered or '<none>'}.",
        "Use a runtime build/package that includes the VGF backend. For source "
        "builds, configure with -DEXECUTORCH_BUILD_VGF=ON and reinstall.",
    )


def _package_dirs(package: str) -> list[Path]:
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, AttributeError, ValueError):
        return []

    if spec is None:
        return []
    if spec.submodule_search_locations:
        return [Path(location) for location in spec.submodule_search_locations]
    if spec.origin:
        return [Path(spec.origin).parent]
    return []


def _candidate_vgf_library_dirs() -> list[Path]:
    repo = _repo_root()
    candidates: list[Path] = []

    for package_dir in _package_dirs("vgf_lib"):
        candidates.extend(
            [
                package_dir / "binaries" / "lib",
                package_dir / "deploy" / "lib",
                package_dir / "lib",
            ]
        )

    scratch_vgf = (
        repo / "examples/arm/arm-scratch/ml-sdk-for-vulkan-manifest/sw/vgf-lib"
    )
    candidates.extend(
        [
            scratch_vgf / "deploy" / "lib",
            scratch_vgf / "build" / "src",
        ]
    )

    candidates.extend(_split_env_paths(os.environ.get("LD_LIBRARY_PATH")))
    candidates.extend(_split_env_paths(os.environ.get("DYLD_LIBRARY_PATH")))
    return _dedupe_paths(candidates)


def _find_existing_lib(
    directories: Sequence[Path],
    names: Sequence[str],
) -> list[Path]:
    found: list[Path] = []
    for directory in directories:
        if not _safe_is_dir(directory):
            continue
        for name in names:
            candidate = directory / name
            if _safe_is_file(candidate):
                found.append(candidate)
    return _dedupe_paths(found)


def _check_vgf_library_path() -> VgfEnvironmentCheck:
    search_dirs = _candidate_vgf_library_dirs()
    found = _find_existing_lib(search_dirs, _VGF_LIBRARY_NAMES)

    if found:
        rendered = "\n".join(f"- {path}" for path in found[:8])
        return VgfEnvironmentCheck(
            "VGF library",
            STATUS_OK,
            f"Found libvgf candidate(s):\n{rendered}",
        )

    rendered_dirs = "\n".join(f"- {path}" for path in search_dirs[:12])
    return VgfEnvironmentCheck(
        "VGF library",
        STATUS_FAIL,
        "Could not find libvgf in the vgf_lib Python package, local scratch "
        f"tree, or loader paths. Searched:\n{rendered_dirs or '<no directories>'}",
        "For pip setup, install the VGF extra or ai_ml_sdk_vgf_library. For "
        "source-built MLSDK components, run "
        "backends/arm/scripts/setup-mlsdk-from-source.sh --enable-vgf-lib.",
    )


def _check_vulkan_sdk() -> VgfEnvironmentCheck:
    vulkan_sdk = os.environ.get("VULKAN_SDK")
    vulkan_sdk_path = Path(vulkan_sdk).expanduser() if vulkan_sdk else None
    vulkan_sdk_ok = vulkan_sdk_path is not None and _safe_is_dir(vulkan_sdk_path)

    glslc = shutil.which("glslc")
    vulkaninfo = shutil.which("vulkaninfo")

    details = [
        f"VULKAN_SDK={vulkan_sdk or '<unset>'}",
        f"glslc={glslc or '<not found>'}",
        f"vulkaninfo={vulkaninfo or '<not found>'}",
    ]

    if vulkan_sdk_ok and glslc and vulkaninfo:
        return VgfEnvironmentCheck(
            "Vulkan SDK",
            STATUS_OK,
            ", ".join(details),
        )

    problems = []
    if not vulkan_sdk_ok:
        problems.append("VULKAN_SDK is unset or does not point to a directory")
    if not glslc:
        problems.append("glslc was not found on PATH")
    if not vulkaninfo:
        problems.append("vulkaninfo was not found on PATH")

    return VgfEnvironmentCheck(
        "Vulkan SDK",
        STATUS_FAIL,
        "; ".join(problems) + ". " + ", ".join(details),
        "Install/source the Vulkan SDK. In the Arm setup flow, run "
        "examples/arm/setup.sh --i-agree-to-the-contained-eula "
        "--disable-ethos-u-deps --enable-mlsdk-deps and source "
        "examples/arm/arm-scratch/setup_path.sh.",
    )


def _split_vk_instance_layers(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part for part in re.split(r"[:;,]\s*", value) if part}


def _emulation_layer_deploy_dirs() -> list[Path]:
    deploy_dirs: list[Path] = []
    for package_dir in _package_dirs("emulation_layer"):
        deploy_dirs.append(package_dir / "deploy")

    repo = _repo_root()
    deploy_dirs.append(
        repo
        / "examples/arm/arm-scratch/ml-sdk-for-vulkan-manifest/sw/emulation-layer/deploy"
    )
    return _dedupe_paths(deploy_dirs)


def _check_emulation_layer() -> VgfEnvironmentCheck:
    layers = _split_vk_instance_layers(os.environ.get("VK_INSTANCE_LAYERS"))
    missing_layers = sorted(_REQUIRED_VKML_INSTANCE_LAYERS - layers)

    discovered_deploy_dirs = [
        path for path in _emulation_layer_deploy_dirs() if _safe_is_dir(path)
    ]
    configured_layer_dirs = _existing_env_paths(("VK_LAYER_PATH", "VK_ADD_LAYER_PATH"))
    configured_lib_dirs = _existing_env_paths(("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"))

    problems: list[str] = []
    if missing_layers:
        problems.append("VK_INSTANCE_LAYERS is missing " + ", ".join(missing_layers))
    if not configured_layer_dirs:
        problems.append(
            "VK_LAYER_PATH/VK_ADD_LAYER_PATH has no existing VKML layer directory"
        )
    if not configured_lib_dirs:
        problems.append(
            "LD_LIBRARY_PATH/DYLD_LIBRARY_PATH has no existing VKML library directory"
        )

    detail = (
        f"VK_INSTANCE_LAYERS={os.environ.get('VK_INSTANCE_LAYERS', '<unset>')}; "
        f"configured_layer_dirs="
        f"{[str(path) for path in configured_layer_dirs] or '<none>'}; "
        f"configured_lib_dirs="
        f"{[str(path) for path in configured_lib_dirs] or '<none>'}; "
        f"discovered_deploy_dirs="
        f"{[str(path) for path in discovered_deploy_dirs] or '<none>'}"
    )

    if problems:
        return VgfEnvironmentCheck(
            "VKML emulation layer",
            STATUS_FAIL,
            "; ".join(problems) + ". " + detail,
            "Source examples/arm/arm-scratch/setup_path.sh after installing "
            "MLSDK dependencies. For source-built MLSDK components, run "
            "backends/arm/scripts/setup-mlsdk-from-source.sh "
            "--enable-emulation-layer --enable-vulkan-sdk and source the "
            "generated setup_path.sh.",
        )

    return VgfEnvironmentCheck(
        "VKML emulation layer",
        STATUS_OK,
        detail,
    )


def _parse_cmake_cache(cache_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        key = key_and_type.split(":", 1)[0]
        values[key] = value
    return values


def _is_cmake_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.upper() in {"1", "ON", "TRUE", "YES", "Y"}


def _find_cmake_cache(
    build_dir: str | os.PathLike[str] | None,
    *,
    search_roots: Sequence[Path] | None = None,
) -> Path | None:
    if build_dir is not None:
        path = Path(build_dir).expanduser()
        if path.name == "CMakeCache.txt":
            return path if _safe_is_file(path) else None
        cache = path / "CMakeCache.txt"
        return cache if _safe_is_file(cache) else None

    roots = (
        list(search_roots) if search_roots is not None else [Path.cwd(), _repo_root()]
    )
    candidate_dirs = ("cmake-out", "cmake-out-vkml", "cmake-out-vgf")
    for root in _dedupe_paths(roots):
        for candidate_dir in candidate_dirs:
            cache = root / candidate_dir / "CMakeCache.txt"
            if _safe_is_file(cache):
                return cache
    return None


def _check_cmake_build_flags(
    build_dir: str | os.PathLike[str] | None,
    require_runtime_build: bool,
    *,
    search_roots: Sequence[Path] | None = None,
) -> VgfEnvironmentCheck:
    cache = _find_cmake_cache(build_dir, search_roots=search_roots)
    if cache is None:
        if build_dir is not None:
            return VgfEnvironmentCheck(
                "VGF source-build CMake flags",
                STATUS_FAIL,
                f"No CMakeCache.txt found for build_dir={build_dir!s}.",
                "Configure the runtime build with -DEXECUTORCH_BUILD_VGF=ON "
                "-DEXECUTORCH_BUILD_VULKAN=ON, then pass --build-dir <dir>.",
            )

        status = STATUS_FAIL if require_runtime_build else STATUS_WARN
        return VgfEnvironmentCheck(
            "VGF source-build CMake flags",
            status,
            "No CMakeCache.txt found in common build directories "
            "(cmake-out, cmake-out-vkml, cmake-out-vgf).",
            "Pass --build-dir <dir> after configuring the runtime build.",
        )

    values = _parse_cmake_cache(cache)
    required = {
        "EXECUTORCH_BUILD_VGF": values.get("EXECUTORCH_BUILD_VGF"),
        "EXECUTORCH_BUILD_VULKAN": values.get("EXECUTORCH_BUILD_VULKAN"),
    }
    bad = [key for key, value in required.items() if not _is_cmake_truthy(value)]
    rendered = ", ".join(
        f"{key}={value if value is not None else '<missing>'}"
        for key, value in required.items()
    )

    if bad:
        return VgfEnvironmentCheck(
            "VGF source-build CMake flags",
            STATUS_FAIL,
            f"{cache}: required runtime flag(s) are disabled or missing: "
            f"{', '.join(bad)}. Current values: {rendered}",
            "Reconfigure CMake with -DEXECUTORCH_BUILD_VGF=ON "
            "-DEXECUTORCH_BUILD_VULKAN=ON.",
        )

    return VgfEnvironmentCheck(
        "VGF source-build CMake flags",
        STATUS_OK,
        f"{cache}: {rendered}",
    )


def _select_report(args: argparse.Namespace) -> VgfEnvironmentReport:
    if args.runtime:
        return check_vgf_runtime_environment()
    if args.host_emulator:
        return check_vgf_host_emulator_environment()
    if args.source_build:
        return check_vgf_source_build_environment(build_dir=args.build_dir)
    return check_vgf_aot_environment()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preflight the Arm VGF backend environment."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--aot",
        action="store_true",
        help="Check VGF AoT/export prerequisites. This is the default.",
    )
    mode.add_argument(
        "--runtime",
        action="store_true",
        help="Check VGF runtime backend registration via executorch.runtime.",
    )
    mode.add_argument(
        "--host-emulator",
        action="store_true",
        help="Check host-emulator runtime prerequisites: runtime, Vulkan, and VKML.",
    )
    mode.add_argument(
        "--source-build",
        action="store_true",
        help="Check source-build diagnostics such as libvgf and CMake flags.",
    )
    parser.add_argument(
        "--build-dir",
        help="CMake build directory or CMakeCache.txt. Valid with --source-build.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable text.",
    )
    args = parser.parse_args(argv)

    if args.build_dir and not args.source_build:
        parser.error("--build-dir is only valid with --source-build")

    report = _select_report(args)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(report.format())

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
