#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024,2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

reset_buck() {
  # On MacOS, buck2 daemon can get into a weird non-responsive state
  buck2 kill && buck2 clean
  rm -rf ~/.buck/buckd
}

retry () {
    "$@" || (sleep 30 && reset_buck && "$@") || (sleep 60 && reset_buck && "$@")
}

clean_executorch_install_folders() {
  ./install_executorch.sh --clean
}

update_tokenizers_git_submodule() {
  echo "Updating tokenizers git submodule..."
  git submodule update --init
  pushd extension/llm/tokenizers
  git submodule update --init
  popd
}

install_executorch() {
  which pip
  # Install executorch, this assumes that Executorch is checked out in the
  # current directory.
  ./install_executorch.sh "$@"
  # Just print out the list of packages for debugging
  pip list
}

install_pip_dependencies() {
  pushd .ci/docker || return
  # Install all Python dependencies, including PyTorch
  pip install --progress-bar off -r requirements-ci.txt
  popd || return
}

dedupe_macos_loader_path_rpaths() {
  if [[ "$(uname)" != "Darwin" ]]; then
    return
  fi

  local torch_lib_dir
  local python_executable=${PYTHON_EXECUTABLE:-python}
  pushd ..
  torch_lib_dir=$("${python_executable}" -c "import importlib.util; print(importlib.util.find_spec('torch').submodule_search_locations[0])")/lib
  popd

  if [[ -z "${torch_lib_dir}" || ! -d "${torch_lib_dir}" ]]; then
    return
  fi

  local torch_libs=(
    "libtorch_cpu.dylib"
    "libtorch.dylib"
    "libc10.dylib"
  )

  for lib_name in "${torch_libs[@]}"; do
    local lib_path="${torch_lib_dir}/${lib_name}"
    if [[ ! -f "${lib_path}" ]]; then
      continue
    fi

    local removed=0
    # Repeatedly remove the @loader_path rpath entries until none remain.
    while install_name_tool -delete_rpath @loader_path "${lib_path}" 2>/dev/null; do
      removed=1
    done

    if [[ "${removed}" == "1" ]]; then
      install_name_tool -add_rpath @loader_path "${lib_path}" || true
    fi
  done
}

install_domains() {
  local python_executable=${PYTHON_EXECUTABLE:-python}
  echo "Install torchvision and torchaudio"
  "${python_executable}" -m pip install --no-build-isolation \
    "git+https://github.com/pytorch/audio.git@${TORCHAUDIO_VERSION}"
  "${python_executable}" -m pip install --no-build-isolation \
    "git+https://github.com/pytorch/vision.git@${TORCHVISION_VERSION}"
}

# DO NOT MERGE: publish only allowlisted fields, never raw Git traces.
_pytorch_clone_diagnostics() {
  local python_executable=${PYTHON_EXECUTABLE:-python}
  "${python_executable}" - "$@" 3<&0 <<'PYTORCH_CLONE_DIAGNOSTICS'
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path


def publish(name, value):
    """Write only sanitized fields to artifacts and the console."""
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    (output / name).write_text(text)
    try:
        print(f"PyTorch clone diagnostics: {name}\n{text}", flush=True)
    except BrokenPipeError:
        pass


def trace_events(path):
    """Keep process relationships, signals and exits, never raw trace text."""
    events, processes, malformed = [], {}, 0
    for line in path.read_text(errors="replace").splitlines() if path.exists() else []:
        try:
            source = json.loads(line)
        except ValueError:
            malformed += 1
            continue
        if not isinstance(source, dict):
            continue
        kind = source.get("event")
        if kind not in ("cmd_name", "child_start", "child_exit", "signal", "exit"):
            continue
        # Session IDs can contain text: publish only numeric process relationships.
        sid = source.get("sid", "")
        sid = sid if isinstance(sid, str) else ""
        if sid not in processes:
            processes[sid] = len(processes)
        event = {"event": kind, "process_index": processes[sid]}
        parent = sid.rsplit("/", 1)[0] if "/" in sid else None
        if parent in processes:
            event["parent_process_index"] = processes[parent]
        for key in ("child_id", "pid", "code", "signo", "t_abs", "t_rel"):
            value = source.get(key)
            if (
                type(value) in (int, float)
                and abs(value) <= 2**63 - 1
                and math.isfinite(value)
            ):
                event[key] = value
        if kind == "cmd_name":
            name = source.get("name")
            commands = ("clone", "remote-curl", "index-pack", "fetch", "fetch-pack")
            event["command"] = name if name in commands else "other"
        events.append(event)
    return {"present": path.is_file(), "events": events, "malformed_lines": malformed}


def transport_events(path):
    """Summarize known transport symptoms without headers, URLs or messages."""
    errors = {
        "dns_failed": r"Could not resolve (host|proxy)",
        "connect_failed": r"Failed to connect|Couldn't connect",
        "connection_reset": r"Connection reset by peer",
        "broken_pipe": r"Broken pipe",
        "timeout": r"timed out|Timeout was reached",
        "tls_error": r"SSL certificate problem|SSL_ERROR_|TLS connect error|SSL connect error",
        "http2_stream_error": r"HTTP/2.*(not closed cleanly|INTERNAL_ERROR|PROTOCOL_ERROR)",
    }
    events = []
    for line in path.read_text(errors="replace").splitlines() if path.exists() else []:
        event = {}
        http = re.search(r"\bHTTP/(1\.0|1\.1|2|3) ([1-5]\d{2})\b", line)
        if http:
            event.update(http_version=http[1], http_status=int(http[2]))
        tls = re.search(r"\bTLSv1(?:\.[0-3])?\b", line)
        if tls:
            event["tls_version"] = tls[0]
        categories = [
            name for name, pattern in errors.items() if re.search(pattern, line, re.I)
        ]
        if categories:
            event["categories"] = categories
        if event:
            events.append(event)
    return {"present": path.is_file(), "events": events}


action, raw, output, attempt = sys.argv[1:5]
raw, output = Path(raw), Path(output)
output.mkdir(parents=True, exist_ok=True)
status_name = f"attempt-{attempt}-status.json"
now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
if action == "start":
    publish(status_name, {"started_at": now, "started_monotonic": time.monotonic()})
    if attempt == "1":
        version = subprocess.run(
            ["git", "--version"], capture_output=True, text=True, timeout=5
        ).stdout
        version = re.match(r"git version (\d+(?:\.\d+){1,3})", version)
        streams = {}
        # fd 0 holds this program; fd 3 preserves the caller's stdin.
        for fd, name in ((3, "stdin"), (1, "stdout"), (2, "stderr")):
            mode = os.fstat(fd).st_mode
            kind = (
                "pipe"
                if stat.S_ISFIFO(mode)
                else "file" if stat.S_ISREG(mode) else "other"
            )
            streams[name] = {"tty": os.isatty(fd), "kind": kind}
        publish(
            "environment.json",
            {"git_version": version[1] if version else "unknown", "streams": streams},
        )
elif action == "finish":
    status = json.loads((output / status_name).read_text())
    status.update(finished_at=now, exit_code=int(sys.argv[5]))
    status["elapsed_seconds"] = time.monotonic() - status.pop("started_monotonic")
    publish(status_name, status)

    for suffix, label, parser in (
        ("trace2.jsonl", "trace2", trace_events),
        ("curl.log", "transport", transport_events),
    ):
        path = raw / f"attempt-{attempt}.{suffix}"
        publish(f"attempt-{attempt}-{label}.json", parser(path))
        path.unlink(missing_ok=True)

PYTORCH_CLONE_DIAGNOSTICS
}

install_pytorch_and_domains() {
  local python_executable=${PYTHON_EXECUTABLE:-python}
  pushd .ci/docker || return
  TORCH_VERSION=$(cat ci_commit_pins/pytorch.txt)
  popd || return

  # DO NOT MERGE: raw traces stay private; only allowlisted fields are uploaded.
  local trace_dir artifact_dir
  trace_dir=$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/pytorch-clone.XXXXXX")
  artifact_dir="${RUNNER_ARTIFACT_DIR:-${RUNNER_TEMP:-/tmp}/artifacts}/pytorch-clone"

  local clone_status
  _pytorch_clone_diagnostics start "${trace_dir}" "${artifact_dir}" 1 \
    || echo "Could not record clone start diagnostics." >&2
  # Keep the original arguments and inherited stdout/stderr (no tee/pipeline).
  if GIT_TRACE2_EVENT="${trace_dir}/attempt-1.trace2.jsonl" \
    GIT_TRACE_CURL="${trace_dir}/attempt-1.curl.log" \
    GIT_TRACE_CURL_NO_DATA=1 GIT_TRACE_REDACT=1 \
    git clone https://github.com/pytorch/pytorch.git; then
    clone_status=0
  else
    clone_status=$?
  fi
  _pytorch_clone_diagnostics finish "${trace_dir}" "${artifact_dir}" 1 "${clone_status}" \
    || echo "Could not publish allowlisted clone diagnostics; raw traces will not be uploaded." >&2
  if [[ "${clone_status}" -ne 0 ]]; then
    return "${clone_status}"
  fi

  # Fetch the target commit
  pushd pytorch || return
  git checkout "${TORCH_VERSION}"

  local system_name=$(uname)
  if [[ "${system_name}" == "Darwin" ]]; then
    local platform=$("${python_executable}" -c 'import sysconfig; import platform; v=platform.mac_ver()[0].split(".")[0]; platform=sysconfig.get_platform().split("-"); platform[1]=f"{v}_0"; print("_".join(platform))')
  fi
  local python_version=$("${python_executable}" -c 'import platform; v=platform.python_version_tuple(); print(f"{v[0]}{v[1]}")')
  local torch_release=$(cat version.txt)
  # Download key must match the upload key below (basename of dist/*.whl,
  # which always carries the build's resolved +gitHASH). Branch-ref pins
  # like `release/2.14` would otherwise produce `+gitrelease` here and
  # never hit the cache.
  local torch_short_hash=$(git rev-parse --short=7 HEAD)
  local torch_wheel_path="cached_artifacts/pytorch/executorch/pytorch_wheels/${system_name}/${python_version}"
  local torch_wheel_name="torch-${torch_release}%2Bgit${torch_short_hash}-cp${python_version}-cp${python_version}-${platform:-}.whl"

  local cached_torch_wheel="https://gha-artifacts.s3.us-east-1.amazonaws.com/${torch_wheel_path}/${torch_wheel_name}"
  # Cache PyTorch wheel is only needed on MacOS, Linux CI already has this as part
  # of the Docker image
  local torch_wheel_not_found=0
  if [[ "${system_name}" == "Darwin" ]]; then
    "${python_executable}" -m pip install "${cached_torch_wheel}" || torch_wheel_not_found=1
  else
    torch_wheel_not_found=1
  fi

  # Found no such wheel, we will build it from source then
  if [[ "${torch_wheel_not_found}" == "1" ]]; then
    echo "No cached wheel found, continue with building PyTorch at ${TORCH_VERSION}"

    git submodule update --init --recursive
    if [[ "$(uname -m)" == "aarch64" ]]; then
      export BUILD_IGNORE_SVE_UNAVAILABLE=1
    fi
    # PyTorch dropped setup.py. Build in a throwaway environment that can see the
    # active one, so its build requirements, which pin a cmake that would be
    # preferred over the one on PATH, cannot disturb what is installed here.
    #
    # Keep in sync with pytorch/pyproject.toml [build-system].requires.
    local build_venv=/tmp/pytorch-build-venv
    rm -rf "${build_venv}"
    "${python_executable}" -m venv --system-site-packages "${build_venv}"
    "${build_venv}/bin/pip" install build "scikit-build-core>=1.0" ninja \
      "packaging>=24.2" "typing-extensions>=4.10.0" pyyaml six numpy
    USE_DISTRIBUTED=1 "${build_venv}/bin/python" -m build --wheel --no-isolation
    rm -rf "${build_venv}"
    "${python_executable}" -m pip install "$(echo dist/*.whl)"
    # A build with no BLAS succeeds silently, so check rather than assume.
    (cd / && "${python_executable}" -c "
import torch
assert torch._C.has_lapack, 'built without LAPACK'
torch.linalg.qr(torch.randn(4, 4))
")

    # Invariant: the basename the build just produced must match the cache
    # URL we'd reconstruct on the next run. If they diverge (someone edits
    # torch_wheel_name above, or PyTorch renames its wheels), the cache
    # will silently miss and every macOS run will fall back to a ~30-min
    # source build. Fail loudly so the regression is caught immediately.
    shopt -s nullglob
    local built_wheels=(dist/*.whl)
    shopt -u nullglob
    if [[ ${#built_wheels[@]} -ne 1 ]]; then
      echo "ERROR: expected exactly 1 wheel in dist/, found ${#built_wheels[@]}" >&2
      exit 1
    fi
    local built_wheel_name
    built_wheel_name=$(basename "${built_wheels[0]}")
    local expected_wheel_name="${torch_wheel_name//\%2B/+}"
    if [[ "${built_wheel_name}" != "${expected_wheel_name}" ]]; then
      echo "ERROR: built torch wheel name does not match cache URL key:" >&2
      echo "  built:    ${built_wheel_name}" >&2
      echo "  expected: ${expected_wheel_name}" >&2
      echo "Fix torch_wheel_name construction in install_pytorch_and_domains" >&2
      echo "in .ci/scripts/utils.sh" >&2
      exit 1
    fi

    # Only AWS runners have access to S3
    if command -v aws && [[ -z "${GITHUB_RUNNER:-}" ]]; then
      for wheel_path in dist/*.whl; do
        local wheel_name=$(basename "${wheel_path}")
        echo "Caching ${wheel_name}"
        aws s3 cp "${wheel_path}" "s3://gha-artifacts/${torch_wheel_path}/${wheel_name}"
      done
    fi
  else
    echo "Use cached wheel at ${cached_torch_wheel}"
  fi

  dedupe_macos_loader_path_rpaths
  # Grab the pinned audio and vision commits from PyTorch
  TORCHAUDIO_VERSION=release/2.11
  export TORCHAUDIO_VERSION
  TORCHVISION_VERSION=release/0.29
  export TORCHVISION_VERSION

  install_domains

  popd || return
  # Print sccache stats for debugging
  sccache --show-stats || true
}

build_executorch_runner_buck2() {
  # Build executorch runtime with retry as this step is flaky on macos CI
  retry buck2 build //examples/portable/executor_runner:executor_runner
}

build_executorch_runner_cmake() {
  CMAKE_OUTPUT_DIR=cmake-out
  # Build executorch runtime using cmake
  clean_executorch_install_folders
  mkdir "${CMAKE_OUTPUT_DIR}"

  local build_type="${1:-Release}"
  local sanitizer_flag=""

  if [[ "${EXECUTORCH_USE_SANITIZER:-OFF}" == "ON" ]]; then
      sanitizer_flag="-DEXECUTORCH_USE_SANITIZER=ON"
  fi

  retry cmake \
    -DPYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" \
    -DCMAKE_BUILD_TYPE="${build_type}" \
    ${sanitizer_flag} \
    ${CMAKE_ARGS:-} \
    -B${CMAKE_OUTPUT_DIR} .

  if [ "$(uname)" == "Darwin" ]; then
    CMAKE_JOBS=$(( $(sysctl -n hw.ncpu) - 1 ))
  else
    CMAKE_JOBS=$(( $(nproc) - 1 ))
  fi
  cmake --build "${CMAKE_OUTPUT_DIR}" -j "${CMAKE_JOBS}"
}

build_executorch_runner() {
  if [[ $1 == "buck2" ]]; then
    build_executorch_runner_buck2
  elif [[ $1 == "cmake" ]]; then
    build_executorch_runner_cmake "$2"
  else
    echo "Invalid build tool $1. Only buck2 and cmake are supported atm"
    exit 1
  fi
}

cmake_install_executorch_lib() {
  build_type="${1:-Release}"
  echo "Installing libexecutorch.a and libportable_kernels.a"
  clean_executorch_install_folders
  retry cmake -DCMAKE_INSTALL_PREFIX=cmake-out \
          -DCMAKE_BUILD_TYPE=${build_type} \
          -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
          -Bcmake-out .
  cmake --build cmake-out -j9 --target install --config ${build_type}
}

download_stories_model_artifacts() {
  # Download stories110M.pt and tokenizer from Github
  curl -Ls --retry 3 --retry-all-errors "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt" --output stories110M.pt
  curl -Ls --retry 3 --retry-all-errors "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model" --output tokenizer.model
  # Create params.json file
  touch params.json
  echo '{"dim": 768, "multiple_of": 32, "n_heads": 12, "n_layers": 12, "norm_eps": 1e-05, "vocab_size": 32000}' > params.json
}

do_not_use_nightly_on_ci() {
  # An assert to make sure that we are not using PyTorch nightly on CI to prevent
  # regression as documented in https://github.com/pytorch/executorch/pull/6564
  TORCH_VERSION=$(pip list | grep -w 'torch ' | awk -F ' ' {'print $2'} | tr -d '\n')

  # The version of PyTorch building from source looks like 2.6.0a0+gitc8a648d that
  # includes the commit while nightly (2.6.0.dev20241019+cpu) or release (2.6.0)
  # won't have that. Note that we couldn't check for the exact commit from the pin
  # ci_commit_pins/pytorch.txt here because the value will be different when running
  # this on PyTorch CI
  if [[ "${TORCH_VERSION}" != *"+git"* ]]; then
    echo "Unexpected torch version. Expected binary built from source, got ${TORCH_VERSION}"
    exit 1
  fi
}


parse_args() {
  local args=("$@")
  local i
  local BUILD_TOOL=""
  local BUILD_MODE=""
  local EDITABLE=""
  for ((i=0; i<${#args[@]}; i++)); do
    case "${args[$i]}" in
      --build-tool)
        BUILD_TOOL="${args[$((i+1))]}"
        i=$((i+1))
        ;;
      --build-mode)
        BUILD_MODE="${args[$((i+1))]}"
        i=$((i+1))
        ;;
      --editable)
        EDITABLE="${args[$((i+1))]}"
        i=$((i+1))
        ;;
      *)
        echo "Invalid argument: ${args[$i]}"
        exit 1
        ;;
    esac
  done

  if [ -z "$BUILD_TOOL" ]; then
    echo "Missing build tool (require buck2 or cmake), exiting..."
    exit 1
  elif ! [[ $BUILD_TOOL =~ ^(cmake|buck2)$ ]]; then
    echo "Require buck2 or cmake for --build-tool, got ${BUILD_TOOL}, exiting..."
    exit 1
  fi
  BUILD_MODE="${BUILD_MODE:-Release}"
  if ! [[ "$BUILD_MODE" =~ ^(Debug|Release)$ ]]; then
    echo "Unsupported build mode ${BUILD_MODE}, options are Debug or Release."
    exit 1
  fi
  EDITABLE="${EDITABLE:-false}"
  if ! [[ $EDITABLE =~ ^(true|false)$ ]]; then
    echo "Require true or false for --editable, got ${EDITABLE}, exiting..."
    exit 1
  fi

  echo "$BUILD_TOOL $BUILD_MODE $EDITABLE"
}
