#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Android Benchmark Script
#
# Cross-compiles executor_runner for Android (arm64-v8a), pushes it and a model
# to an Android device via adb, runs the benchmark, and summarizes results.
#
# Usage:
#   ./devtools/scripts/benchmark_android.sh <model.pte> [options]
#
# Options:
#   --build-tool <tool>  Build system: cmake (default) or buck
#   --warmup <N>         Number of warmup executions (default: 1)
#   --iterations <N>     Number of timed executions (default: 10)
#   --num-threads <N>    CPU threads for inference (default: -1, auto-detect)
#   --method <name>      Method to run (default: first method in the program)
#   --backends <list>    Comma-separated backends (default: xnnpack)
#                        Supported: xnnpack, coreml, vulkan, qnn
#                        (cmake only; buck links all backends)
#   --device <serial>    ADB device serial (for multiple devices)
#   --etdump             Enable event tracer and pull etdump back to host
#   --no-cleanup         Leave model file on device after benchmarking
#   --rebuild            Force cmake reconfigure and rebuild (cmake only)
#   --build-dir <path>   Reuse existing build directory (cmake only)

set -euo pipefail

# Pre-scan for --build-tool (needed before ExecuTorch root validation).
BUILD_TOOL="cmake"
_prev=""
for _arg in "$@"; do
  if [[ "$_prev" == "--build-tool" ]]; then BUILD_TOOL="$_arg"; break; fi
  _prev="$_arg"
done
unset _prev _arg

# --- Locate ExecuTorch root from script path (cmake only) ---
if [[ "$BUILD_TOOL" != "buck" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  EXECUTORCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

  if [[ ! -f "$EXECUTORCH_ROOT/CMakeLists.txt" ]] || [[ ! -d "$EXECUTORCH_ROOT/devtools" ]]; then
    echo "Error: Could not locate ExecuTorch root from script path: $SCRIPT_DIR"
    exit 1
  fi

  cd "$EXECUTORCH_ROOT"
fi

# --- Defaults ---
MODEL_PATH=""
# BUILD_TOOL is set by pre-scan above; re-declare here for documentation.
WARMUP=1
ITERATIONS=10
NUM_THREADS=-1
METHOD=""
BACKENDS="xnnpack"
DEVICE=""
ETDUMP=false
NO_CLEANUP=false
REBUILD=false
BUILD_DIR=""

DEVICE_DIR="/data/local/tmp/et_benchmark"

# --- Argument parsing ---
usage() {
  cat <<EOF
Usage: $0 <model.pte> [options]

Options:
  --build-tool <tool>  Build system: cmake (default) or buck
  --warmup <N>         Number of warmup executions (default: 1)
  --iterations <N>     Number of timed executions (default: 10)
  --num-threads <N>    CPU threads for inference (default: -1, auto-detect)
  --method <name>      Method to run (default: first method in the program)
  --backends <list>    Comma-separated backends to build (default: xnnpack)
                       Supported: xnnpack, vulkan, qnn (cmake only)
  --device <serial>    ADB device serial (for multiple devices)
  --etdump             Enable event tracer and pull etdump back to host
  --no-cleanup         Leave model file on device after benchmarking
  --rebuild            Force cmake reconfigure and rebuild (cmake only)
  --build-dir <path>   Reuse existing build directory (cmake only)
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-tool) BUILD_TOOL="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --num-threads) NUM_THREADS="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --backends) BACKENDS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --etdump) ETDUMP=true; shift ;;
    --no-cleanup) NO_CLEANUP=true; shift ;;
    --rebuild) REBUILD=true; shift ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    -h|--help) usage ;;
    *)
      if [[ -z "$MODEL_PATH" && "$1" != -* ]]; then
        MODEL_PATH="$1"; shift
      else
        echo "Unknown option: $1"; usage
      fi
      ;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  echo "Error: model path is required."
  usage
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Error: Model file not found: $MODEL_PATH"
  exit 1
fi

MODEL_NAME=$(basename "$MODEL_PATH")

# --- ADB helper ---
adb_cmd() {
  if [[ -n "$DEVICE" ]]; then
    adb -s "$DEVICE" "$@"
  else
    adb "$@"
  fi
}

# --- Locate NDK ---
find_ndk() {
  # Try directories in priority order.
  local candidates=()

  if [[ -n "${ANDROID_NDK:-}" ]]; then
    candidates+=("$ANDROID_NDK")
  fi
  if [[ -n "${ANDROID_NDK_HOME:-}" ]]; then
    candidates+=("$ANDROID_NDK_HOME")
  fi

  # SDK-relative ndk/ directories (pick latest version).
  local sdk_roots=()
  [[ -n "${ANDROID_HOME:-}" ]] && sdk_roots+=("$ANDROID_HOME")
  [[ -n "${ANDROID_SDK_ROOT:-}" ]] && sdk_roots+=("$ANDROID_SDK_ROOT")
  sdk_roots+=("$HOME/Library/Android/sdk")

  for sdk in "${sdk_roots[@]}"; do
    if [[ -d "$sdk/ndk" ]]; then
      local latest
      latest=$(ls -d "$sdk/ndk"/*/ 2>/dev/null | sort -V | tail -1)
      [[ -n "$latest" ]] && candidates+=("$latest")
    fi
  done

  candidates+=("/opt/ndk")

  local toolchain="build/cmake/android.toolchain.cmake"
  for ndk in "${candidates[@]}"; do
    # Strip trailing slash.
    ndk="${ndk%/}"
    if [[ -f "$ndk/$toolchain" ]]; then
      echo "$ndk"
      return 0
    fi
  done

  echo "Error: Could not find Android NDK. Searched:" >&2
  for c in "${candidates[@]}"; do
    echo "  $c" >&2
  done
  echo "" >&2
  echo "Set ANDROID_NDK or install the NDK via Android Studio." >&2
  return 1
}

# --- Build executor_runner ---
if [[ "$BUILD_TOOL" == "buck" ]]; then
  BUCK_TARGET="fbsource//xplat/executorch/examples/portable/executor_runner:executor_runner_optAndroid#android-arm64"
  BUCK_ARGS=(
    @fbsource//fbandroid/mode/static_linking
    @fbsource//fbandroid/mode/opt
    --config cxx.default_platform=android-arm64
  )
  if [[ "$ETDUMP" == true ]]; then
    BUCK_ARGS+=(-c executorch.event_tracer_enabled=true)
  fi
  echo "Building executor_runner with Buck..."
  BUCK_BUILD_OUTPUT=$(mktemp)
  if ! buck2 build "${BUCK_ARGS[@]}" "$BUCK_TARGET" --show-output >"$BUCK_BUILD_OUTPUT" 2>&1; then
    echo "Error: Buck build failed."
    cat "$BUCK_BUILD_OUTPUT"
    rm -f "$BUCK_BUILD_OUTPUT"
    exit 1
  fi
  # --show-output strips the #flavor suffix, so match on target without it.
  BUCK_TARGET_NO_FLAVOR="${BUCK_TARGET%%#*}"
  RUNNER_BIN=$(grep "$BUCK_TARGET_NO_FLAVOR" "$BUCK_BUILD_OUTPUT" | awk '{print $2}')
  rm -f "$BUCK_BUILD_OUTPUT"
  if [[ -z "$RUNNER_BIN" || ! -f "$RUNNER_BIN" ]]; then
    echo "Error: Could not locate output binary from buck build."
    exit 1
  fi
  echo "Build complete."
elif [[ -z "$BUILD_DIR" ]]; then
  BUILD_DIR="cmake-out-android-benchmark"

  ANDROID_NDK=$(find_ndk)

  # Resolve backend names to CMake flags.
  backends_lower=$(echo "$BACKENDS" | tr '[:upper:]' '[:lower:]')
  BACKEND_FLAGS=()
  IFS=',' read -ra backend_list <<< "$backends_lower"
  for b in "${backend_list[@]}"; do
    b=$(echo "$b" | tr -d ' ')
    case "$b" in
      xnnpack) BACKEND_FLAGS+=("-DEXECUTORCH_BUILD_XNNPACK=ON") ;;
      vulkan)  BACKEND_FLAGS+=("-DEXECUTORCH_BUILD_VULKAN=ON") ;;
      qnn)     BACKEND_FLAGS+=("-DEXECUTORCH_BUILD_QNN=ON") ;;
      *) echo "Error: Unknown backend '$b'. Supported: xnnpack, vulkan, qnn"; exit 1 ;;
    esac
  done

  cmake_args=(
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake"
    --preset android-arm64-v8a
    -DANDROID_PLATFORM=android-26
    -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON
    -DEXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL=ON
    -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON
    -DEXECUTORCH_ENABLE_LOGGING=ON
    -DCMAKE_BUILD_TYPE=Release
    "${BACKEND_FLAGS[@]}"
  )

  if [[ "$ETDUMP" == true ]]; then
    cmake_args+=(
      -DEXECUTORCH_ENABLE_EVENT_TRACER=ON
      -DEXECUTORCH_BUILD_DEVTOOLS=ON
    )
  else
    cmake_args+=(-DEXECUTORCH_ENABLE_EVENT_TRACER=OFF)
  fi

  # Check if we can skip the build entirely.
  ARGS_STAMP="$BUILD_DIR/.benchmark_cmake_args"
  CURRENT_ARGS=$(printf '%s\n' "${cmake_args[@]}" | sort)
  PREV_ARGS=""
  [[ -f "$ARGS_STAMP" ]] && PREV_ARGS=$(cat "$ARGS_STAMP")
  ARGS_CHANGED=false
  [[ "$CURRENT_ARGS" != "$PREV_ARGS" ]] && ARGS_CHANGED=true

  if [[ "$REBUILD" == false && "$ARGS_CHANGED" == false && -f "$BUILD_DIR/executor_runner" ]]; then
    echo "executor_runner already built, skipping build. Use --rebuild to force."
  else
    echo "Using NDK: $ANDROID_NDK"
    # Re-configure if args changed, --rebuild forced, or no CMakeCache yet.
    if [[ "$REBUILD" == true || "$ARGS_CHANGED" == true || ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
      echo "Configuring build..."
      cmake . "${cmake_args[@]}" -B "$BUILD_DIR"
      echo "$CURRENT_ARGS" > "$ARGS_STAMP"
    else
      echo "Build configuration unchanged, skipping configure."
    fi


    if [[ "$(uname)" == "Darwin" ]]; then
      CMAKE_JOBS=$(( $(sysctl -n hw.ncpu) - 1 ))
    else
      CMAKE_JOBS=$(( $(nproc) - 1 ))
    fi
    [[ "$CMAKE_JOBS" -lt 1 ]] && CMAKE_JOBS=1

    echo "Building executor_runner..."
    cmake --build "$BUILD_DIR" -j "$CMAKE_JOBS" --target executor_runner
    echo "Build complete."
  fi
else
  echo "Using existing build directory: $BUILD_DIR"
fi

# RUNNER_BIN is set by the buck build path above; for cmake, derive it here.
if [[ "$BUILD_TOOL" != "buck" ]]; then
  RUNNER_BIN="$BUILD_DIR/executor_runner"
  if [[ ! -f "$RUNNER_BIN" ]]; then
    echo "Error: executor_runner not found at $RUNNER_BIN"
    exit 1
  fi
fi

# --- Push to device ---
RUNNER_NAME=$(basename "$RUNNER_BIN")
echo "Pushing files to device..."
adb_cmd shell mkdir -p "$DEVICE_DIR"
adb_cmd push --sync "$RUNNER_BIN" "$DEVICE_DIR/"
adb_cmd push --sync "$MODEL_PATH" "$DEVICE_DIR/"
adb_cmd shell chmod +x "$DEVICE_DIR/$RUNNER_NAME"

DEVICE_MODEL="$DEVICE_DIR/$MODEL_NAME"

# --- Runner args ---
runner_args=(
  "--model_path=$DEVICE_MODEL"
  "--cpu_threads=$NUM_THREADS"
  "--print_output=none"
)

if [[ -n "$METHOD" ]]; then
  runner_args+=("--method_name=$METHOD")
fi

# --- Run helper: capture output, print on failure ---
RUNNER_OUTPUT=$(mktemp)
run_on_device() {
  local rc=0
  adb_cmd shell "$DEVICE_DIR/$RUNNER_NAME" "$@" > "$RUNNER_OUTPUT" 2>&1 || rc=$?
  if [[ "$rc" -ne 0 ]]; then
    echo ""
    echo "Error: executor_runner exited with code $rc"
    echo "--- device output ---"
    cat "$RUNNER_OUTPUT"
    echo "--- end output ---"
    # Also dump recent logcat for ET_LOG messages.
    echo "--- logcat (ExecuTorch) ---"
    adb_cmd logcat -d -s ExecuTorch:* | tail -30
    echo "--- end logcat ---"
    rm -f "$RUNNER_OUTPUT"
    exit 1
  fi
}

# --- Warmup ---
if [[ "$WARMUP" -gt 0 ]]; then
  echo "Running $WARMUP warmup iteration(s)..."
  warmup_args=("${runner_args[@]}" "--num_executions=$WARMUP")
  if [[ "$ETDUMP" == true ]]; then
    warmup_args+=("--etdump_path=$DEVICE_DIR/warmup.etdump")
  fi
  run_on_device "${warmup_args[@]}"
  adb_cmd shell rm -f "$DEVICE_DIR/warmup.etdump"
fi

# Clear logcat after warmup so the benchmark progress reader doesn't pick up
# stale entries. This must happen synchronously before the pipeline starts.
adb_cmd logcat -c

# --- Benchmark ---
# Clear any stale progress output from orphaned readers of previous runs.
[[ -t 1 ]] && printf "\r\033[K"
echo "Running $ITERATIONS benchmark iteration(s)..."

bench_args=("${runner_args[@]}" "--num_executions=$ITERATIONS")
if [[ "$ETDUMP" == true ]]; then
  bench_args+=("--etdump_path=$DEVICE_DIR/model.etdump")
fi

# Stream logcat for live progress. A wrapper subshell records adb logcat's PID
# so we can kill it directly at cleanup -- killing only the reader subshell of a
# plain pipeline leaves adb logcat alive, blocking subsequent adb calls.
LOGCAT_PID=""
READER_PID=""

if [[ -t 1 ]]; then
  LOGCAT_PID_FILE=$(mktemp)
  (
    adb_cmd logcat -s ExecuTorch:I 2>/dev/null &
    echo $! > "$LOGCAT_PID_FILE"
    wait
  ) 2>/dev/null | \
    while IFS= read -r line; do
      if [[ "$line" == *"Iteration "* ]]; then
        progress=$(echo "$line" | sed -n "s/.*Iteration \([0-9]* of $ITERATIONS\): \([0-9.]*\) ms/\1 (\2 ms)/p")
        [[ -n "$progress" ]] && printf "\r\033[K  Progress: %s" "$progress"
      fi
    done &
  READER_PID=$!
  disown "$READER_PID" 2>/dev/null || true
  while [[ ! -s "$LOGCAT_PID_FILE" ]]; do sleep 0.1; done
  LOGCAT_PID=$(cat "$LOGCAT_PID_FILE")
  rm -f "$LOGCAT_PID_FILE"
fi

run_on_device "${bench_args[@]}"

# Shut down the logcat pipeline: kill adb logcat → pipe closes → reader exits.
# disown above suppresses bash's "Terminated" job notification.
if [[ -n "$LOGCAT_PID" ]]; then kill "$LOGCAT_PID" 2>/dev/null || true; fi
if [[ -n "$READER_PID" ]]; then kill "$READER_PID" 2>/dev/null || true; fi
sleep 0.5
[[ -t 1 ]] && printf "\r\033[K"

# Parse the timing line from logcat.
LOGCAT_OUTPUT=$(adb_cmd logcat -d -s ExecuTorch:I)
TIMING_LINE=$(echo "$LOGCAT_OUTPUT" | grep "Model executed successfully" | tail -1 || true)
LOAD_LINE=$(echo "$LOGCAT_OUTPUT" | grep "Model loaded in" | tail -1 || true)
ITER_TIMES=$(echo "$LOGCAT_OUTPUT" | grep "Iteration " | sed -n 's/.*: \([0-9.]*\) ms/\1/p' || true)

# --- Pull etdump ---
if [[ "$ETDUMP" == true ]]; then
  ETDUMP_LOCAL="./${MODEL_NAME%.pte}.etdump"
  echo ""
  echo "Pulling etdump from device..."
  adb_cmd pull "$DEVICE_DIR/model.etdump" "$ETDUMP_LOCAL"
  adb_cmd shell rm -f "$DEVICE_DIR/model.etdump"
  echo "ETDump saved to $ETDUMP_LOCAL"

  # Run the inspector CLI to print a tabular summary.
  echo ""
  if [[ "$BUILD_TOOL" == "buck" ]]; then
    if ! buck2 run fbcode//executorch/devtools/inspector:inspector_cli -- \
        --etdump_path="$ETDUMP_LOCAL" 2>&1; then
      echo "Warning: inspector CLI failed. Analyze the ETDump manually:"
      echo "  buck2 run fbcode//executorch/devtools/inspector:inspector_cli -- --etdump_path=$ETDUMP_LOCAL"
    fi
  else
    "$EXECUTORCH_ROOT/run_python_script.sh" \
      "$EXECUTORCH_ROOT/devtools/inspector/inspector_cli.py" \
      --etdump_path="$ETDUMP_LOCAL"
  fi
fi

# --- Cleanup ---
if [[ "$NO_CLEANUP" == false ]]; then
  echo "Cleaning up model on device..."
  adb_cmd shell rm -f "$DEVICE_MODEL"
else
  echo "Skipping cleanup (--no-cleanup)."
fi
rm -f "$RUNNER_OUTPUT"

# --- Summarize ---
echo ""
echo "========================================="
echo " Benchmark Results"
echo "========================================="
echo "Model:      $MODEL_NAME"
if [[ -n "$DEVICE" ]]; then
  echo "Device:     $DEVICE"
fi
echo "Warmup:     $WARMUP iteration(s)"

if [[ -n "$LOAD_LINE" ]]; then
  LOAD_MS=$(echo "$LOAD_LINE" | sed -n 's/.*Model loaded in \([0-9.]*\) ms\..*/\1/p')
  if [[ -n "$LOAD_MS" ]]; then
    echo "Load:       $(printf '%.3f' "$LOAD_MS") ms"
  fi
fi

if [[ -n "$TIMING_LINE" ]]; then
  TOTAL_MS=$(echo "$TIMING_LINE" | sed -n 's/.*in \([0-9.]*\) ms\..*/\1/p')
  if [[ -n "$TOTAL_MS" ]]; then
    AVG_MS=$(echo "scale=3; $TOTAL_MS / $ITERATIONS" | bc)
    echo "Benchmark:  $ITERATIONS iteration(s) in $(printf '%.3f' "$TOTAL_MS") ms"
    echo "Average:    ${AVG_MS} ms/iteration"
    if [[ -n "$ITER_TIMES" ]]; then
      MIN_MS=$(echo "$ITER_TIMES" | sort -g | head -1)
      MAX_MS=$(echo "$ITER_TIMES" | sort -g | tail -1)
      echo "Min:        $(printf '%.3f' "$MIN_MS") ms"
      echo "Max:        $(printf '%.3f' "$MAX_MS") ms"
    fi
  else
    echo "Benchmark:  $ITERATIONS iteration(s) (could not parse timing)"
    echo "Raw output: $TIMING_LINE"
  fi
else
  echo "Benchmark:  $ITERATIONS iteration(s) (no timing data captured)"
  echo "Check logcat for ExecuTorch output."
fi

if [[ "$NUM_THREADS" -ne -1 ]]; then
  echo "Threads:    $NUM_THREADS"
fi
if [[ "$ETDUMP" == true ]]; then
  echo "ETDump:     $ETDUMP_LOCAL"
fi
echo "========================================="
