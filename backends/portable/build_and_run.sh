#!/bin/bash
# Portable Backend Build & Test Script
# Usage: ./build_and_run.sh [generate|build|run|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ET_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$ET_ROOT/cmake-out"
PYTHON="${PYTHON:-/Users/scroy/miniconda3/envs/et-testing/bin/python}"
MODEL_PATH="/tmp/add_delegated.pte"

# Metal v2 options
USE_METAL_V2="${USE_METAL_V2:-1}"      # Enable metal_v2 by default
METAL_HEAP_SIZE="${METAL_HEAP_SIZE:-536870912}"  # 512MB default

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[portable]${NC} $1"; }
warn() { echo -e "${YELLOW}[portable]${NC} $1"; }
error() { echo -e "${RED}[portable]${NC} $1"; }
info() { echo -e "${CYAN}[portable]${NC} $1"; }

generate() {
    log "Generating test models..."
    cd "$ET_ROOT"
    $PYTHON backends/portable/test_export.py
    log "Models generated at /tmp/*_delegated.pte"
}

configure() {
    log "Configuring CMake..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Metal v2 option
    local METAL_V2_OPT="ON"
    if [ "$USE_METAL_V2" = "0" ]; then
        METAL_V2_OPT="OFF"
        info "Using original Metal runtime"
    else
        info "Using metal_v2 runtime with Metal 4 features"
    fi

    cmake "$ET_ROOT" \
        -DEXECUTORCH_BUILD_PORTABLE_BACKEND=ON \
        -DEXECUTORCH_PORTABLE_USE_METAL_V2=$METAL_V2_OPT

    log "CMake configured"
    if [ "$METAL_V2_OPT" = "ON" ]; then
        info "  metal_v2: ENABLED (ICB, ResidencySet, Heap, Binary Archives)"
        info "  Metal 4 features: GPU addresses, MTLResidencySet"
    fi
}

build() {
    log "Building portable_backend and executor_runner..."

    # Configure if needed
    if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
        configure
    fi

    cd "$BUILD_DIR"
    cmake --build . --target portable_backend executor_runner -j4

    if [ $? -eq 0 ]; then
        log "Build succeeded!"
    else
        error "Build failed!"
        exit 1
    fi
}

run() {
    local model="${1:-$MODEL_PATH}"

    if [ ! -f "$model" ]; then
        error "Model not found: $model"
        error "Run './build_and_run.sh generate' first"
        exit 1
    fi

    if [ ! -f "$BUILD_DIR/executor_runner" ]; then
        error "executor_runner not found"
        error "Run './build_and_run.sh build' first"
        exit 1
    fi

    log "Running model: $model"
    if [ "$USE_METAL_V2" = "1" ]; then
        info "Using metal_v2 runtime"
    fi
    echo "----------------------------------------"
    "$BUILD_DIR/executor_runner" --model_path "$model"
    echo "----------------------------------------"
    log "Done!"
}

bench() {
    local model="${1:-$MODEL_PATH}"
    local iterations="${2:-100}"

    if [ ! -f "$model" ]; then
        error "Model not found: $model"
        exit 1
    fi

    log "Benchmarking model: $model ($iterations iterations)"
    if [ "$USE_METAL_V2" = "1" ]; then
        info "metal_v2 enabled - expect fast replay after first inference"
    fi
    echo "----------------------------------------"
    # Run with timing
    time for i in $(seq 1 $iterations); do
        "$BUILD_DIR/executor_runner" --model_path "$model" 2>/dev/null
    done
    echo "----------------------------------------"
    log "Benchmark complete"
}

all() {
    generate
    build
    run
}

# Run the suite of metal_v2 test models exported by test_export.py.
# Exits with non-zero if any model fails.
test_metal_v2() {
    if [ ! -f "$BUILD_DIR/executor_runner" ]; then
        configure
        build
    fi

    log "Running metal_v2 test suite..."
    local models=(
        "/tmp/add_delegated.pte"            # binary vv (add/mul/sub kernels)
        "/tmp/matmul_delegated.pte"         # naive matmul (small)
        "/tmp/matmul_large_delegated.pte"   # matmul_simd (double-buffered, vec4)
        "/tmp/gemv_delegated.pte"           # gemv (N=1)
        "/tmp/matmul_m1_delegated.pte"      # gemv_t (M=1)
        "/tmp/linear_delegated.pte"         # matmul_nt via weight.t()
        "/tmp/attention_qk_delegated.pte"   # matmul_nt (Q @ K^T)
        "/tmp/matmul_tn_delegated.pte"      # matmul_tn (A^T @ B)
        "/tmp/bmm_delegated.pte"            # batched matmul_simd (tgid.z)
        "/tmp/all_ops_delegated.pte"        # mixed ops
    )

    local failed=0
    for m in "${models[@]}"; do
        if [ ! -f "$m" ]; then
            warn "Missing $m -- run './build_and_run.sh generate' first"
            failed=1
            continue
        fi
        info "  → $(basename $m)"
        if ! "$BUILD_DIR/executor_runner" --model_path "$m" >/tmp/_etest.out 2>&1; then
            error "    FAILED  ($m)"
            tail -20 /tmp/_etest.out | sed 's/^/      /'
            failed=1
        else
            log "    OK"
        fi
    done

    if [ $failed -ne 0 ]; then
        error "metal_v2 test suite: FAIL"
        exit 1
    fi
    log "metal_v2 test suite: PASS"
}

clean() {
    log "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
    log "Clean complete"
}

usage() {
    echo "Portable Backend Build & Test Script"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  generate        Export test models (add_delegated.pte, etc.)"
    echo "  configure       Configure CMake build"
    echo "  build           Build portable_backend and executor_runner"
    echo "  run [path]      Run model (default: /tmp/add_delegated.pte)"
    echo "  bench [path] [n] Benchmark model (default: 100 iterations)"
    echo "  test_metal_v2   Run all metal_v2 test models (build + run sweep)"
    echo "  all             Generate, build, and run"
    echo "  clean           Remove build directory"
    echo ""
    echo "Environment Variables:"
    echo "  USE_METAL_V2=1      Enable metal_v2 runtime (default: 1)"
    echo "  USE_METAL_V2=0      Use original Metal runtime"
    echo "  METAL_HEAP_SIZE=N   Heap size in bytes (default: 512MB)"
    echo ""
    echo "Metal v2 Features (when USE_METAL_V2=1):"
    echo "  - ICB (Indirect Command Buffer) for command replay"
    echo "  - GPU addresses via argument buffers (Metal 4)"
    echo "  - MTLResidencySet for GPU-resident memory (Metal 4)"
    echo "  - MTLHeap for fast allocation"
    echo "  - Binary Archives for pre-compiled shaders"
    echo "  - LRU Buffer Pool"
    echo ""
    echo "Examples:"
    echo "  $0 all                              # Full test cycle with metal_v2"
    echo "  USE_METAL_V2=0 $0 all               # Use original Metal runtime"
    echo "  $0 bench /tmp/linear_delegated.pte 50  # Benchmark 50 iterations"
}

# Main
case "${1:-}" in
    generate)
        generate
        ;;
    configure)
        configure
        ;;
    build)
        build
        ;;
    run)
        run "$2"
        ;;
    bench)
        bench "$2" "$3"
        ;;
    test_metal_v2)
        test_metal_v2
        ;;
    all)
        all
        ;;
    clean)
        clean
        ;;
    -h|--help|"")
        usage
        ;;
    *)
        error "Unknown command: $1"
        usage
        exit 1
        ;;
esac
