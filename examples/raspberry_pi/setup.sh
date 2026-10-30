#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# Prerequisites: Create a virtual environment using conda or venv and activate it.
# setup.sh - ExecuTorch Raspberry Pi Cross-Compilation Setup
# Usage: ./setup.sh [pi4|pi5]


set -euo pipefail

# Configuration
SCRIPT_VERSION="1.0"
TOOLCHAIN_VERSION="14.3.rel1"
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WORKSPACE_DIR"
TOOLCHAIN_DIR="${WORKSPACE_DIR}/examples/raspberry_pi/gen/arm-toolchain"
SYSROOT_BASE_DIR="${WORKSPACE_DIR}/examples/raspberry_pi/gen"
TARGET_OS="${TARGET_OS:-}"
SYSROOT_PATH=""
CMAKE_OUT_DIR="${WORKSPACE_DIR}/cmake-out"

FORCE_REBUILD="${FORCE_REBUILD:-false}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-false}"
SKIP_ENV_SETUP="${SKIP_ENV_SETUP:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
RPI_MODEL=""
HOST_OS=""
HOST_ARCH=""

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${GREEN}==== $1 ====${NC}"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            pi4|pi5)
                RPI_MODEL="$1"
                shift
                ;;
            --force-rebuild)
                FORCE_REBUILD=true
                shift
                ;;
            --force-download)
                FORCE_DOWNLOAD=true
                shift
                ;;
            --skip-env-setup)
                SKIP_ENV_SETUP=true
                shift
                ;;
            --os)
                TARGET_OS="${2:-}"
                shift 2
                ;;
            --os=*)
                TARGET_OS="${1#--os=}"
                shift
                ;;
            --clean)
                CLEAN=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    if [[ -z "$RPI_MODEL" ]]; then
        log_error "Please specify Raspberry Pi model: pi4 or pi5"
        show_help
        exit 1
    fi

    case "$TARGET_OS" in
        ""|bookworm|trixie) ;;
        bullseye)
            log_error "Bullseye (glibc 2.31) is not supported: the libraries bundled with the Arm toolchain"
            log_error "(e.g. libgcc's unwinder) require glibc >= 2.35. Bullseye is EOL; upgrade the Pi to Bookworm."
            exit 1
            ;;
        *)
            log_error "Unsupported --os: $TARGET_OS (expected bookworm or trixie)"
            exit 1
            ;;
    esac
}

show_help() {
    cat << EOF
ExecuTorch Raspberry Pi Cross-Compilation Setup v${SCRIPT_VERSION}

Usage: $0 [pi4|pi5] [options]

Arguments:
  pi4|pi5               Target Raspberry Pi model

Options:
  --os <codename>       Raspberry Pi OS release on the target: bookworm or trixie.
                        Builds against that Debian release's glibc (via mmdebstrap) so the
                        binaries run on it. Omit to use the toolchain's bundled glibc.
                        Check with: ssh pi@<rpi-ip> grep VERSION_CODENAME /etc/os-release
  --force-rebuild       Force complete rebuild of all components
  --force-download      Force re-download of the ARM toolchain and Debian sysroot
  --skip-env-setup      Skip Python environment setup (just rebuild code)
  --clean               Remove build artifacts and exit
  -h, --help            Show this help message

Examples:
  $0 pi5                              # Setup for Pi5
  $0 pi5 --os bookworm                # Pi5 running Raspberry Pi OS Bookworm
  $0 pi4 --os trixie                  # Pi4 running Raspberry Pi OS Trixie
  $0 pi5 --force-rebuild              # Force complete rebuild
  $0 pi5 --skip-env-setup --force-rebuild  # Just rebuild code, skip env setup
  $0 pi5 --force-download             # Force re-download toolchain
  $0 pi5 --clean                      # Clean up build artifacts and exit

EOF
}

# Clean up build artifacts
clean_up() {
    log_step "Cleaning up build artifacts"
    rm -rf "$TOOLCHAIN_DIR" "$SYSROOT_BASE_DIR"/sysroot-* "$CMAKE_OUT_DIR" arm-toolchain-*.cmake
    log_success "Cleanup complete. Exiting."
    exit 0
}

# Detect host system
detect_host() {
    log_step "Detecting Host System"

    HOST_OS=$(uname -s)
    HOST_ARCH=$(uname -m)

    case "$HOST_OS" in
        Linux)
            log_info "Detected Linux host"
            ;;
        Darwin)
            HOST_OS="Darwin"
            log_info "Detected macOS host"
            log_warning "Cross compilation on mac host is WIP"
            exit 1
            ;;
        *)
            log_error "Unsupported host OS: $HOST_OS"
            log_error "This script supports Linux and macOS only"
            exit 1
            ;;
    esac

    case "$HOST_ARCH" in
        x86_64|amd64)
            HOST_ARCH="x86_64"
            ;;
        aarch64|arm64)
            HOST_ARCH="aarch64"
            ;;
        *)
            log_error "Unsupported host architecture: $HOST_ARCH"
            exit 1
            ;;
    esac

    log_success "Host: $HOST_OS $HOST_ARCH"
}

# Check prerequisites
check_prerequisites() {
    log_step "Checking Prerequisites"

    # Check required tools
    local required_tools=("cmake" "git" "curl" "tar" "md5sum")

    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is not installed"
            case "$HOST_OS" in
                Linux)
                    log_info "Install with: sudo apt-get install $tool (Ubuntu/Debian) or equivalent"
                    ;;
                Darwin)
                    log_info "Install with: brew install $tool"
                    ;;
            esac
            exit 1
        fi
    done

    # Check CMake version
    local cmake_version=$(cmake --version | head -n1 | cut -d' ' -f3)
    local required_cmake="3.19"

    if ! command -v python3 -c "import sys; sys.exit(0 if tuple(map(int, '$cmake_version'.split('.'))) >= tuple(map(int, '$required_cmake'.split('.'))) else 1)" 2>/dev/null; then
        log_warning "CMake version $cmake_version detected. Recommended: $required_cmake+"
    fi

    # Check if we're in ExecuTorch directory
    if [[ ! -f "CMakeLists.txt" ]] || [[ ! -d "examples/models/llama" ]]; then
        log_error "Not in ExecuTorch root directory"
        log_info "Please run this script from the ExecuTorch root directory"
        exit 1
    fi

    log_success "All prerequisites satisfied"
}

# Detect and setup Python environment
setup_environment() {
    log_step "Setting Up Python Environment"

    if [[ "$SKIP_ENV_SETUP" == "true" ]]; then
        log_info "Skipping Python environment setup"
        return 0
    fi

    # Check if ExecuTorch is already installed
    if python3 -c "import executorch" 2>/dev/null; then
        log_success "ExecuTorch already installed and importable"
        return 0
    else
        log_warning "ExecuTorch not found in Python environment"
    fi

    # Check if already in virtual environment
    if [[ -n "${VIRTUAL_ENV:-}" ]] || [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
        if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
            log_success "Using existing conda environment: ${CONDA_DEFAULT_ENV}"
        else
            log_success "Using existing virtual environment: $(basename "${VIRTUAL_ENV:-}")"
        fi
    else
        log_warning "No virtual environment detected"

        # Check if conda is available
        if command -v conda &> /dev/null; then
            log_info "Creating conda environment..."
            conda create -yn executorch python=3.11
            eval "$(conda shell.bash hook)"
            conda activate executorch
            log_success "Created and activated conda environment: executorch"
        elif command -v python3 &> /dev/null; then
            log_info "Creating Python virtual environment..."
            python3 -m venv .venv
            source .venv/bin/activate
            pip install --upgrade pip
            log_success "Created and activated virtual environment: .venv"
        else
            log_error "Neither conda nor python3 found"
            exit 1
        fi
    fi

    # Initialize git submodules
    if [[ -f "extension/llm/tokenizers/CMakeLists.txt" ]]; then
        log_success "Git submodules already initialized"
    else
        log_info "Initializing git submodules..."
        git submodule update --init --recursive
    fi

    # Install ExecuTorch
    log_info "Installing ExecuTorch dependencies..."
    if [[ -f "./install_executorch.sh" ]]; then
        ./install_executorch.sh
    else
        log_error "./install_executorch.sh not found"
        log_info "Make sure you're in the ExecuTorch root directory"
        exit 1
    fi

    log_success "Python environment setup complete"
}

# Download and setup ARM toolchain
download_toolchain() {
    log_step "Setting Up ARM Toolchain"

    local toolchain_name="arm-gnu-toolchain-${TOOLCHAIN_VERSION}-${HOST_ARCH}-aarch64-none-linux-gnu"
    TOOLCHAIN_PATH="$TOOLCHAIN_DIR/$toolchain_name"

    # Check if toolchain already exists and force download not set
    if [[ -f "$TOOLCHAIN_PATH/bin/aarch64-none-linux-gnu-gcc" ]] && [[ "$FORCE_DOWNLOAD" != "true" ]]; then
        log_success "Toolchain already exists: $TOOLCHAIN_PATH"
        return 0
    fi

    local toolchain_archive="${toolchain_name}.tar.xz"
    local toolchain_url="https://developer.arm.com/-/media/Files/downloads/gnu/${TOOLCHAIN_VERSION}/binrel/${toolchain_archive}"

    # Create toolchain directory
    mkdir -p "$TOOLCHAIN_DIR"

    log_info "Downloading ARM toolchain: $toolchain_name"
    log_info "URL: $toolchain_url"

    cd "$TOOLCHAIN_DIR"

    # Download toolchain
    if ! curl -L -o "$toolchain_archive" "$toolchain_url"; then
        log_error "Failed to download toolchain"
        log_info "Please check the URL or download manually from:"
        log_info "https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads"
        exit 1
    fi

    # Verify MD5 checksum
    log_info "Verifying toolchain integrity..."

    # Define expected MD5 checksums
    local expected_md5=""
    case "${HOST_ARCH}-${TOOLCHAIN_VERSION}" in
        x86_64-14.3.rel1)
            expected_md5="b82906df613ca762058e3ca39ac2e23a"
            ;;
        aarch64-14.3.rel1)
            expected_md5="b82906df613ca762058e3ca39ac2e23a"
            ;;
        *)
            log_warning "No MD5 checksum available for ${HOST_ARCH}-${TOOLCHAIN_VERSION}, skipping verification"
            ;;
    esac

    if [[ -n "$expected_md5" ]]; then
        local actual_md5=$(md5sum "$toolchain_archive" | cut -d' ' -f1)
        if [[ "$actual_md5" != "$expected_md5" ]]; then
            log_error "MD5 checksum mismatch!"
            log_error "Expected: $expected_md5"
            log_error "Got:      $actual_md5"
            log_error "Toolchain download may be corrupted"
            exit 1
        fi
        log_success "MD5 checksum verified"
    fi

    # Extract toolchain
    log_info "Extracting toolchain..."
    tar -xf "$toolchain_archive"

    # Cleanup
    rm "$toolchain_archive"

    cd "$WORKSPACE_DIR"

    # Set toolchain path
    TOOLCHAIN_PATH="$TOOLCHAIN_DIR/$toolchain_name"

    # Verify critical toolchain files
    local critical_files=(
        "$TOOLCHAIN_PATH/bin/aarch64-none-linux-gnu-gcc"
        "$TOOLCHAIN_PATH/bin/aarch64-none-linux-gnu-g++"
    )

    for file in "${critical_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Critical toolchain file missing: $file"
            exit 1
        fi
    done

    log_success "ARM toolchain ready: $TOOLCHAIN_PATH"
}

# Create Debian sysroot for the target OS
create_debian_sysroot() {
    if [[ -z "$TARGET_OS" ]]; then
        return 0
    fi

    log_step "Creating Debian $TARGET_OS Sysroot"

    SYSROOT_PATH="$SYSROOT_BASE_DIR/sysroot-$TARGET_OS"
    SYSROOT_COMPAT_OBJ="$SYSROOT_PATH/glibc_compat.o"

    if [[ -f "$SYSROOT_PATH/usr/include/features.h" ]] && [[ "$FORCE_DOWNLOAD" != "true" ]]; then
        log_success "Sysroot already exists: $SYSROOT_PATH"
        log_info "glibc: $(sysroot_glibc_version "$SYSROOT_PATH")"
    else
        if ! command -v mmdebstrap &> /dev/null; then
            log_error "mmdebstrap is required for --os"
            log_info "Install with: sudo apt-get install mmdebstrap debian-archive-keyring"
            exit 1
        fi

        rm -rf "$SYSROOT_PATH"
        mkdir -p "$SYSROOT_BASE_DIR"

        log_info "Downloading arm64 glibc packages for Debian $TARGET_OS..."
        mmdebstrap \
            --mode=chrootless \
            --arch=arm64 \
            --variant=extract \
            --include=libc6,libc6-dev,linux-libc-dev \
            "$TARGET_OS" "$SYSROOT_PATH"

        if [[ ! -f "$SYSROOT_PATH/usr/include/features.h" ]]; then
            log_error "Sysroot creation failed: features.h missing in $SYSROOT_PATH"
            exit 1
        fi

        log_success "Sysroot ready: $SYSROOT_PATH (glibc $(sysroot_glibc_version "$SYSROOT_PATH"))"
    fi

    # merged-/usr releases create /lib -> usr/lib in base-files postinst, which extract mode skips.
    if [[ ! -e "$SYSROOT_PATH/lib" ]]; then
        ln -s usr/lib "$SYSROOT_PATH/lib"
    fi

    # Debian ships absolute symlinks (libm.so -> /lib/...), which ld follows outside the sysroot.
    find "$SYSROOT_PATH" -lname '/*' | while read -r link; do
        local target
        target="$SYSROOT_PATH$(readlink "$link")"
        ln -sfn "$(realpath -m --relative-to="$(dirname "$link")" "$target")" "$link"
    done

    # The fixincludes copy of glibc 2.38 pthread.h shadows the sysroot's and breaks <mutex>.
    rm -f "$TOOLCHAIN_PATH"/lib/gcc/aarch64-none-linux-gnu/*/include-fixed/pthread.h

    log_info "Compiling glibc compatibility shims..."
    "$TOOLCHAIN_PATH/bin/aarch64-none-linux-gnu-gcc" \
        --sysroot="$SYSROOT_PATH" \
        -isystem "$SYSROOT_PATH/usr/include/aarch64-linux-gnu" \
        -O2 -fPIC -c "$WORKSPACE_DIR/examples/raspberry_pi/glibc_compat.c" \
        -o "$SYSROOT_COMPAT_OBJ"
}

sysroot_glibc_version() {
    local features="$1/usr/include/features.h"
    local major minor
    major=$(grep -E '^#define[[:space:]]+__GLIBC__[[:space:]]' "$features" | awk '{print $3}')
    minor=$(grep -E '^#define[[:space:]]+__GLIBC_MINOR__[[:space:]]' "$features" | awk '{print $3}')
    echo "${major}.${minor}"
}

# Create CMake toolchain file
create_toolchain_cmake() {
    log_step "Creating CMake Toolchain File"

    local cmake_file="$WORKSPACE_DIR/arm-toolchain-${RPI_MODEL}.cmake"

    # Set architecture flags based on RPi model
    local arch_flags
    case "$RPI_MODEL" in
        pi4)
            arch_flags="-march=armv8-a -mtune=cortex-a72"
            ;;
        pi5)
            arch_flags="-march=armv8.2-a+dotprod+fp16 -mtune=cortex-a76"
            ;;
    esac

    log_info "Creating toolchain file: $cmake_file"
    log_info "Target: Raspberry Pi $RPI_MODEL"
    log_info "Architecture flags: $arch_flags"

    local sysroot_block
    if [[ -n "$SYSROOT_PATH" ]]; then
        sysroot_block=$(cat << EOF
# Sysroot: Debian $TARGET_OS arm64 (glibc $(sysroot_glibc_version "$SYSROOT_PATH"))
set(CMAKE_SYSROOT "$SYSROOT_PATH")
set(CMAKE_FIND_ROOT_PATH "\${CMAKE_SYSROOT}")

set(DEBIAN_MULTIARCH_INCLUDE "-isystem \${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu")
set(DEBIAN_MULTIARCH_LIB "-B\${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu -L\${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu")
set(STATIC_RUNTIME_FLAGS "-static-libstdc++ -static-libgcc $SYSROOT_COMPAT_OBJ")

# CMake reads this file once per enabled language, so guard against appending twice
# (a duplicated glibc_compat.o would be a multiple-definition link error).
if(NOT CMAKE_EXE_LINKER_FLAGS MATCHES "glibc_compat")
  set(CMAKE_C_FLAGS "\${CMAKE_C_FLAGS} \${DEBIAN_MULTIARCH_INCLUDE}" CACHE STRING "" FORCE)
  set(CMAKE_CXX_FLAGS "\${CMAKE_CXX_FLAGS} \${DEBIAN_MULTIARCH_INCLUDE}" CACHE STRING "" FORCE)
  set(CMAKE_EXE_LINKER_FLAGS "\${CMAKE_EXE_LINKER_FLAGS} \${DEBIAN_MULTIARCH_LIB} \${STATIC_RUNTIME_FLAGS}" CACHE STRING "" FORCE)
  set(CMAKE_SHARED_LINKER_FLAGS "\${CMAKE_SHARED_LINKER_FLAGS} \${DEBIAN_MULTIARCH_LIB} \${STATIC_RUNTIME_FLAGS}" CACHE STRING "" FORCE)
  set(CMAKE_MODULE_LINKER_FLAGS "\${CMAKE_MODULE_LINKER_FLAGS} \${DEBIAN_MULTIARCH_LIB} \${STATIC_RUNTIME_FLAGS}" CACHE STRING "" FORCE)
endif()
EOF
)
    else
        sysroot_block=$(cat << EOF
# Sysroot: bundled with the ARM toolchain
set(CMAKE_SYSROOT "\${TOOLCHAIN_PATH}/aarch64-none-linux-gnu/libc")
set(CMAKE_FIND_ROOT_PATH "\${CMAKE_SYSROOT}")
EOF
)
    fi

    cat > "$cmake_file" << EOF
# ARM Toolchain for Raspberry Pi $RPI_MODEL
# Generated by setup.sh v${SCRIPT_VERSION}

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_CROSSCOMPILING TRUE)

# Path to ARM toolchain
set(TOOLCHAIN_PATH "$TOOLCHAIN_PATH")

# Compilers
set(CMAKE_C_COMPILER "\${TOOLCHAIN_PATH}/bin/aarch64-none-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "\${TOOLCHAIN_PATH}/bin/aarch64-none-linux-gnu-g++")

$sysroot_block

# Architecture flags for Raspberry Pi $RPI_MODEL
set(ARCH_FLAGS "$arch_flags")

set(CMAKE_C_FLAGS "\${CMAKE_C_FLAGS} \${ARCH_FLAGS}" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "\${CMAKE_CXX_FLAGS} \${ARCH_FLAGS}" CACHE STRING "" FORCE)

# Search paths
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
EOF

    CMAKE_TOOLCHAIN_FILE="$cmake_file"
    log_success "Toolchain file created: $cmake_file"
}

# Build ExecuTorch core libraries
build_executorch_core() {
    log_step "Start building ExecuTorch core"

    # Check if already built (static only)
    local extension_static="$CMAKE_OUT_DIR/extension/module/libextension_module.a"

    if [[ -f "$extension_static" ]]; then
        log_success "Core libraries already built: $(basename "$extension_static")"
        if [[ "$FORCE_REBUILD" != "true" ]]; then
            log_info "Use --force-rebuild to rebuild core libraries"
            return 0
        else
            log_info "Force rebuilding core libraries..."
        fi
    fi

    # Only clean if forced or first build
    if [[ "$FORCE_REBUILD" == "true" ]] || [[ ! -d "$CMAKE_OUT_DIR" ]]; then
        log_info "Cleaning previous build..."
        rm -rf "$CMAKE_OUT_DIR"
    fi

    log_info "Building ExecuTorch core libraries..."

    # Configure
    cmake --preset llm \
        -DCMAKE_TOOLCHAIN_FILE="$CMAKE_TOOLCHAIN_FILE" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$CMAKE_OUT_DIR" \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON

    # Build with proper CPU count detection for Mac/Linux
    local cpu_count
    if command -v nproc &> /dev/null; then
        cpu_count=$(nproc)  # Linux
    elif command -v sysctl &> /dev/null; then
        cpu_count=$(sysctl -n hw.ncpu)  # Mac
    else
        cpu_count=4  # Fallback
    fi

    cmake --build "$CMAKE_OUT_DIR" -j"$cpu_count" --target install --config Release

    log_success "Core libraries built successfully"
}

# Build LLaMA runner
build_llama_runner() {
    log_step "Building LLaMA Runner"

    log_info "Building llama_main and other libs..."

    # Configure LLaMA build
    cmake -DCMAKE_TOOLCHAIN_FILE="$CMAKE_TOOLCHAIN_FILE" \
        -DCMAKE_INSTALL_PREFIX="$CMAKE_OUT_DIR" \
        -DBUILD_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -B"$CMAKE_OUT_DIR/examples/models/llama" \
        examples/models/llama

    # Build LLaMA runner
    cmake --build "$CMAKE_OUT_DIR/examples/models/llama" -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) --config Release

    log_success "LLaMA runner built successfully"
}

# Verify build outputs
verify_build() {
    log_step "Verifying Build Outputs"

    local all_good=true

    # Required files
    local required_files=(
        "$CMAKE_OUT_DIR/examples/models/llama/llama_main"
        "$CMAKE_OUT_DIR/examples/models/llama/runner/libllama_runner.so"
    )

    log_info "Checking required binaries..."

    for file in "${required_files[@]}"; do
        if [[ -f "$file" ]]; then
            local size=$(ls -lh "$file" | awk '{print $5}')
            log_success "✓ $(basename "$file") ($size)"
        else
            log_error "✗ Missing: $file"
            all_good=false
        fi
    done

    # Check extension module (static only)
    local extension_static="$CMAKE_OUT_DIR/extension/module/libextension_module.a"

    if [[ -f "$extension_static" ]]; then
        local size=$(ls -lh "$extension_static" | awk '{print $5}')
        log_success "✓ libextension_module.a ($size) - statically linked"
    else
        log_error "✗ Missing: extension module"
        all_good=false
    fi

    if [[ "$all_good" == "true" ]]; then
        log_success "All required binaries built successfully!"
    else
        log_error "Some binaries are missing. Build may have failed."
        exit 1
    fi

    check_glibc_requirement "${required_files[@]}"
}

check_glibc_requirement() {
    local objdump="$TOOLCHAIN_PATH/bin/aarch64-none-linux-gnu-objdump"
    if [[ ! -x "$objdump" ]]; then
        return 0
    fi

    local required
    required=$("$objdump" -T "$@" 2>/dev/null | grep -o 'GLIBC_2\.[0-9]*' | sort -t. -k2 -n -u | tail -1)
    required="${required#GLIBC_}"
    log_info "Binaries require glibc >= $required"

    if [[ -z "$SYSROOT_PATH" ]]; then
        log_warning "Built against the toolchain's bundled glibc. If the Pi reports 'GLIBC_$required not found', rerun with --os <codename>"
        return 0
    fi

    local available
    available=$(sysroot_glibc_version "$SYSROOT_PATH")
    if [[ "$(printf '%s\n%s\n' "$required" "$available" | sort -t. -k2 -n | tail -1)" == "$available" ]]; then
        log_success "Compatible with Debian $TARGET_OS (glibc $available)"
    else
        log_error "Binaries require glibc $required but Debian $TARGET_OS ships $available"
        exit 1
    fi
}

# Display next steps
display_next_steps() {
    log_step "Setup Complete!"

    echo -e "\n${GREEN}✓ ExecuTorch cross-compilation setup completed successfully!${NC}\n"

    echo "📦 Built binaries:"
    echo "  • llama_main: $CMAKE_OUT_DIR/examples/models/llama/llama_main"
    echo "  • libllama_runner.so: $CMAKE_OUT_DIR/examples/models/llama/runner/libllama_runner.so"
    echo "  • libextension_module.a: Statically linked into llama_main ✅"

    echo -e "\n📋 Next steps:"

    echo "1. [host PC] Create deployment directory and copy binaries to your Raspberry Pi $RPI_MODEL:"
    echo "   ssh pi@<rpi-ip> 'mkdir -p ~/executorch-deployment'"
    echo "   scp $CMAKE_OUT_DIR/examples/models/llama/llama_main pi@<rpi-ip>:~/executorch-deployment/"
    echo "   scp $CMAKE_OUT_DIR/examples/models/llama/runner/libllama_runner.so pi@<rpi-ip>:~/executorch-deployment/"

    echo -e "\n2. [Host PC → Raspberry Pi] Set up library environment:"
    echo "   ssh pi@<rpi-ip>"
    echo "   cd ~/executorch-deployment"
    echo "   echo 'export LD_LIBRARY_PATH=\$(pwd):\$LD_LIBRARY_PATH' > setup_env.sh"
    echo "   chmod +x setup_env.sh"

    echo -e "\n3. [Raspberry Pi] Test the deployment:"
    echo "   source setup_env.sh"
    echo "   ./llama_main --help"
    echo "   # Ensure there are no errors before proceeding. If there are errors, check the RaspberryPi Tutorials -> Troubleshooting section: hhttps://docs.pytorch.org/executorch/1.0/embedded-section.html#tutorials"

    echo -e "\n4. [Raspberry Pi] Download your model and tokenizer (they must end up in ~/executorch-deployment on the Raspberry Pi):"
    echo "   # Make sure you have downloaded the model and tokenizer files from Hugging Face or your source"
    echo "   # Refer to the official documentation for exact details"

    echo -e "\n5. [Raspberry Pi] Run ExecuTorch with your model:"
    echo "   ./llama_main --model_path ./model.pte --tokenizer_path ./tokenizer.model --seq_len 128 --prompt \"What is the meaning of life?\""

    echo -e "\n${GREEN}🎯 Deployment Summary:${NC}"
    echo "  📁 Files to deploy: 2 (llama_main + libllama_runner.so)"
    echo "  🏗️  Extension module: Statically linked (built-in)"
    echo "  📂 Target directory: ~/executorch-deployment/"
    echo "  🔧 Library path: Local directory with LD_LIBRARY_PATH"

    echo -e "\n🔧 Toolchain saved at: $TOOLCHAIN_PATH"
    echo "🔧 CMake toolchain file: $CMAKE_TOOLCHAIN_FILE"
    echo -e "\n${GREEN}Happy inferencing! 🚀${NC}"
}

# Main execution
main() {
    echo -e "${BLUE}"
    cat << "EOF"
╔══════════════════════════════════════════════════════════════╗
║              ExecuTorch Raspberry Pi Setup                   ║
║                Cross-Compilation Script                      ║
╚══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"

    parse_args "$@"

    # Handle clean option
    if [[ "${CLEAN:-false}" == "true" ]]; then
        clean_up
    fi

    detect_host
    check_prerequisites
    setup_environment
    download_toolchain
    create_debian_sysroot
    create_toolchain_cmake
    build_executorch_core
    build_llama_runner
    verify_build
    display_next_steps
}

# Run main function
main "$@"
