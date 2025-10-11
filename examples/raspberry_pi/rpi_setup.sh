#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# Prerequisites: Create a virtual environment using conda or venv and activate it.
# rpi_setup.sh - ExecuTorch Raspberry Pi Cross-Compilation Setup
# Usage: ./rpi_setup.sh [pi4|pi5]


set -euo pipefail

# Configuration
SCRIPT_VERSION="1.0"
TOOLCHAIN_VERSION="14.3.rel1"
WORKSPACE_DIR="$(pwd)"
TOOLCHAIN_DIR="${WORKSPACE_DIR}/arm-toolchain"
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
BUNDLED_GLIBC=true
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
            --bundled-glibc)
                BUNDLED_GLIBC=true
                shift
                ;;
            --no-bundled-glibc)
                BUNDLED_GLIBC=false
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
}

show_help() {
    cat << EOF
ExecuTorch Raspberry Pi Cross-Compilation Setup v${SCRIPT_VERSION}

Usage: $0 [pi4|pi5] [options]

Arguments:
  pi4|pi5               Target Raspberry Pi model

Options:
  --bundled-glibc       Bundle toolchain GLIBC libraries (default)
  --no-bundled-glibc    Use system GLIBC (requires RPi GLIBC upgrade)
  --force-rebuild       Force complete rebuild of all components
  --force-download      Force re-download of the ARM toolchain
  --skip-env-setup      Skip Python environment setup (just rebuild code)
  --clean               Remove build artifacts and exit
  -h, --help            Show this help message

Examples:
  $0 pi5                              # Setup for Pi5 with bundled GLIBC
  $0 pi4 --no-bundled-glibc           # Setup for Pi4 without bundled GLIBC
  $0 pi5 --force-rebuild              # Force complete rebuild
  $0 pi5 --skip-env-setup --force-rebuild  # Just rebuild code, skip env setup
  $0 pi5 --force-download             # Force re-download toolchain
  $0 pi5 --clean                      # Clean up build artifacts and exit

Note:
  Use the bundled GLIBC install script on your Raspberry Pi ONLY if you encounter a GLIBC version mismatch error when running llama_main.

EOF
}

# Clean up build artifacts
clean_up() {
    log_step "Cleaning up build artifacts"
    rm -rf "$TOOLCHAIN_DIR" "$CMAKE_OUT_DIR" arm-toolchain-*.cmake
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
        arm64|aarch64)
            if [[ "$HOST_OS" == "Darwin" ]]; then
                HOST_ARCH="x86_64"  # Use x86_64 toolchain on Apple Silicon via Rosetta
                log_info "Using x86_64 toolchain via Rosetta on Apple Silicon"
            else
                HOST_ARCH="aarch64"
            fi
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
    local required_tools=("cmake" "git" "curl" "tar")

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
            conda create -yn executorch python=3.10.0
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

    # Install ExecuTorch (CORRECTED SCRIPT NAME)
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

    # Check - verify compiler exists
    if [[ -f "$TOOLCHAIN_PATH/bin/aarch64-none-linux-gnu-gcc" ]]; then
        log_success "Toolchain already exists: $TOOLCHAIN_PATH"
        return 0
    fi

    local toolchain_archive="${toolchain_name}.tar.xz"
    local toolchain_url="https://developer.arm.com/-/media/Files/downloads/gnu/${TOOLCHAIN_VERSION}/binrel/${toolchain_archive}"

    # Create toolchain directory
    mkdir -p "$TOOLCHAIN_DIR"

    # Check if toolchain already exists
    if [[ -d "$TOOLCHAIN_DIR/$toolchain_name" ]]; then
        log_info "Toolchain already exists: $TOOLCHAIN_DIR/$toolchain_name"
    else
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

        # Extract toolchain
        log_info "Extracting toolchain..."
        tar -xf "$toolchain_archive"

        # Cleanup
        rm "$toolchain_archive"

        cd "$WORKSPACE_DIR"
    fi

    # Set toolchain path
    TOOLCHAIN_PATH="$TOOLCHAIN_DIR/$toolchain_name"

    # Verify toolchain
    if [[ ! -f "$TOOLCHAIN_PATH/bin/aarch64-none-linux-gnu-gcc" ]]; then
        log_error "Toolchain verification failed: compiler not found"
        exit 1
    fi

    log_success "ARM toolchain ready: $TOOLCHAIN_PATH"
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
            arch_flags="-march=armv8.2-a+dotprod+fp16"
            ;;
    esac

    log_info "Creating toolchain file: $cmake_file"
    log_info "Target: Raspberry Pi $RPI_MODEL"
    log_info "Architecture flags: $arch_flags"

    cat > "$cmake_file" << EOF
# ARM Toolchain for Raspberry Pi $RPI_MODEL
# Generated by rpi_setup.sh v${SCRIPT_VERSION}

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_CROSSCOMPILING TRUE)

# Path to ARM toolchain
set(TOOLCHAIN_PATH "$TOOLCHAIN_PATH")

# Compilers
set(CMAKE_C_COMPILER "\${TOOLCHAIN_PATH}/bin/aarch64-none-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "\${TOOLCHAIN_PATH}/bin/aarch64-none-linux-gnu-g++")

# Sysroot configuration
set(CMAKE_SYSROOT "\${TOOLCHAIN_PATH}/aarch64-none-linux-gnu/libc")
set(CMAKE_FIND_ROOT_PATH "\${CMAKE_SYSROOT}")

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

    # Check if already built (static OR shared)
    local extension_static="$CMAKE_OUT_DIR/extension/module/libextension_module.a"
    local extension_shared="$CMAKE_OUT_DIR/extension/module/libextension_module.so"

    if [[ -f "$extension_static" ]] || [[ -f "$extension_shared" ]]; then
        if [[ -f "$extension_static" ]]; then
            log_success "Core libraries already built (static): $(basename "$extension_static")"
        fi
        if [[ -f "$extension_shared" ]]; then
            log_success "Core libraries already built (shared): $(basename "$extension_shared")"
        fi

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

    # Configure (with -DBUILD_SHARED_LIBS=ON for shared libs)
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

# Extract bundled libraries (GLIBC bundling)
extract_bundled_libs() {
    if [[ "$BUNDLED_GLIBC" != "true" ]]; then
        return 0
    fi

    log_step "Extracting Bundled Libraries"

    local bundle_dir="$CMAKE_OUT_DIR/bundled-libs"
    mkdir -p "$bundle_dir"

    log_info "Extracting GLIBC libraries from toolchain..."

    # Copy essential libraries from toolchain
    local toolchain_lib_dir="$TOOLCHAIN_PATH/aarch64-none-linux-gnu/libc/lib64"
    local toolchain_usr_lib_dir="$TOOLCHAIN_PATH/aarch64-none-linux-gnu/libc/usr/lib/aarch64-linux-gnu"

    if [[ -d "$toolchain_lib_dir" ]]; then
        cp "$toolchain_lib_dir"/libc.so.* "$bundle_dir/" 2>/dev/null || true
        cp "$toolchain_lib_dir"/libm.so.* "$bundle_dir/" 2>/dev/null || true
        cp "$toolchain_lib_dir"/ld-linux-aarch64.so.* "$bundle_dir/" 2>/dev/null || true
    fi

    if [[ -d "$toolchain_usr_lib_dir" ]]; then
        cp "$toolchain_usr_lib_dir"/libstdc++.so.* "$bundle_dir/" 2>/dev/null || true
        cp "$toolchain_usr_lib_dir"/libgcc_s.so.* "$bundle_dir/" 2>/dev/null || true
    fi

    log_warning "Use bundled GLIBC script on RPI device ONLY if you encounter a GLIBC mismatch error when running llama_main."
    # Create deployment script
    cat > "$bundle_dir/install_libs.sh" << 'EOF'
#!/bin/bash
# Install bundled libraries on Raspberry Pi
# Usage: sudo ./install_libs.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="/usr/local/lib/executorch"

echo "Installing ExecuTorch bundled libraries..."

# Create directory
sudo mkdir -p "$LIB_DIR"

# Copy libraries
sudo cp "$SCRIPT_DIR"/*.so.* "$LIB_DIR/" 2>/dev/null || true

# Update library cache
sudo ldconfig

# Create environment setup script
cat > "$SCRIPT_DIR/setup_env.sh" << 'ENVEOF'
#!/bin/bash
export LD_LIBRARY_PATH="/usr/local/lib/executorch:$LD_LIBRARY_PATH"
ENVEOF

chmod +x "$SCRIPT_DIR/setup_env.sh"

echo "Installation complete!"
echo "Before running ExecuTorch binaries, source the environment:"
echo "  source setup_env.sh"
EOF

    chmod +x "$bundle_dir/install_libs.sh"

    log_success "Bundled libraries prepared in: $bundle_dir"
    log_info "On Raspberry Pi, run: sudo ./install_libs.sh"
}

# Verify build outputs
verify_build() {
    log_step "Verifying Build Outputs"

    local all_good=true
    local static_linked=false

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

    # Check extension module - both static AND shared might exist
    local extension_static="$CMAKE_OUT_DIR/extension/module/libextension_module.a"
    local extension_shared="$CMAKE_OUT_DIR/extension/module/libextension_module.so"

    if [[ -f "$extension_static" ]] && [[ -f "$extension_shared" ]]; then
        local size_a=$(ls -lh "$extension_static" | awk '{print $5}')
        local size_so=$(ls -lh "$extension_shared" | awk '{print $5}')
        log_success "✓ libextension_module.a ($size_a) - static"
        log_success "✓ libextension_module.so ($size_so) - shared"

        # Check if statically linked into llama_main
        if command -v nm &> /dev/null; then
            if nm "$CMAKE_OUT_DIR/examples/models/llama/llama_main" 2>/dev/null | grep -q "extension"; then
                log_info "📦 libextension_module.a is statically linked into llama_main"
                static_linked=true
            fi
        fi

    elif [[ -f "$extension_shared" ]]; then
        local size=$(ls -lh "$extension_shared" | awk '{print $5}')
        log_success "✓ libextension_module.so ($size) - shared library"
    elif [[ -f "$extension_static" ]]; then
        local size=$(ls -lh "$extension_static" | awk '{print $5}')
        log_success "✓ libextension_module.a ($size) - static library"
        static_linked=true
    else
        log_error "✗ Missing: extension module"
        all_good=false
    fi

    # Store for display_next_steps
    EXTENSION_STATIC_LINKED="$static_linked"

    if [[ "$all_good" == "true" ]]; then
        log_success "All required binaries built successfully!"
    else
        log_error "Some binaries are missing. Build may have failed."
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

    # Smart extension module reporting
    if [[ "$EXTENSION_STATIC_LINKED" == "true" ]]; then
        echo "  • libextension_module.a: Statically linked into llama_main ✅"
        if [[ -f "$CMAKE_OUT_DIR/extension/module/libextension_module.so" ]]; then
            echo "  • libextension_module.so: Built but not needed for deployment"
        fi
    else
        echo "  • libextension_module.so: $CMAKE_OUT_DIR/extension/module/libextension_module.so"
    fi

    if [[ "$BUNDLED_GLIBC" == "true" ]]; then
        echo "  • Bundled libraries: $CMAKE_OUT_DIR/bundled-libs/"
    fi

    echo -e "\n📋 Next steps:"

    echo "1. Copy binaries to your Raspberry Pi $RPI_MODEL:"
    echo "   scp $CMAKE_OUT_DIR/examples/models/llama/llama_main pi@<rpi-ip>:~/"
    echo "   scp $CMAKE_OUT_DIR/examples/models/llama/runner/libllama_runner.so pi@<rpi-ip>:~/"
    if [[ "$EXTENSION_STATIC_LINKED" != "true" ]]; then
        echo "   scp $CMAKE_OUT_DIR/extension/module/libextension_module.so pi@<rpi-ip>:~/"
    fi
    if [[ "$BUNDLED_GLIBC" == "true" ]]; then
        echo "   scp -r $CMAKE_OUT_DIR/bundled-libs/ pi@<rpi-ip>:~/"
    fi

    echo -e "\n2. Copy shared libraries to system location:"
    echo "   sudo cp libllama_runner.so /lib/  # Only this one needed!"
    echo "   sudo ldconfig"

    echo -e "\n3. Dry run to check for GLIBC or other issues:"
    echo "   ./llama_main --help"
    echo "   # Ensure there are no GLIBC or other errors before proceeding."

    echo -e "\n4. If you see GLIBC errors, install bundled libraries:"
    echo "   cd ~/bundled-libs && sudo ./install_libs.sh"
    echo "   source setup_env.sh"
    echo "   # Only do this if you encounter a GLIBC version mismatch or similar error."

    echo -e "\n5. Download your model and tokenizer:"
    echo "   # Make sure you have downloaded the model and tokenizer files from Hugging Face or your source."
    echo "   # Refer to the official documentation for exact details."

    echo -e "\n6. Run ExecuTorch with your model:"
    echo "   ./llama_main --model_path ./model.pte --tokenizer_path ./tokenizer.model --seq_len 128 --prompt \"What is the meaning of life ?\""

    echo -e "\n${GREEN}🎯 Deployment Summary:${NC}"
    if [[ "$EXTENSION_STATIC_LINKED" == "true" ]]; then
        echo "  📁 Files to copy: 2 (llama_main + libllama_runner.so)"
        echo "  🏗️  Extension module: Built-in (no separate .so needed)"
    else
        echo "  📁 Files to copy: 3 (llama_main + 2 .so files)"
        echo "  🏗️  Extension module: Separate .so file"
    fi

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
    detect_host
    check_prerequisites
    setup_environment
    download_toolchain
    create_toolchain_cmake
    build_executorch_core
    build_llama_runner
    extract_bundled_libs
    verify_build
    display_next_steps
}

# Run main function
main "$@"
