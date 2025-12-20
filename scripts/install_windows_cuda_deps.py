#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to install Windows CUDA dependencies for cross-compilation.
Supports Fedora/RHEL and WSL environments.

Detects CUDA version from the installed PyTorch to ensure compatibility.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Mapping of CUDA versions to their corresponding driver versions for Windows installers
# Source: https://developer.nvidia.com/cuda-toolkit-archive
CUDA_DRIVER_VERSION_MAP = {
    # CUDA 12.9.x
    "12.9.1": "576.57",
    "12.9.0": "576.33",
    # CUDA 12.8.x
    "12.8.1": "572.61",
    "12.8.0": "571.96",
    # CUDA 12.6.x
    "12.6.3": "561.17",
    "12.6.2": "560.94",
    "12.6.1": "560.94",
    "12.6.0": "560.76",
}


class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    NC = "\033[0m"  # No Color


def log_info(msg: str) -> None:
    print(f"{Colors.GREEN}[INFO]{Colors.NC} {msg}")


def log_warn(msg: str) -> None:
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {msg}")


def log_error(msg: str) -> None:
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")


def get_pytorch_cuda_version() -> tuple[str, str] | None:
    """
    Get the CUDA version from the installed PyTorch.

    Returns:
        A tuple of (cuda_version, driver_version) if found, None otherwise.
    """
    try:
        import torch
    except ImportError:
        log_error("PyTorch is not installed. Cannot detect CUDA version.")
        return None

    cuda_version = torch.version.cuda
    if cuda_version is None:
        log_error("PyTorch is not built with CUDA support.")
        return None

    log_info(f"Detected PyTorch CUDA version: {cuda_version}")

    # torch.version.cuda returns something like "12.4" (major.minor only)
    # We need to find a matching full version in our map
    matching_versions = [
        v for v in CUDA_DRIVER_VERSION_MAP.keys() if v.startswith(cuda_version)
    ]

    if not matching_versions:
        log_error(
            f"CUDA version {cuda_version} is not in the known version map. "
            f"Known versions: {', '.join(sorted(CUDA_DRIVER_VERSION_MAP.keys()))}"
        )
        return None

    # Use the latest patch version available
    full_cuda_version = sorted(matching_versions, reverse=True)[0]
    driver_version = CUDA_DRIVER_VERSION_MAP[full_cuda_version]

    log_info(f"Using CUDA {full_cuda_version} with driver {driver_version}")
    return full_cuda_version, driver_version


def run_command(
    cmd: list[str], check: bool = True, capture_output: bool = False
) -> subprocess.CompletedProcess:
    """Run a command and optionally check for errors."""
    log_info(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=True)


def detect_environment() -> str:
    """Detect the current environment (wsl, fedora, or unknown)."""
    # Check if running on Linux
    if platform.system() != "Linux":
        return "unknown"

    # Check for WSL
    try:
        with open("/proc/version", "r") as f:
            if "microsoft" in f.read().lower():
                return "wsl"
    except FileNotFoundError:
        pass

    # Check for RHEL/Fedora
    if Path("/etc/redhat-release").exists() or shutil.which("dnf"):
        return "fedora"

    return "unknown"


def install_mingw_fedora() -> None:
    """Install mingw64 on Fedora/RHEL."""
    log_info("Installing mingw64 for Fedora (dnf)...")
    run_command(["sudo", "dnf", "install", "-y", "mingw64-gcc-c++"])

    log_info("Verifying installation...")
    run_command(["x86_64-w64-mingw32-gcc", "--version"])


def install_mingw_wsl() -> None:
    """Install mingw64 on WSL."""
    log_info("Installing mingw64 for WSL...")
    run_command(["sudo", "apt", "update"])
    run_command(["sudo", "apt", "install", "-y", "g++-mingw-w64-x86-64-win32"])

    log_info("Verifying installation...")
    run_command(["x86_64-w64-mingw32-g++", "--version"])


def install_7zip(env_type: str) -> None:
    """Install 7zip if not already available."""
    if shutil.which("7z"):
        log_info("7zip already installed")
        return

    log_info("Installing 7zip...")
    if env_type == "fedora":
        run_command(["sudo", "dnf", "install", "-y", "p7zip", "p7zip-plugins"])
    else:
        run_command(["sudo", "apt", "install", "-y", "p7zip-full"])


def find_windows_cuda_install(cuda_version: str) -> Path | None:
    """
    Check if CUDA is installed on Windows (accessible via WSL mount).

    Args:
        cuda_version: The full CUDA version (e.g., "12.6.0")

    Returns:
        Path to the CUDA installation if found, None otherwise.
    """
    cuda_major_minor = ".".join(cuda_version.split(".")[:2])
    windows_cuda_path = Path(
        f"/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v{cuda_major_minor}"
    )

    if windows_cuda_path.exists():
        log_info(f"Found Windows CUDA installation at: {windows_cuda_path}")
        return windows_cuda_path

    log_info(f"No Windows CUDA installation found at: {windows_cuda_path}")
    return None


def set_windows_cuda_home(cuda_home_path: Path) -> None:
    """
    Set WINDOWS_CUDA_HOME environment variable in the user's shell config.

    Adds the export to ~/.bashrc and ~/.zshrc if they exist.
    Also sets it in the current environment.
    """
    export_line = f'export WINDOWS_CUDA_HOME="{cuda_home_path}"'

    # Set in current environment
    os.environ["WINDOWS_CUDA_HOME"] = str(cuda_home_path)
    log_info(f"Set WINDOWS_CUDA_HOME={cuda_home_path}")

    # Add to shell config files
    shell_configs = [
        Path.home() / ".bashrc",
        Path.home() / ".zshrc",
    ]

    for config_file in shell_configs:
        if not config_file.exists():
            continue

        # Check if already set
        content = config_file.read_text()
        if "WINDOWS_CUDA_HOME" in content:
            log_info(f"WINDOWS_CUDA_HOME already in {config_file}, updating...")
            # Remove old line(s) and add new one
            lines = [
                line for line in content.splitlines() if "WINDOWS_CUDA_HOME" not in line
            ]
            lines.append(export_line)
            config_file.write_text("\n".join(lines) + "\n")
        else:
            log_info(f"Adding WINDOWS_CUDA_HOME to {config_file}")
            with open(config_file, "a") as f:
                f.write(f"\n# Windows CUDA path for cross-compilation\n")
                f.write(f"{export_line}\n")


def download_and_extract_cuda(
    cuda_version: str, cuda_driver_version: str, install_dir: Path, env_type: str
) -> None:
    """Download and extract CUDA toolkit for Windows."""
    log_info("Setting up CUDA toolkit for Windows cross-compilation...")

    install_dir.mkdir(parents=True, exist_ok=True)

    cuda_installer = f"cuda_{cuda_version}_{cuda_driver_version}_windows.exe"
    cuda_installer_path = install_dir / cuda_installer
    cuda_url = (
        f"https://developer.download.nvidia.com/compute/cuda/{cuda_version}/"
        f"local_installers/{cuda_installer}"
    )

    # Download CUDA installer if not present
    if not cuda_installer_path.exists():
        log_info(f"Downloading CUDA {cuda_version} Windows installer...")
        run_command(["wget", cuda_url, "-O", str(cuda_installer_path)])
    else:
        log_info("CUDA installer already downloaded, skipping download...")

    # Install 7zip if needed
    install_7zip(env_type)

    # Extract CUDA toolkit
    extracted_dir = install_dir / "extracted"
    if not extracted_dir.exists():
        log_info("Extracting CUDA toolkit...")
        run_command(["7z", "x", str(cuda_installer_path), f"-o{extracted_dir}", "-y"])
    else:
        log_info("CUDA already extracted, skipping extraction...")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install Windows CUDA dependencies for cross-compilation. "
        "CUDA version is automatically detected from PyTorch installation."
    )
    parser.add_argument(
        "--install-dir",
        type=Path,
        default=Path(os.environ.get("INSTALL_DIR", Path.home() / "cuda-windows")),
        help="Installation directory (default: $HOME/cuda-windows)",
    )

    args = parser.parse_args()

    env_type = detect_environment()
    log_info(f"Detected environment: {env_type}")

    if env_type == "unknown":
        log_error("Unknown environment. This script supports Fedora/RHEL and WSL.")
        return 1

    # Install mingw
    try:
        if env_type == "fedora":
            install_mingw_fedora()
        elif env_type == "wsl":
            install_mingw_wsl()
    except subprocess.CalledProcessError as e:
        log_error(f"Failed to install mingw: {e}")
        return 1

    # Get CUDA version from PyTorch
    cuda_info = get_pytorch_cuda_version()
    if cuda_info is None:
        return 1

    cuda_version, cuda_driver_version = cuda_info

    # For WSL, check if CUDA is already installed on Windows
    if env_type == "wsl":
        windows_cuda_path = find_windows_cuda_install(cuda_version)
        if windows_cuda_path is not None:
            log_info("Using existing Windows CUDA installation.")
            set_windows_cuda_home(windows_cuda_path)
            log_info("")
            log_info("Installation complete!")
            return 0

        log_info("Will download CUDA toolkit instead...")

    # Download and extract CUDA
    try:
        download_and_extract_cuda(
            cuda_version,
            cuda_driver_version,
            args.install_dir,
            env_type,
        )

        cuda_home_path = args.install_dir / "extracted" / "cuda_cudart" / "cudart"
        set_windows_cuda_home(cuda_home_path)
    except subprocess.CalledProcessError as e:
        log_error(f"Failed to download/extract CUDA: {e}")
        return 1

    log_info("")
    log_info("Installation complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
