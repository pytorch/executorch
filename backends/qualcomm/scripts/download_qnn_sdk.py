# Add these imports for additional logging
import ctypes
import os
import pathlib
import platform
import shutil
import tarfile
import tempfile
import urllib.request
from typing import Optional

SDK_DIR = pathlib.Path(__file__).parent.parent / "sdk" / "qnn"
PKG_ROOT = pathlib.Path(__file__).parent


def is_linux_x86() -> bool:
    """
    Check if the current platform is Linux x86_64.

    Returns:
        bool: True if the system is Linux x86_64, False otherwise.
    """
    system = platform.system().lower()
    machine = platform.machine().lower()

    return system == "linux" and machine in ("x86_64", "amd64", "i386", "i686")


def _download_qnn_sdk() -> Optional[pathlib.Path]:
    """
    Download and extract the Qualcomm SDK into SDK_DIR.

    Notes:
        - Only runs on Linux x86 platforms. Skips otherwise.
    """
    print("Downloading Qualcomm SDK...")
    qairt_url = (
        "https://softwarecenter.qualcomm.com/api/download/software/sdks/"
        "Qualcomm_AI_Runtime_Community/All/2.34.0.250424/v2.34.0.250424.zip"
    )
    qairt_content_dir = "qairt/2.34.0.250424"

    if not is_linux_x86():
        print("Skipping Qualcomm SDK (only supported on Linux x86).")
        return None

    SDK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"SDK_DIR is {SDK_DIR}, exists: {SDK_DIR.exists()}")
    print(f"Current working directory: {os.getcwd()}")

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = pathlib.Path(tmpdir) / pathlib.Path(qairt_url).name
        print(f"Temporary directory: {tmpdir}")
        print(f"Archive will be saved to: {archive_path}")

        print(f"Downloading Qualcomm SDK from {qairt_url}...")
        try:
            # Use urlretrieve with a reporthook to show progress
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = downloaded / total_size * 100
                print(
                    f"Downloaded: {downloaded}/{total_size} bytes ({percent:.2f}%)",
                    end="\r",
                )

            urllib.request.urlretrieve(qairt_url, archive_path, report_progress)
            print("\nDownload completed!")

            # Check if file was downloaded successfully
            if archive_path.exists():
                file_size = archive_path.stat().st_size
                print(f"Downloaded file size: {file_size} bytes")
                if file_size == 0:
                    print("WARNING: Downloaded file is empty!")
            else:
                print("ERROR: File was not downloaded!")
                return None

        except Exception as e:
            print(f"Error during download: {e}")
            return None

        # Check extraction method
        if qairt_url.endswith(".zip"):
            print("Extracting ZIP archive...")
            _extract_zip(archive_path, qairt_content_dir, SDK_DIR)
        elif qairt_url.endswith((".tar.gz", ".tgz")):
            print("Extracting TAR archive...")
            _extract_tar(archive_path, qairt_content_dir, SDK_DIR)
        else:
            raise ValueError(f"Unsupported archive format: {qairt_url}")

        # Verify extraction
        print(f"Verifying extraction to {SDK_DIR}")
        if SDK_DIR.exists():
            print(f"SDK directory exists. Contents:")
            for item in SDK_DIR.iterdir():
                print(f"  {item.name}")
        else:
            print("ERROR: SDK directory was not created!")

        print(f"Qualcomm SDK extracted to {SDK_DIR}")

    return SDK_DIR


# You might also want to add detailed logging to your extraction functions
def _extract_zip(archive_path, content_dir, target_dir):
    print(f"Extracting {archive_path} to {target_dir}")
    print(f"Looking for content in subdirectory: {content_dir}")

    # Add your zip extraction code here, with additional logging
    # For example:
    import zipfile

    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        # List all files in the archive
        print("Files in archive:")
        for file in zip_ref.namelist():
            print(f"  {file}")

        # Extract only the specific content directory
        for file in zip_ref.namelist():
            if file.startswith(content_dir):
                # Extract with path relative to content_dir
                relative_path = os.path.relpath(file, content_dir)
                if relative_path == ".":
                    continue  # Skip the directory entry itself
                target_path = target_dir / relative_path
                if file.endswith("/"):
                    # Create directory
                    target_path.mkdir(parents=True, exist_ok=True)
                else:
                    # Extract file
                    with zip_ref.open(file) as source, open(
                        target_path, "wb"
                    ) as target:
                        shutil.copyfileobj(source, target)
                    print(f"Extracted: {relative_path}")


LLVM_VERSION = "14.0.0"
LIBCXX_LIB_DIR = (
    PKG_ROOT / "executorch" / "backends" / "qualcomm" / "sdk" / f"libcxx-{LLVM_VERSION}"
)
LIBCXX_BASE_NAME = f"clang+llvm-{LLVM_VERSION}-x86_64-linux-gnu-ubuntu-20.04"


def stage_libcxx(target_dir: pathlib.Path):
    """
    Download (if needed) and stage libc++ shared libraries into the wheel package.
    - target_dir: destination folder in the wheel, e.g.
      executorch/backends/qualcomm/sdk/libcxx-14.0.0
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if already staged
    existing_files = list(target_dir.glob("*"))
    if existing_files:
        print(f"[libcxx] Already staged at {target_dir}, skipping")
        return

    # URL to a tarball with prebuilt libc++ shared libraries
    LLVM_URL = f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{LLVM_VERSION}/{LIBCXX_BASE_NAME}.tar.xz"
    temp_tar = pathlib.Path("/tmp") / f"{LIBCXX_BASE_NAME}.tar.xz"
    temp_extract = pathlib.Path("/tmp") / f"{LIBCXX_BASE_NAME}"

    # Download if not already exists
    if not temp_tar.exists():
        print(f"[libcxx] Downloading {LLVM_URL}")
        urllib.request.urlretrieve(LLVM_URL, temp_tar)

    # Extract
    print(f"[libcxx] Extracting {temp_tar}")
    with tarfile.open(temp_tar, "r:xz") as tar:
        tar.extractall(temp_extract.parent)

    # Copy only the required .so files
    lib_src = temp_extract / "lib"
    required_files = [
        "libc++.so.1.0",
        "libc++abi.so.1.0",
        "libunwind.so.1",
        "libm.so.6",
        "libpython3.10.so.1.0",
    ]
    for fname in required_files:
        src_path = lib_src / fname
        if not src_path.exists():
            raise FileNotFoundError(f"{fname} not found in extracted LLVM")
        shutil.copy(src_path, target_dir / fname)

    # Create symlinks
    os.symlink("libc++.so.1.0", target_dir / "libc++.so.1")
    os.symlink("libc++.so.1", target_dir / "libc++.so")
    os.symlink("libc++abi.so.1.0", target_dir / "libc++abi.so.1")
    os.symlink("libc++abi.so.1", target_dir / "libc++abi.so")

    print(f"[libcxx] Staged libc++ to {target_dir}")


def _load_libcxx_libs(lib_path):
    """
    Load libc++ shared libraries from the given directory.

    Ensures libc++abi is loaded first, then libc++, then any other .so files.
    """
    candidates = list(lib_path.glob("*.so*"))

    priority = ["libc++abi", "libc++"]
    sorted_candidates = []

    for name in priority:
        for f in candidates:
            if f.name.startswith(name):
                sorted_candidates.append(f)

    for f in candidates:
        if f not in sorted_candidates:
            sorted_candidates.append(f)

    for sofile in sorted_candidates:
        try:
            ctypes.CDLL(str(sofile), mode=ctypes.RTLD_GLOBAL)
            print(f"Loaded {sofile.name}")
        except OSError as e:
            print(f"[WARN] Failed to load {sofile.name}: {e}")
