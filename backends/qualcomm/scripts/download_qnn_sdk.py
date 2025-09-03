# Add these imports for additional logging
import ctypes
import os
import pathlib
import platform
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from typing import Optional

from tqdm import tqdm

PKG_ROOT = pathlib.Path(__file__).parent.parent
SDK_DIR = PKG_ROOT / "sdk" / "qnn"


def is_linux_x86() -> bool:
    """
    Check if the current platform is Linux x86_64.

    Returns:
        bool: True if the system is Linux x86_64, False otherwise.
    """
    return platform.system().lower() == "linux" and platform.machine().lower() in (
        "x86_64",
        "amd64",
        "i386",
        "i686",
    )


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

            def make_report_progress():
                last_reported = 0

                def report_progress(block_num, block_size, total_size):
                    nonlocal last_reported
                    downloaded = block_num * block_size
                    percent = downloaded / total_size * 100
                    if percent - last_reported >= 20 or percent >= 100:
                        print(
                            f"Downloaded: {downloaded}/{total_size} bytes ({percent:.2f}%)"
                        )
                        last_reported = percent

                return report_progress

            urllib.request.urlretrieve(qairt_url, archive_path, make_report_progress())
            print("Download completed!")

            if archive_path.exists() and archive_path.stat().st_size == 0:
                print("WARNING: Downloaded file is empty!")
            elif not archive_path.exists():
                print("ERROR: File was not downloaded!")
                return None

        except Exception as e:
            print(f"Error during download: {e}")
            return None

        if qairt_url.endswith(".zip"):
            print("Extracting ZIP archive...")
            _extract_zip(archive_path, qairt_content_dir, SDK_DIR)
        elif qairt_url.endswith((".tar.gz", ".tgz")):
            print("Extracting TAR archive...")
            _extract_tar(archive_path, qairt_content_dir, SDK_DIR)
        else:
            raise ValueError(f"Unsupported archive format: {qairt_url}")

        print(f"Verifying extraction to {SDK_DIR}")
        if SDK_DIR.exists():
            print(f"SDK directory exists. Contents:")
            for item in SDK_DIR.iterdir():
                print(f"  {item.name}")
        else:
            print("ERROR: SDK directory was not created!")

        print(f"Qualcomm SDK extracted to {SDK_DIR}")

    return SDK_DIR


def _extract_zip(archive_path, content_dir, target_dir):
    print(f"Extracting {archive_path} to {target_dir}")
    print(f"Looking for content in subdirectory: {content_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        files_to_extract = [f for f in zip_ref.namelist() if f.startswith(content_dir)]

        for file in tqdm(files_to_extract, desc="Extracting files"):
            relative_path = pathlib.Path(file).relative_to(content_dir)
            if relative_path == pathlib.Path("."):
                continue

            out_path = target_dir / relative_path
            if file.endswith("/"):
                out_path.mkdir(parents=True, exist_ok=True)
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with zip_ref.open(file) as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)


def _extract_tar(archive_path: pathlib.Path, prefix: str, target_dir: pathlib.Path):
    with tarfile.open(archive_path, "r:gz") as tf:
        for m in tf.getmembers():
            if not m.name.startswith(prefix + "/"):
                continue
            relpath = pathlib.Path(m.name).relative_to(prefix)
            if not relpath.parts or relpath.parts[0] == "..":
                continue

            out_path = target_dir / relpath
            if m.isdir():
                out_path.mkdir(parents=True, exist_ok=True)
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(m)
                if src is None:
                    continue
                with src, open(out_path, "wb") as dst:
                    dst.write(src.read())


LLVM_VERSION = "14.0.0"
LIBCXX_BASE_NAME = f"clang+llvm-{LLVM_VERSION}-x86_64-linux-gnu-ubuntu-18.04"
LLVM_URL = f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{LLVM_VERSION}/{LIBCXX_BASE_NAME}.tar.xz"
REQUIRED_LIBS = [
    "libc++.so.1.0",
    "libc++abi.so.1.0",
    "libunwind.so.1",
    "libm.so.6",
    "libpython3.10.so.1.0",
]


def _stage_libcxx(target_dir: pathlib.Path):
    target_dir.mkdir(parents=True, exist_ok=True)

    if all((target_dir / libname).exists() for libname in REQUIRED_LIBS):
        print(f"[libcxx] Already staged at {target_dir}, skipping download")
        return

    temp_tar = pathlib.Path("/tmp") / f"{LIBCXX_BASE_NAME}.tar.xz"
    temp_extract = pathlib.Path("/tmp") / LIBCXX_BASE_NAME

    if not temp_tar.exists():
        print(f"[libcxx] Downloading {LLVM_URL}")
        urllib.request.urlretrieve(LLVM_URL, temp_tar)

    print(f"[libcxx] Extracting {temp_tar}")
    with tarfile.open(temp_tar, "r:xz") as tar:
        tar.extractall(temp_extract.parent)

    lib_src = temp_extract / "lib"
    for fname in REQUIRED_LIBS:
        src_path = lib_src / fname
        if not src_path.exists():
            print(f"[libcxx] Warning: {fname} not found in extracted LLVM")
            continue
        shutil.copy(src_path, target_dir / fname)

    libcxx = target_dir / "libc++.so.1.0"
    libcxx_abi = target_dir / "libc++abi.so.1.0"
    if libcxx.exists():
        os.symlink("libc++.so.1.0", target_dir / "libc++.so.1")
        os.symlink("libc++.so.1", target_dir / "libc++.so")
    if libcxx_abi.exists():
        os.symlink("libc++abi.so.1.0", target_dir / "libc++abi.so.1")
        os.symlink("libc++abi.so.1", target_dir / "libc++abi.so")

    print(f"[libcxx] Staged libc++ to {target_dir}")


def _load_libcxx_libs(lib_path):
    candidates = list(lib_path.glob("*.so*"))
    priority = ["libc++abi", "libc++"]
    sorted_candidates = [
        f for name in priority for f in candidates if f.name.startswith(name)
    ]
    sorted_candidates += [f for f in candidates if f not in sorted_candidates]

    for sofile in sorted_candidates:
        try:
            ctypes.CDLL(str(sofile), mode=ctypes.RTLD_GLOBAL)
            print(f"Loaded {sofile.name}")
        except OSError as e:
            print(f"[WARN] Failed to load {sofile.name}: {e}")


def install_qnn_sdk(force_download: bool = True) -> bool:
    print(f"[INIT] SDK_DIR: {SDK_DIR}")
    if not SDK_DIR.exists():
        print("[INIT] Qualcomm SDK not found. Downloading...")
        _download_qnn_sdk()

    os.environ["QNN_SDK_ROOT"] = str(SDK_DIR)

    qnn_lib = SDK_DIR / "lib" / "x86_64-linux-clang" / "libQnnHtp.so"
    print(f"[INIT] qnn_lib: {qnn_lib}")
    try:
        ctypes.CDLL(str(qnn_lib), mode=ctypes.RTLD_GLOBAL)
        print(f"[INIT] Loaded QNN library from {qnn_lib}")
    except OSError as e:
        print(f"[ERROR] Failed to load QNN library at {qnn_lib}: {e}")
        return False

    try:
        libcxx_dir = PKG_ROOT / "sdk" / f"libcxx-{LLVM_VERSION}"
        _stage_libcxx(libcxx_dir)
        _load_libcxx_libs(libcxx_dir)
        print(f"[INIT] Loaded libc++ from {libcxx_dir}")
    except Exception as e:
        print(f"[libcxx] Warning: failed to stage/load libc++: {e}")

    return True
