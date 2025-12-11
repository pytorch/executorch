import argparse
import ctypes
import logging
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter, Retry

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

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


#########################
# Cache directory helper
#########################

APP_NAMESPACE = ["executorch", "qnn"]


def _get_staging_dir(*parts: str) -> pathlib.Path:
    r"""
    Return a cross-platform staging directory for staging SDKs/libraries.

    - On Linux:
        ~/.cache/executorch/qnn/<parts...>
        (falls back to $HOME/.cache if $XDG_CACHE_HOME is unset)

    - On Windows (not supported yet, but as placeholder):
        %LOCALAPPDATA%\executorch\qnn\<parts...>
        (falls back to $HOME/AppData/Local if %LOCALAPPDATA% is unset)

    - Override:
        If QNN_STAGING_DIR is set in the environment, that path is used instead.

    Args:
        parts (str): Subdirectories to append under the root staging dir.

    Returns:
        pathlib.Path: Fully qualified staging path.
    """
    # Environment override wins
    base = os.environ.get("QNN_STAGING_DIR")
    if base:
        return pathlib.Path(base).joinpath(*parts)

    system = platform.system().lower()
    if system == "windows":
        # On Windows, prefer %LOCALAPPDATA%, fallback to ~/AppData/Local
        base = pathlib.Path(
            os.environ.get("LOCALAPPDATA", pathlib.Path.home() / "AppData" / "Local")
        )
    elif is_linux_x86():
        # On Linux/Unix, prefer $XDG_CACHE_HOME, fallback to ~/.cache
        base = pathlib.Path(
            os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")
        )
    else:
        raise ValueError(f"Unsupported platform: {system}")

    return base.joinpath(*APP_NAMESPACE, *parts)


def _atomic_download(url: str, dest: pathlib.Path):
    """
    Download URL into dest atomically:
      - Write to a temp file in the same dir
      - Move into place if successful
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Temp file in same dir (guarantees atomic rename)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        tmp_path = pathlib.Path(tmp.name)

    try:
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(dest)  # atomic rename
    except Exception:
        # Clean up partial file on failure
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


####################
# qnn sdk download management
####################


def _download_archive(url: str, archive_path: pathlib.Path) -> bool:
    """Streaming download with retry + resume support."""

    logger.debug("Archive will be saved to: %s", archive_path)

    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))

    # ------------------------------------------------------------
    # 1. Detect total file size (HEAD is broken on Qualcomm)
    # ------------------------------------------------------------
    try:
        # NOTE:
        # Qualcomm's download endpoint does not return accurate metadata on HEAD requests.
        # Many Qualcomm URLs first redirect to an HTML "wrapper" page (typically ~134 bytes),
        # and the HEAD request reflects *that wrapper* rather than the actual ZIP archive.
        #
        # Example:
        #   HEAD -> Content-Length: 134, Content-Type: text/html
        #   GET  -> Content-Length: 1354151797, Content-Type: application/zip
        #
        # Because Content-Length from HEAD is frequently incorrect, we fall back to issuing
        # a GET request with stream=True to obtain the real Content-Length without downloading
        # the full file. This ensures correct resume logic and size validation.
        r_head = session.get(url, stream=True)
        r_head.raise_for_status()

        if "content-length" not in r_head.headers:
            logger.error("Server did not return content-length!")
            return False

        total_size = int(r_head.headers["content-length"])
    except Exception as e:
        logger.exception("Failed to determine file size: %s", e)
        return False

    # ------------------------------------------------------------
    # 2. If partial file exists, resume
    # ------------------------------------------------------------
    downloaded = archive_path.stat().st_size if archive_path.exists() else 0
    if downloaded > total_size:
        logger.warning("Existing file is larger than expected. Removing.")
        archive_path.unlink()
        downloaded = 0

    logger.info("Resuming download from %d / %d bytes", downloaded, total_size)

    headers = {}
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"

    try:
        # resume GET
        with session.get(url, stream=True, headers=headers) as r:
            r.raise_for_status()

            chunk_size = 1024 * 1024  # 1MB
            mode = "ab" if downloaded > 0 else "wb"

            with open(archive_path, mode) as f:
                for chunk in r.iter_content(chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

    except Exception as e:
        logger.exception("Error during download: %s", e)
        return False

    # ------------------------------------------------------------
    # 3. Validate final size
    # ------------------------------------------------------------
    final_size = archive_path.stat().st_size
    if final_size != total_size:
        logger.error(
            "Download incomplete: expected %d, got %d",
            total_size,
            final_size,
        )
        return False

    logger.info("Download completed successfully!")
    return True


def _extract_archive(
    url: str, archive_path: pathlib.Path, content_dir: str, dst_folder: pathlib.Path
):
    """Extract archive based on type (zip or tar)."""
    if url.endswith(".zip"):
        logger.info("Extracting ZIP archive...")
        _extract_zip(archive_path, content_dir, dst_folder)
    elif url.endswith((".tar.gz", ".tgz")):
        logger.info("Extracting TAR archive...")
        _extract_tar(archive_path, content_dir, dst_folder)
    else:
        raise ValueError(f"Unsupported archive format: {url}")


def _verify_extraction(dst_folder: pathlib.Path):
    """Check if extraction succeeded and log contents."""
    logger.info("Verifying extraction to %s", dst_folder)
    if dst_folder.exists():
        logger.debug("SDK directory exists. Contents:")
        for item in dst_folder.iterdir():
            logger.debug("  %s", item.name)
    else:
        logger.error("SDK directory was not created!")


def _download_qnn_sdk(dst_folder=SDK_DIR) -> Optional[pathlib.Path]:
    """
    Download and extract the Qualcomm SDK into dst_folder.
    Only runs on Linux x86 platforms.
    """
    QNN_VERSION = "2.37.0.250724"
    logger.info("Downloading Qualcomm SDK...")
    QAIRT_URL = (
        f"https://softwarecenter.qualcomm.com/api/download/software/sdks/"
        f"Qualcomm_AI_Runtime_Community/All/{QNN_VERSION}/v{QNN_VERSION}.zip"
    )
    QAIRT_CONTENT_DIR = f"qairt/{QNN_VERSION}"
    if not is_linux_x86():
        logger.info("[QNN] Skipping Qualcomm SDK (only supported on Linux x86).")
        return None
    else:
        logger.info("[QNN] Downloading Qualcomm SDK for Linux x86")

    dst_folder.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = pathlib.Path(tmpdir) / pathlib.Path(QAIRT_URL).name
        if not _download_archive(QAIRT_URL, archive_path):
            return None

        _extract_archive(QAIRT_URL, archive_path, QAIRT_CONTENT_DIR, dst_folder)
        _verify_extraction(dst_folder)

    return dst_folder


def _extract_zip(archive_path, content_dir, target_dir):
    logger.debug("Extracting %s to %s", archive_path, target_dir)
    logger.debug("Looking for content in subdirectory: %s", content_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        files_to_extract = [f for f in zip_ref.namelist() if f.startswith(content_dir)]

        for file in files_to_extract:
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


####################
# libc management
####################

GLIBC_VERSION = "2.34"
GLIBC_REEXEC_GUARD = "QNN_GLIBC_REEXEC"
MINIMUM_LIBC_VERSION = GLIBC_VERSION


def _get_glibc_libdir() -> pathlib.Path:
    glibc_root = _get_staging_dir(f"glibc-{GLIBC_VERSION}")
    return glibc_root / "lib"


def _parse_version(v: str) -> tuple[int, int]:
    """Turn '2.34' → (2,34) so it can be compared."""
    parts = v.split(".")
    return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0


def _current_glibc_version() -> str:
    """Return system glibc version string (via ctypes)."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        func = libc.gnu_get_libc_version
        func.restype = ctypes.c_char_p
        return func().decode()
    except Exception as e:
        return f"error:{e}"


def _resolve_glibc_loader() -> pathlib.Path | None:
    """Return staged ld.so path if available."""
    for p in [
        _get_glibc_libdir() / f"ld-{GLIBC_VERSION}.so",
        _get_glibc_libdir() / "ld-linux-x86-64.so.2",
    ]:
        if p.exists():
            return p
    return None


def _stage_prebuilt_glibc():
    """Download + extract Fedora 35 glibc RPM into /tmp."""
    logger.info(">>> Staging prebuilt glibc-%s from Fedora 35 RPM", GLIBC_VERSION)
    _get_glibc_libdir().mkdir(parents=True, exist_ok=True)
    rpm_path = _get_staging_dir("glibc") / "glibc.rpm"
    work_dir = _get_staging_dir("glibc") / "extracted"
    rpm_url = (
        "https://archives.fedoraproject.org/pub/archive/fedora/linux/releases/35/"
        "Everything/x86_64/os/Packages/g/glibc-2.34-7.fc35.x86_64.rpm"
    )

    rpm_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[glibc] Downloading %s -> %s", rpm_url, rpm_path)
    try:
        urllib.request.urlretrieve(rpm_url, rpm_path)
    except Exception as e:
        logger.error("[glibc] Failed to download %s: %s", rpm_url, e)
        raise

    # Extract
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    subprocess.check_call(["bsdtar", "-C", str(work_dir), "-xf", str(rpm_path)])

    # Copy runtime libs
    staged = [
        "ld-linux-x86-64.so.2",
        "libc.so.6",
        "libdl.so.2",
        "libpthread.so.0",
        "librt.so.1",
        "libm.so.6",
        "libutil.so.1",
    ]
    for lib in staged:
        src = work_dir / "lib64" / lib
        if src.exists():
            shutil.copy2(src, _get_glibc_libdir() / lib)
            logger.info("[glibc] Staged %s", lib)
        else:
            logger.warning("[glibc] Missing %s in RPM", lib)


def ensure_glibc_minimum(min_version: str = GLIBC_VERSION):
    """
    Ensure process runs under glibc >= min_version.
    - If system glibc is new enough → skip.
    - Else → stage Fedora RPM and re-exec under staged loader.
    """
    current = _current_glibc_version()
    logger.info("[glibc] Current loaded glibc: %s", current)

    # If system glibc already sufficient → skip everything
    m = re.match(r"(\d+\.\d+)", current)
    if m and _parse_version(m.group(1)) >= _parse_version(min_version):
        logger.info("[glibc] System glibc >= %s, no staging needed.", min_version)
        return

    # Avoid infinite loop
    if os.environ.get(GLIBC_REEXEC_GUARD) == "1":
        logger.info("[glibc] Already re-exec'd once, continuing.")
        return

    # Stage prebuilt if not already staged
    if not (_get_glibc_libdir() / "libc.so.6").exists():
        _stage_prebuilt_glibc()

    loader = _resolve_glibc_loader()
    if not loader:
        logger.error("[glibc] Loader not found in %s", _get_glibc_libdir())
        return

    logger.info(
        "[glibc] Re-execing under loader %s with libdir %s", loader, _get_glibc_libdir()
    )
    os.environ[GLIBC_REEXEC_GUARD] = "1"
    os.execv(
        str(loader),
        [str(loader), "--library-path", str(_get_glibc_libdir()), sys.executable]
        + sys.argv,
    )


####################
# libc++ management
####################

LLVM_VERSION = "14.0.0"
LIBCXX_BASE_NAME = f"clang+llvm-{LLVM_VERSION}-x86_64-linux-gnu-ubuntu-18.04"
LLVM_URL = f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{LLVM_VERSION}/{LIBCXX_BASE_NAME}.tar.xz"
REQUIRED_LIBCXX_LIBS = [
    "libc++.so.1.0",
    "libc++abi.so.1.0",
    "libunwind.so.1",
]


def _stage_libcxx(target_dir: pathlib.Path):
    target_dir.mkdir(parents=True, exist_ok=True)

    if all((target_dir / libname).exists() for libname in REQUIRED_LIBCXX_LIBS):
        logger.info("[libcxx] Already staged at %s, skipping download", target_dir)
        return

    libcxx_stage = _get_staging_dir(f"libcxx-{LLVM_VERSION}")
    temp_tar = libcxx_stage / f"{LIBCXX_BASE_NAME}.tar.xz"
    temp_extract = libcxx_stage / LIBCXX_BASE_NAME

    if not temp_tar.exists():
        logger.info("[libcxx] Downloading %s", LLVM_URL)
        _atomic_download(LLVM_URL, temp_tar)

    # Sanity check before extracting
    if not temp_tar.exists() or temp_tar.stat().st_size == 0:
        raise FileNotFoundError(f"[libcxx] Tarball missing or empty: {temp_tar}")

    logger.info("[libcxx] Extracting %s", temp_tar)
    with tarfile.open(temp_tar, "r:xz") as tar:
        tar.extractall(temp_extract.parent)

    lib_src = temp_extract / "lib" / "x86_64-unknown-linux-gnu"
    for fname in REQUIRED_LIBCXX_LIBS:
        src_path = lib_src / fname
        if not src_path.exists():
            logger.warning(
                "[libcxx] %s not found in extracted LLVM src_path %s", fname, src_path
            )
            continue
        shutil.copy(src_path, target_dir / fname)

    logger.info("[libcxx] Staged libc++ to %s", target_dir)


REQUIRED_QNN_LIBS: List[str] = [
    "libQnnHtp.so",
]


def _ld_library_paths() -> List[pathlib.Path]:
    """Split LD_LIBRARY_PATH into ordered directories (skip empties)."""
    raw = os.environ.get("LD_LIBRARY_PATH", "")
    return [pathlib.Path(p) for p in raw.split(":") if p.strip()]


def _find_lib_in_ld_paths(
    libname: str, ld_dirs: Optional[List[pathlib.Path]] = None
) -> Optional[pathlib.Path]:
    """Return first matching path to `libname` in LD_LIBRARY_PATH, or None."""
    if ld_dirs is None:
        ld_dirs = _ld_library_paths()
    for d in ld_dirs:
        candidate = d / libname
        try:
            if candidate.exists():
                return candidate.resolve()
        except Exception:
            # Ignore unreadable / permission issues, keep looking.
            pass
    return None


def _check_libs_in_ld(
    libnames: List[str],
) -> Tuple[bool, Dict[str, Optional[pathlib.Path]]]:
    """
    Check if each lib in `libnames` exists in LD_LIBRARY_PATH directories.

    Returns:
        all_present: True iff every lib was found
        locations:   mapping lib -> path (or None if missing)
    """
    ld_dirs = _ld_library_paths()
    locations: Dict[str, Optional[pathlib.Path]] = {}
    for lib in libnames:
        locations[lib] = _find_lib_in_ld_paths(lib, ld_dirs)
    all_present = all(locations[lib] is not None for lib in libnames)
    return all_present, locations


# -----------------------
# Ensure QNN SDK library
# -----------------------
def _ensure_qnn_sdk_lib() -> bool:
    """
    Ensure libQnnHtp.so is available.
      - If found in LD_LIBRARY_PATH: do nothing, return True.
      - Otherwise: ensure packaged SDK is present, then load libQnnHtp.so from it.
    """
    all_present, locs = _check_libs_in_ld(REQUIRED_QNN_LIBS)
    if all_present:
        logger.info(
            "[QNN] libQnnHtp.so found in LD_LIBRARY_PATH; skipping SDK install."
        )
        for lib, p in locs.items():
            logger.info("      - %s: %s", lib, p)
        return True

    # Not found → use packaged SDK
    qnn_sdk_dir = SDK_DIR
    logger.info("[QNN] libQnnHtp.so not found in LD_LIBRARY_PATH.")
    if not qnn_sdk_dir.exists():
        logger.info("[QNN] SDK dir missing; downloading...")
        _download_qnn_sdk()
    else:
        logger.info("[QNN] Using existing SDK at %s", qnn_sdk_dir)

    os.environ["QNN_SDK_ROOT"] = str(qnn_sdk_dir)

    qnn_lib = qnn_sdk_dir / "lib" / "x86_64-linux-clang" / "libQnnHtp.so"
    logger.info("[QNN] Loading %s", qnn_lib)
    lib_loaded = False
    try:
        ctypes.CDLL(str(qnn_lib), mode=ctypes.RTLD_GLOBAL)
        logger.info("[QNN] Loaded libQnnHtp.so from packaged SDK.")
        lib_loaded = True
    except OSError as e:
        logger.error("[QNN][ERROR] Failed to load %s: %s", qnn_lib, e)
    return lib_loaded


def _load_libcxx_libs(lib_path):
    logger.debug("running _load_libcxx_libs")
    candidates = list(lib_path.glob("*.so*"))
    priority = ["libc++abi", "libc++"]
    sorted_candidates = [
        f for name in priority for f in candidates if f.name.startswith(name)
    ]
    sorted_candidates += [f for f in candidates if f not in sorted_candidates]
    logger.debug("sorted_candidates: %s", sorted_candidates)
    for sofile in sorted_candidates:
        try:
            ctypes.CDLL(str(sofile), mode=ctypes.RTLD_GLOBAL)
            logger.info("Loaded %s", sofile.name)
        except OSError as e:
            logger.warning("[WARN] Failed to load %s: %s", sofile.name, e)


# ---------------------
# Ensure libc++ family
# ---------------------
def _ensure_libcxx_stack() -> bool:
    """
    Ensure libc++ stack is available.
      - If all required libc++ libs are found in LD_LIBRARY_PATH: do nothing.
      - Otherwise: stage and load the packaged libc++ bundle.
    """
    all_present, locs = _check_libs_in_ld(REQUIRED_LIBCXX_LIBS)
    if all_present:
        logger.info(
            "[libcxx] All libc++ libs present in LD_LIBRARY_PATH; skipping staging."
        )
        for lib, p in locs.items():
            logger.info("         - %s: %s", lib, p)
        return True

    logger.info(
        "[libcxx] Some libc++ libs missing in LD_LIBRARY_PATH; staging packaged libc++..."
    )
    lib_loaded = False
    try:
        libcxx_dir = PKG_ROOT / "sdk" / f"libcxx-{LLVM_VERSION}"
        _stage_libcxx(libcxx_dir)
        _load_libcxx_libs(libcxx_dir)
        logger.info("[libcxx] Staged and loaded libc++ from %s", libcxx_dir)
        lib_loaded = True
    except Exception as e:
        logger.exception("[libcxx][ERROR] Failed to stage/load libc++: %s", e)
    return lib_loaded


# ---------------
# Public entrypoint
# ---------------
def install_qnn_sdk() -> bool:
    """
    Initialize Qualcomm backend with separated logic:

    QNN SDK:
      - If libQnnHtp.so exists in LD_LIBRARY_PATH: do nothing.
      - Else: ensure packaged SDK, load libQnnHtp.so.

    libc++ stack:
      - If required libc++ libs exist in LD_LIBRARY_PATH: do nothing.
      - Else: stage and load packaged libc++.

    Returns:
        True if both steps succeeded (or were already satisfied), else False.
    """
    logger.info("[QNN] Starting SDK installation")

    # Make sure we’re running under >= 2.34
    ensure_glibc_minimum(GLIBC_VERSION)

    # libc++ and QNN SDK setup
    return _ensure_libcxx_stack() and _ensure_qnn_sdk_lib()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Helper utility for Qualcomm SDK staging."
    )
    parser.add_argument(
        "--dst-folder",
        type=pathlib.Path,
        default=SDK_DIR,
        help="Destination directory for the Qualcomm SDK.",
    )
    parser.add_argument(
        "--print-sdk-path",
        action="store_true",
        help="Print the resolved Qualcomm SDK path to stdout.",
    )
    parser.add_argument(
        "--install-sdk",
        action="store_true",
        help="Ensure the SDK and runtime libraries are staged and loaded.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    sdk_path: Optional[pathlib.Path]
    if args.install_sdk:
        if not install_qnn_sdk():
            return 1
        sdk_path = pathlib.Path(os.environ.get("QNN_SDK_ROOT", args.dst_folder))
    else:
        sdk_path = _download_qnn_sdk(dst_folder=args.dst_folder)
        if sdk_path is None:
            return 1

    if args.print_sdk_path and sdk_path is not None:
        print(sdk_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
