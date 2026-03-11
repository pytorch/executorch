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
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

PKG_ROOT = pathlib.Path(__file__).parent.parent


def _progress(msg: str) -> None:
    """Print a progress line with carriage return (no newline). Not suited for logging."""
    print(msg, end="", flush=True)


def _progress_newline() -> None:
    """End a progress line."""
    print(flush=True)


##########################
# Version from qnn_config
##########################


def _read_qnn_config() -> Dict[str, str]:
    """Parse qnn_config.sh to extract QNN_VERSION and QNN_ZIP_URL."""
    config_path = pathlib.Path(__file__).parent / "qnn_config.sh"
    config: Dict[str, str] = {}
    if not config_path.exists():
        return config
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            # Strip quotes and resolve bash-style ${VAR} references
            val = val.strip('"')
            config[key.strip()] = val
    # Resolve ${QNN_VERSION} in QNN_ZIP_URL
    if "QNN_ZIP_URL" in config and "QNN_VERSION" in config:
        config["QNN_ZIP_URL"] = config["QNN_ZIP_URL"].replace(
            "${QNN_VERSION}", config["QNN_VERSION"]
        )
    return config


_QNN_CONFIG = _read_qnn_config()
QNN_VERSION = _QNN_CONFIG.get("QNN_VERSION", "2.37.0.250724")
QNN_ZIP_URL = _QNN_CONFIG.get(
    "QNN_ZIP_URL",
    f"https://softwarecenter.qualcomm.com/api/download/software/sdks/"
    f"Qualcomm_AI_Runtime_Community/All/{QNN_VERSION}/v{QNN_VERSION}.zip",
)


def _get_sdk_dir() -> pathlib.Path:
    """Get the versioned SDK cache directory (e.g. ~/.cache/executorch/qnn/sdk-2.37.0.250724/)."""
    try:
        return _get_staging_dir(f"sdk-{QNN_VERSION}")
    except ValueError:
        return PKG_ROOT / "sdk" / "qnn"


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


def _atomic_download(url: str, dest: pathlib.Path, label: str = ""):
    """
    Download URL into dest atomically:
      - Write to a temp file in the same dir
      - Move into place if successful
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Temp file in same dir (guarantees atomic rename)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        tmp_path = pathlib.Path(tmp.name)

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded * 100 / total_size, 100)
            dl_mb = downloaded // (1024 * 1024)
            total_mb = total_size // (1024 * 1024)
            prefix = f"[QNN] Downloading {label}: " if label else "[QNN] Downloading: "
            _progress(f"\r{prefix}{dl_mb}/{total_mb} MB ({pct:.0f}%)")

    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=_reporthook)
        if label:
            _progress_newline()
        tmp_path.replace(dest)  # atomic rename
    except Exception:
        if label:
            _progress_newline()
        # Clean up partial file on failure
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


####################
# qnn sdk download management
####################


def _stream_to_file(
    session: requests.Session,
    url: str,
    archive_path: pathlib.Path,
    attempt: int,
    max_retries: int,
) -> bool:
    """Single download attempt with resume support. Returns True on success."""
    downloaded = archive_path.stat().st_size if archive_path.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded > 0 else {}

    with session.get(url, stream=True, headers=headers) as r:
        if r.status_code == 200 and downloaded > 0:
            downloaded = 0  # Server doesn't support Range — restart
        r.raise_for_status()

        total = downloaded + int(r.headers.get("content-length", 0))
        mode = "ab" if downloaded > 0 else "wb"

        if attempt > 1:
            dl_mb = downloaded // (1024 * 1024)
            total_mb = total // (1024 * 1024)
            logger.info(
                f"[QNN] Resuming download from {dl_mb}/{total_mb} MB "
                f"(attempt {attempt}/{max_retries})..."
            )

        with open(archive_path, mode) as f:
            for chunk in r.iter_content(1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 / total
                    dl_mb = downloaded // (1024 * 1024)
                    total_mb = total // (1024 * 1024)
                    _progress(
                        f"\r[QNN] Downloading: {dl_mb}/{total_mb} MB ({pct:.0f}%)"
                    )
        if total:
            _progress_newline()

    logger.info("[QNN] Download complete.")
    return True


def _download_archive(
    url: str, archive_path: pathlib.Path, max_retries: int = 3
) -> bool:
    """Streaming download with retry + resume on mid-stream failures."""
    logger.debug("Archive will be saved to: %s", archive_path)

    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))

    for attempt in range(1, max_retries + 1):
        try:
            if _stream_to_file(session, url, archive_path, attempt, max_retries):
                break
        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as e:
            _progress_newline()
            if attempt < max_retries:
                logger.warning(
                    f"[QNN] Download interrupted: {type(e).__name__}. "
                    f"Retrying ({attempt}/{max_retries})..."
                )
            else:
                logger.error(f"[QNN] Download failed after {max_retries} attempts: {e}")
                return False
        except Exception as e:
            _progress_newline()
            logger.error(f"[QNN] Download error: {e}")
            return False

    if not archive_path.exists() or archive_path.stat().st_size == 0:
        logger.error("[QNN] Downloaded file is empty or missing!")
        return False

    return True


def _extract_archive(
    url: str, archive_path: pathlib.Path, content_dir: str, dst_folder: pathlib.Path
):
    """Extract archive based on type (zip or tar)."""
    logger.info("[QNN] Extracting SDK...")
    if url.endswith(".zip"):
        _extract_zip(archive_path, content_dir, dst_folder)
    elif url.endswith((".tar.gz", ".tgz")):
        _extract_tar(archive_path, content_dir, dst_folder)
    else:
        raise ValueError(f"Unsupported archive format: {url}")


def _verify_extraction(dst_folder: pathlib.Path):
    """Check if extraction succeeded and log contents."""
    if dst_folder.exists():
        logger.debug("SDK directory exists. Contents:")
        for item in dst_folder.iterdir():
            logger.debug("  %s", item.name)
    else:
        logger.error("[QNN] Error: SDK directory was not created!")


def _download_qnn_sdk(
    dst_folder: Optional[pathlib.Path] = None,
) -> Optional[pathlib.Path]:
    """
    Download and extract the Qualcomm SDK into dst_folder.
    Only runs on Linux x86 platforms.
    """
    if dst_folder is None:
        dst_folder = _get_sdk_dir()

    qairt_content_dir = f"qairt/{QNN_VERSION}"
    if not is_linux_x86():
        logger.info("[QNN] Skipping Qualcomm SDK (only supported on Linux x86).")
        return None

    logger.info(f"[QNN] Downloading Qualcomm AI Runtime SDK v{QNN_VERSION}...")

    dst_folder.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = pathlib.Path(tmpdir) / pathlib.Path(QNN_ZIP_URL).name
        if not _download_archive(QNN_ZIP_URL, archive_path):
            return None

        _extract_archive(QNN_ZIP_URL, archive_path, qairt_content_dir, dst_folder)
        _verify_extraction(dst_folder)

    logger.info(f"[QNN] QNN SDK v{QNN_VERSION} ready at {dst_folder}")
    return dst_folder


def _extract_zip(archive_path, content_dir, target_dir):
    logger.debug("Extracting %s to %s", archive_path, target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        files_to_extract = [f for f in zip_ref.namelist() if f.startswith(content_dir)]
        total = len(files_to_extract)

        for i, file in enumerate(files_to_extract):
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

            if total > 0:
                pct = (i + 1) * 100 // total
                _progress(f"\r[QNN] Extracting SDK: {pct}%")
        if total > 0:
            _progress_newline()


def _extract_tar(archive_path: pathlib.Path, prefix: str, target_dir: pathlib.Path):
    with tarfile.open(archive_path, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.name.startswith(prefix + "/")]
        total = len(members)

        for i, m in enumerate(members):
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

            if total > 0:
                pct = (i + 1) * 100 // total
                _progress(f"\r[QNN] Extracting SDK: {pct}%")
        if total > 0:
            _progress_newline()


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
    """Download + extract Fedora 35 glibc RPM."""
    logger.info(f"[QNN] Staging glibc {GLIBC_VERSION}...")
    _get_glibc_libdir().mkdir(parents=True, exist_ok=True)
    rpm_path = _get_staging_dir("glibc") / "glibc.rpm"
    work_dir = _get_staging_dir("glibc") / "extracted"
    rpm_url = (
        "https://archives.fedoraproject.org/pub/archive/fedora/linux/releases/35/"
        "Everything/x86_64/os/Packages/g/glibc-2.34-7.fc35.x86_64.rpm"
    )

    rpm_path.parent.mkdir(parents=True, exist_ok=True)

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded * 100 / total_size, 100)
            dl_mb = downloaded // (1024 * 1024)
            total_mb = total_size // (1024 * 1024)
            _progress(f"\r[QNN] Downloading glibc: {dl_mb}/{total_mb} MB ({pct:.0f}%)")

    try:
        urllib.request.urlretrieve(rpm_url, rpm_path, reporthook=_reporthook)
        _progress_newline()
    except Exception as e:
        logger.error(f"\n[QNN] Failed to download glibc: {e}")
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
            logger.debug("[glibc] Staged %s", lib)
        else:
            logger.warning("[glibc] Missing %s in RPM", lib)


def ensure_glibc_minimum(min_version: str = GLIBC_VERSION):
    """
    Ensure process runs under glibc >= min_version.
    - If system glibc is new enough → skip.
    - Else → stage Fedora RPM and re-exec under staged loader.
    """
    current = _current_glibc_version()
    logger.debug("[glibc] Current loaded glibc: %s", current)

    # If system glibc already sufficient → skip everything
    m = re.match(r"(\d+\.\d+)", current)
    if m and _parse_version(m.group(1)) >= _parse_version(min_version):
        return

    # Avoid infinite loop
    if os.environ.get(GLIBC_REEXEC_GUARD) == "1":
        logger.debug("[glibc] Already re-exec'd once, continuing.")
        return

    # Stage prebuilt if not already staged
    if not (_get_glibc_libdir() / "libc.so.6").exists():
        _stage_prebuilt_glibc()

    loader = _resolve_glibc_loader()
    if not loader:
        logger.error(f"[QNN] Warning: glibc loader not found in {_get_glibc_libdir()}")
        return

    logger.error(
        f"[QNN] System glibc ({current}) is older than required ({min_version}).\n"
        "[QNN] Re-launching Python under a staged glibc loader.\n"
        "[QNN] To avoid this, set QNN_SDK_ROOT and LD_LIBRARY_PATH manually."
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
        logger.debug("[libcxx] Already staged at %s, skipping download", target_dir)
        return

    libcxx_stage = _get_staging_dir(f"libcxx-{LLVM_VERSION}")
    temp_tar = libcxx_stage / f"{LIBCXX_BASE_NAME}.tar.xz"
    temp_extract = libcxx_stage / LIBCXX_BASE_NAME

    if not temp_tar.exists():
        logger.info(f"[QNN] Downloading libc++ (LLVM {LLVM_VERSION})...")
        _atomic_download(LLVM_URL, temp_tar, label="libc++")

    # Sanity check before extracting
    if not temp_tar.exists() or temp_tar.stat().st_size == 0:
        raise FileNotFoundError(f"[libcxx] Tarball missing or empty: {temp_tar}")

    logger.info("[QNN] Extracting libc++...")
    with tarfile.open(temp_tar, "r:xz") as tar:
        members = tar.getmembers()
        total = len(members)
        for i, member in enumerate(members):
            tar.extract(member, temp_extract.parent)
            if total > 0:
                pct = (i + 1) * 100 // total
                _progress(f"\r[QNN] Extracting libc++: {pct}%")
        if total > 0:
            _progress_newline()

    lib_src = temp_extract / "lib" / "x86_64-unknown-linux-gnu"
    for fname in REQUIRED_LIBCXX_LIBS:
        src_path = lib_src / fname
        if not src_path.exists():
            logger.warning(
                "[libcxx] %s not found in extracted LLVM src_path %s", fname, src_path
            )
            continue
        shutil.copy(src_path, target_dir / fname)

    logger.debug("[libcxx] Staged libc++ to %s", target_dir)


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
        logger.info("[QNN] Using QNN SDK libs from LD_LIBRARY_PATH")
        return True

    # Not found → use cached/packaged SDK
    # Directory name includes version (sdk-X.Y.Z), so a version bump
    # in qnn_config.sh naturally creates a new directory.
    qnn_sdk_dir = _get_sdk_dir()
    if not qnn_sdk_dir.exists():
        _download_qnn_sdk(qnn_sdk_dir)
    else:
        logger.info(f"[QNN] Using cached QNN SDK v{QNN_VERSION} at {qnn_sdk_dir}")

    os.environ["QNN_SDK_ROOT"] = str(qnn_sdk_dir)

    sdk_lib_dir = str(qnn_sdk_dir / "lib" / "x86_64-linux-clang")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if sdk_lib_dir not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = (
            f"{sdk_lib_dir}:{ld_path}" if ld_path else sdk_lib_dir
        )

    qnn_lib = qnn_sdk_dir / "lib" / "x86_64-linux-clang" / "libQnnHtp.so"
    lib_loaded = False
    try:
        ctypes.CDLL(str(qnn_lib), mode=ctypes.RTLD_GLOBAL)
        lib_loaded = True
    except OSError as e:
        logger.error(f"[QNN] Failed to load {qnn_lib}: {e}")
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
        logger.debug(
            "[libcxx] All libc++ libs present in LD_LIBRARY_PATH; skipping staging."
        )
        return True

    lib_loaded = False
    try:
        libcxx_dir = _get_staging_dir(f"libcxx-{LLVM_VERSION}")
        _stage_libcxx(libcxx_dir)
        _load_libcxx_libs(libcxx_dir)
        lib_loaded = True
    except Exception as e:
        logger.error(f"[QNN] Failed to stage/load libc++: {e}")
        logger.exception("[libcxx][ERROR] Failed to stage/load libc++: %s", e)
    return lib_loaded


# ---------------
# Public entrypoint
# ---------------
def install_qnn_sdk() -> bool:
    """
    Initialize Qualcomm backend:

    1. Ensure glibc >= 2.34 (may re-exec under staged loader)
    2. Ensure libc++ is available (download + stage if needed)
    3. Ensure QNN SDK is available (download if needed, detect version upgrades)
    4. Set QNN_SDK_ROOT and LD_LIBRARY_PATH

    Returns:
        True if all steps succeeded (or were already satisfied), else False.
    """
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
        default=None,
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

    dst = args.dst_folder if args.dst_folder else _get_sdk_dir()

    sdk_path: Optional[pathlib.Path]
    if args.install_sdk:
        if not install_qnn_sdk():
            return 1
        sdk_path = pathlib.Path(os.environ.get("QNN_SDK_ROOT", dst))
    else:
        sdk_path = _download_qnn_sdk(dst_folder=dst)
        if sdk_path is None:
            return 1

    if args.print_sdk_path and sdk_path is not None:
        print(sdk_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
