import pathlib
import platform
import sys
import tarfile
import tempfile
import urllib.request
import zipfile


def is_linux_x86() -> bool:
    """
    Check if the current system is Linux running on an x86 architecture.

    Returns:
        bool: True if the system is Linux and the architecture is one of
              x86_64, i386, or i686. False otherwise.
    """
    return sys.platform.startswith("linux") and platform.machine() in {
        "x86_64",
        "i386",
        "i686",
    }


SDK_DIR = pathlib.Path(__file__).parent / "sdk"


def _download_qnn_sdk() -> pathlib.Path:
    """
    Download and extract the Qualcomm SDK from the given URL into target_dir.

    Args:
        url (str): The URL of the archive (.zip, .tar.gz, or .tgz).
        prefix_to_strip (str): Top-level directory inside the archive to strip
                               from extracted file paths.
        target_dir (pathlib.Path): Directory to extract the SDK into.

    Notes:
        - Only runs on Linux x86 platforms. Skips otherwise.
        - Creates the target_dir if it does not exist.
    """
    # Default path where the Qualcomm SDK will be installed (under the script directory).

    # URL to download the Qualcomm AI Runtime SDK archive.
    qairt_url = (
        "https://softwarecenter.qualcomm.com/api/download/software/sdks/"
        "Qualcomm_AI_Runtime_Community/All/2.34.0.250424/v2.34.0.250424.zip"
    )

    # Top-level directory inside the SDK archive to extract.
    qairt_content_dir = "qairt/2.34.0.250424"

    if not is_linux_x86():
        print("Skipping Qualcomm SDK (only supported on Linux x86).")
        return

    SDK_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = pathlib.Path(tmpdir) / pathlib.Path(qairt_url).name

        print(f"Downloading Qualcomm SDK from {qairt_url}...")
        urllib.request.urlretrieve(qairt_url, archive_path)

        if qairt_url.endswith(".zip"):
            _extract_zip(archive_path, qairt_content_dir, SDK_DIR)
        elif qairt_url.endswith((".tar.gz", ".tgz")):
            _extract_tar(archive_path, qairt_content_dir, SDK_DIR)
        else:
            raise ValueError(f"Unsupported archive format: {qairt_url}")

        print(f"Qualcomm SDK extracted to {SDK_DIR}")

    return SDK_DIR


def _extract_zip(archive_path: pathlib.Path, prefix: str, target_dir: pathlib.Path):
    """
    Extract files from a zip archive into target_dir, stripping a prefix.

    Args:
        archive_path (pathlib.Path): Path to the .zip archive.
        prefix (str): Prefix folder inside the archive to strip.
        target_dir (pathlib.Path): Destination directory.
    """
    with zipfile.ZipFile(archive_path, "r") as zf:
        for member in zf.infolist():
            if not member.filename.startswith(prefix + "/"):
                continue
            relpath = pathlib.Path(member.filename).relative_to(prefix)
            if not relpath.parts or relpath.parts[0] == "..":
                continue
            out_path = target_dir / relpath
            if member.is_dir():
                out_path.mkdir(parents=True, exist_ok=True)
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())


def _extract_tar(archive_path: pathlib.Path, prefix: str, target_dir: pathlib.Path):
    """
    Extract files from a tar.gz archive into target_dir, stripping a prefix.

    Args:
        archive_path (pathlib.Path): Path to the .tar.gz or .tgz archive.
        prefix (str): Prefix folder inside the archive to strip.
        target_dir (pathlib.Path): Destination directory.
    """
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
                    # Skip non-regular files (links, devices, etc.)
                    continue
                with src, open(out_path, "wb") as dst:
                    dst.write(src.read())
