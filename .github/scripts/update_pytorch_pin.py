#!/usr/bin/env python3

import argparse
import base64
import json
import re
import runpy
import subprocess
import sys
import urllib.request
from pathlib import Path
from urllib.parse import unquote


_DEPENDENCY_CONFIG = runpy.run_path("torch_pin.py")
PYTORCH_INDEX_URL = _DEPENDENCY_CONFIG["PYTORCH_INDEX_URL"]
TORCHAO_INDEX_URL = _DEPENDENCY_CONFIG["TORCHAO_INDEX_URL"]
PYTORCH_PACKAGES = ("torch", "torchvision", "torchaudio")
REQUIRED_PYTHON_TAG = "cp310"
CPU_WHEEL_PLATFORMS = (
    "manylinux_2_28_x86_64",
    "manylinux_2_28_aarch64",
    "macosx_14_0_arm64",
    "win_amd64",
)
CUDA_WHEEL_PLATFORMS = ("manylinux_2_28_x86_64", "manylinux_2_28_aarch64")


def parse_nightly_version(pytorch_version):
    """
    Parse a full nightly wheel version into its source-snapshot date.

    Args:
        pytorch_version: Version such as '2.15.0.dev20251004'

    Returns:
        Date string in format 'YYYY-MM-DD'
    """
    match = re.fullmatch(r"\d+\.\d+\.\d+\.dev(\d{4})(\d{2})(\d{2})", pytorch_version)
    if not match:
        raise ValueError(f"Invalid PyTorch nightly version: {pytorch_version}")

    return format_nightly_date("".join(match.groups()))


def format_nightly_date(nightly_date):
    match = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", nightly_date)
    if not match:
        raise ValueError(f"Invalid nightly date: {nightly_date}")
    year, month, day = match.groups()
    return f"{year}-{month}-{day}"


def get_pytorch_version():
    """
    Read the authoritative PyTorch wheel version from torch_pin.py.

    Returns:
        PYTORCH_VERSION string
    """
    with open("torch_pin.py", "r") as f:
        content = f.read()

    match = re.search(r'PYTORCH_VERSION\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError("Could not find PYTORCH_VERSION in torch_pin.py")

    return match.group(1)


def get_pytorch_nightly_versions(max_date):
    """Return one compatible nightly package set available on every wheel train."""
    config = runpy.run_path("torch_pin.py")
    channels = ["cpu", *config["CUDA_WHEEL_VERSIONS"]]
    versions_by_package = {}
    common_dates = None

    for package in PYTORCH_PACKAGES:
        package_versions = {}
        package_dates = None
        for channel in channels:
            url = f"{PYTORCH_INDEX_URL}/{channel}/{package}/"
            request = urllib.request.Request(
                url, headers={"User-Agent": "ExecuTorch-Bot"}
            )
            with urllib.request.urlopen(request) as response:
                index_html = unquote(response.read().decode())
            wheels_by_date = {}
            pattern = re.compile(
                rf"^{re.escape(package)}-(\d+\.\d+\.\d+\.dev(\d{{8}}))"
                rf"(?:\+[^-]+)?-{REQUIRED_PYTHON_TAG}-[^-]+-.*\.whl$"
            )
            for link in re.findall(r'href="([^"]+)"', index_html):
                filename = link.rsplit("/", 1)[-1].split("#", 1)[0]
                match = pattern.match(filename)
                if match is None:
                    continue
                version, date = match.groups()
                if date <= max_date:
                    versions, filenames = wheels_by_date.setdefault(date, (set(), []))
                    versions.add(version)
                    filenames.append(filename)

            required_platforms = (
                CPU_WHEEL_PLATFORMS if channel == "cpu" else CUDA_WHEEL_PLATFORMS
            )
            by_date = {
                date: versions
                for date, (versions, filenames) in wheels_by_date.items()
                if all(
                    any(platform_name in filename for filename in filenames)
                    for platform_name in required_platforms
                )
            }
            package_dates = (
                set(by_date) if package_dates is None else package_dates & set(by_date)
            )
            for date, versions in by_date.items():
                package_versions.setdefault(date, set()).update(versions)

        package_dates = package_dates or set()
        versions_by_package[package] = {
            date: next(iter(package_versions[date]))
            for date in package_dates
            if len(package_versions[date]) == 1
        }
        common_dates = (
            set(versions_by_package[package])
            if common_dates is None
            else common_dates & set(versions_by_package[package])
        )

    if not common_dates:
        raise ValueError(
            f"Could not find a PyTorch nightly on or before {max_date} for "
            f"{', '.join(PYTORCH_PACKAGES)} on {', '.join(channels)}"
        )
    selected_date = max(common_dates)
    return {
        package: versions_by_package[package][selected_date]
        for package in PYTORCH_PACKAGES
    }


def update_pytorch_package_pins(versions):
    """Update the wheel versions that form the one selected PyTorch nightly."""
    config_path = Path("torch_pin.py")
    content = config_path.read_text()
    assignments = {
        "PYTORCH_VERSION": versions["torch"],
        "TORCHVISION_VERSION": versions["torchvision"],
        "TORCHAUDIO_VERSION": versions["torchaudio"],
    }
    for name, version in assignments.items():
        content, count = re.subn(
            rf'^(?P<prefix>{name}\s*=\s*["\'])[^"\']+(?P<suffix>["\'])$',
            rf"\g<prefix>{version}\g<suffix>",
            content,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise ValueError(f"Could not find one {name} assignment in {config_path}")
    config_path.write_text(content)


def get_json(url):
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "ExecuTorch-Bot")
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())


def get_commit_hash_for_nightly(date_str):
    """
    Fetch commit hash from PyTorch nightly branch for a given date.

    Args:
        date_str: Date string in format 'YYYY-MM-DD'

    Returns:
        Commit hash string
    """
    api_url = "https://api.github.com/repos/pytorch/pytorch/commits"
    params = "?sha=nightly&per_page=50"
    url = api_url + params

    try:
        commits = get_json(url)
    except Exception as e:
        print(f"Error fetching commits: {e}", file=sys.stderr)
        sys.exit(1)

    # Look for commit with title matching "{date_str} nightly release"
    target_title = f"{date_str} nightly release"

    for commit in commits:
        commit_msg = commit.get("commit", {}).get("message", "")
        # Check if the first line of commit message matches
        first_line = commit_msg.split("\n")[0].strip()
        if first_line.startswith(f"{date_str} nightly"):
            return extract_hash_from_title(first_line)

    raise ValueError(
        f"Could not find commit with title matching '{target_title}' in nightly branch"
    )


def extract_hash_from_title(title):
    match = re.search(r"\(([0-9a-fA-F]{7,40})\)", title)
    if not match:
        raise ValueError(f"Could not extract commit hash from title '{title}'")
    return match.group(1)


def update_pytorch_pin(commit_hash):
    """
    Update .ci/docker/ci_commit_pins/pytorch.txt with the new commit hash.

    Args:
        commit_hash: Commit hash to write
    """
    pin_file = ".ci/docker/ci_commit_pins/pytorch.txt"
    with open(pin_file, "w") as f:
        f.write(f"{commit_hash}\n")
    print(f"Updated {pin_file} with commit hash: {commit_hash}")


def get_supported_torchao_channels():
    cuda_versions = runpy.run_path("torch_pin.py")["CUDA_WHEEL_VERSIONS"]
    return ["cpu", *cuda_versions]


def get_torchao_versions(channel):
    url = f"{TORCHAO_INDEX_URL}/{channel}/torchao/"
    req = urllib.request.Request(url, headers={"User-Agent": "ExecuTorch-Bot"})
    with urllib.request.urlopen(req) as response:
        index_html = unquote(response.read().decode())

    wheel_tags = {}
    pattern = re.compile(
        rf"^torchao-(\d+\.\d+\.\d+\.dev\d{{8}})\+{re.escape(channel)}-(.+)\.whl$"
    )
    for filename in re.findall(r'href="[^"]*/(torchao-[^"]+\.whl)"', index_html):
        match = pattern.match(filename)
        if match:
            wheel_tags.setdefault(match.group(1), set()).add(match.group(2))

    required_tags = ("py3-none-any", "aarch64") if channel == "cpu" else ("x86_64",)
    return {
        version
        for version, tags in wheel_tags.items()
        if all(any(required in tag for tag in tags) for required in required_tags)
    }


def get_latest_torchao_nightly(max_date):
    common_versions = None
    channels = get_supported_torchao_channels()
    for channel in channels:
        versions = get_torchao_versions(channel)
        common_versions = (
            versions if common_versions is None else common_versions & versions
        )

    candidates = [
        version
        for version in common_versions or []
        if version.rsplit(".dev", 1)[-1] <= max_date
    ]
    if not candidates:
        raise ValueError(
            f"Could not find a TorchAO nightly on or before {max_date} for "
            f"all supported channels: {', '.join(channels)}"
        )
    return max(candidates, key=lambda version: (version.rsplit(".dev", 1)[-1], version))


def get_torchao_commit_hash(nightly_version):
    date = nightly_version.rsplit(".dev", 1)[-1]
    formatted_date = format_nightly_date(date)
    url = (
        "https://api.github.com/repos/pytorch/ao/actions/workflows/"  # @lint-ignore
        "build_wheels_linux_x86.yml/runs?event=schedule&status=success&"
        f"created={formatted_date}&per_page=100"
    )
    runs = get_json(url).get("workflow_runs", [])
    if not runs:
        raise ValueError(
            f"Could not find the successful TorchAO wheel build for {nightly_version}"
        )
    return runs[0]["head_sha"]


def update_torchao_pins(nightly_version, commit_hash):
    config_path = Path("torch_pin.py")
    content = config_path.read_text()
    content, default_replacements = re.subn(
        r'^(?P<prefix>TORCHAO_NIGHTLY_VERSION\s*=\s*["\'])[^"\']+(?P<suffix>["\'])$',
        rf"\g<prefix>{nightly_version}\g<suffix>",
        content,
        flags=re.MULTILINE,
    )
    if default_replacements != 1:
        raise ValueError(f"Could not find the TorchAO nightly pin in {config_path}")
    config_path.write_text(content)

    for command in (
        ["git", "submodule", "update", "--init", "third-party/ao"],
        [
            "git",
            "-C",
            "third-party/ao",
            "fetch",
            "--depth=1",
            "origin",
            commit_hash,
        ],
        ["git", "-C", "third-party/ao", "checkout", "--detach", commit_hash],
    ):
        subprocess.run(command, check=True)
    print(
        f"Updated TorchAO nightly pins to {nightly_version} and third-party/ao "
        f"to {commit_hash}"
    )


def should_skip_file(filename):
    """
    Check if a file should be skipped during sync (build files).

    Args:
        filename: Base filename to check

    Returns:
        True if file should be skipped
    """
    skip_files = {"BUCK", "CMakeLists.txt", "TARGETS", "targets.bzl"}
    return filename in skip_files


def fetch_file_content(commit_hash, file_path):
    """
    Fetch file content from GitHub API.

    Args:
        commit_hash: Commit hash to fetch from
        file_path: File path in the repository

    Returns:
        File content as bytes
    """
    api_url = f"https://api.github.com/repos/pytorch/pytorch/contents/{file_path}?ref={commit_hash}"

    req = urllib.request.Request(api_url)
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "ExecuTorch-Bot")

    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            # Content is base64 encoded
            content = base64.b64decode(data["content"])
            return content
    except urllib.request.HTTPError as e:
        print(f"Error fetching file {file_path}: {e}", file=sys.stderr)
        raise


def sync_directory(et_dir, pt_path, commit_hash):
    """
    Sync files from PyTorch to ExecuTorch using GitHub API.
    Only syncs files that already exist in ExecuTorch - does not add new files.

    Args:
        et_dir: ExecuTorch directory path
        pt_path: PyTorch directory path in the repository (e.g., "c10")
        commit_hash: Commit hash to fetch from

    Returns:
        Number of files grafted
    """
    files_grafted = 0
    print(f"Checking {et_dir} vs pytorch/{pt_path}...")

    if not et_dir.exists():
        print(f"Warning: ExecuTorch directory {et_dir} does not exist, skipping")
        return 0

    # Loop through files in ExecuTorch directory
    for et_file in et_dir.rglob("*"):
        if not et_file.is_file():
            continue

        # Skip build files
        if should_skip_file(et_file.name):
            continue

        # Construct corresponding path in PyTorch
        rel_path = et_file.relative_to(et_dir)
        pt_file_path = f"{pt_path}/{rel_path}".replace("\\", "/")

        # Fetch content from PyTorch and compare
        try:
            pt_content = fetch_file_content(commit_hash, pt_file_path)
            et_content = et_file.read_bytes()

            if pt_content != et_content:
                print(f"⚠️  Difference detected in {rel_path}")
                print(f"📋 Grafting from PyTorch commit {commit_hash}...")

                et_file.write_bytes(pt_content)
                print(f"✅ Grafted {et_file}")
                files_grafted += 1
        except urllib.request.HTTPError as e:
            if e.code != 404:  # It's ok to have more files in ET than pytorch/pytorch.
                print(f"Error fetching {rel_path} from PyTorch: {e}")
        except Exception as e:
            print(f"Error syncing {rel_path}: {e}")
            continue

    return files_grafted


def sync_c10_directories(commit_hash):
    """
    Sync c10 and torch/headeronly directories from PyTorch to ExecuTorch using GitHub API.

    Args:
        commit_hash: PyTorch commit hash to sync from

    Returns:
        Total number of files grafted
    """
    print("\n🔄 Syncing c10 directories from PyTorch via GitHub API...")

    # Get repository root
    repo_root = Path.cwd()

    # Define directory pairs to sync (from check_c10_sync.sh)
    # Format: (executorch_dir, pytorch_path_in_repo)
    dir_pairs = [
        (
            repo_root / "runtime/core/portable_type/c10/c10",
            "c10",
        ),
        (
            repo_root / "runtime/core/portable_type/c10/torch/headeronly",
            "torch/headeronly",
        ),
    ]

    total_grafted = 0
    for et_dir, pt_path in dir_pairs:
        files_grafted = sync_directory(et_dir, pt_path, commit_hash)
        total_grafted += files_grafted

    if total_grafted > 0:
        print(f"\n✅ Successfully grafted {total_grafted} file(s) from PyTorch")
    else:
        print("\n✅ No differences found - c10 is in sync")

    return total_grafted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nightly-date",
        default="",
        help="newest PyTorch nightly date to consider, in YYYYMMDD form",
    )
    args = parser.parse_args()
    try:
        if args.nightly_date:
            if re.fullmatch(r"\d{8}", args.nightly_date) is None:
                raise ValueError("--nightly-date must use YYYYMMDD")
            package_versions = get_pytorch_nightly_versions(args.nightly_date)
            update_pytorch_package_pins(package_versions)
            print(f"Selected PyTorch package versions: {package_versions}")

        pytorch_version = get_pytorch_version()
        print(f"Found PYTORCH_VERSION: {pytorch_version}")

        # The wheel version is authoritative; derive its source snapshot date.
        date_str = parse_nightly_version(pytorch_version)
        print(f"Parsed date: {date_str}")

        # Fetch commit hash from PyTorch nightly branch
        commit_hash = get_commit_hash_for_nightly(date_str)
        print(f"Found commit hash: {commit_hash}")

        # Update the pin file
        update_pytorch_pin(commit_hash)

        # Sync c10 directories from PyTorch
        sync_c10_directories(commit_hash)

        # Select the newest TorchAO nightly available for every supported CUDA
        # channel and align the source submodule with the commit that built it.
        max_torchao_date = date_str.replace("-", "")
        torchao_version = get_latest_torchao_nightly(max_torchao_date)
        print(f"Found TorchAO nightly version: {torchao_version}")
        torchao_commit_hash = get_torchao_commit_hash(torchao_version)
        print(f"Found TorchAO commit hash: {torchao_commit_hash}")
        update_torchao_pins(torchao_version, torchao_commit_hash)

        print(
            "\n✅ Successfully updated PyTorch and TorchAO pins and synced c10 directories!"
        )

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
