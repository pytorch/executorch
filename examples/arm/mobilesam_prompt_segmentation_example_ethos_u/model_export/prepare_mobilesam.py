# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import subprocess  # nosec B404
from pathlib import Path


MOBILE_SAM_SOURCE_URL = "https://github.com/ChaoningZhang/MobileSAM.git"
MOBILE_SAM_SOURCE_REVISION = "f706ad9c4eb7f219c00d9050e46328518ffb65d2"
PATCH_DIR = Path(__file__).resolve().parent / "patches" / "mobile_sam"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)  # nosec B603


def default_source_dir() -> Path:
    return (
        Path.home()
        / ".cache"
        / "executorch"
        / "mobilesam"
        / MOBILE_SAM_SOURCE_REVISION
        / "source"
    )


def prepare_source(source_dir: Path, *, local_files_only: bool) -> None:
    source_dir = source_dir.expanduser().resolve()
    marker = source_dir.parent / f".{source_dir.name}.executorch-managed"

    if source_dir.exists() and not marker.is_file():
        raise RuntimeError(
            f"Refusing to modify unmanaged MobileSAM directory: {source_dir}"
        )

    if not source_dir.exists():
        if local_files_only:
            raise FileNotFoundError(
                f"Managed MobileSAM checkout not found: {source_dir}"
            )
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                MOBILE_SAM_SOURCE_URL,
                str(source_dir),
            ]
        )
        marker.write_text(MOBILE_SAM_SOURCE_REVISION + "\n")
        run(
            [
                "git",
                "sparse-checkout",
                "set",
                "mobile_sam",
            ],
            cwd=source_dir,
        )

    if not local_files_only:
        run(
            ["git", "fetch", "--quiet", "origin", MOBILE_SAM_SOURCE_REVISION],
            cwd=source_dir,
        )

    try:
        run(
            ["git", "cat-file", "-e", f"{MOBILE_SAM_SOURCE_REVISION}^{{commit}}"],
            cwd=source_dir,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"MobileSAM revision {MOBILE_SAM_SOURCE_REVISION} is unavailable locally."
        ) from error

    run(
        ["git", "checkout", "--detach", "--force", MOBILE_SAM_SOURCE_REVISION],
        cwd=source_dir,
    )
    run(["git", "reset", "--hard", MOBILE_SAM_SOURCE_REVISION], cwd=source_dir)

    patches = sorted(PATCH_DIR.glob("*.patch"))
    if not patches:
        raise FileNotFoundError(f"No MobileSAM patches found in {PATCH_DIR}")
    for patch in patches:
        run(["git", "apply", "--check", str(patch)], cwd=source_dir)
        run(["git", "apply", str(patch)], cwd=source_dir)

    print(f"Prepared patched MobileSAM source at {source_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the pinned MobileSAM source with ExecuTorch patches."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=default_source_dir(),
        help="Managed checkout destination.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Reuse an existing managed checkout without network access.",
    )
    args = parser.parse_args()
    prepare_source(args.source_dir, local_files_only=args.local_files_only)


if __name__ == "__main__":
    main()
