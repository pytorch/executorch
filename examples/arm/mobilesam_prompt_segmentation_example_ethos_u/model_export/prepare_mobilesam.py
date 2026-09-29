# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import subprocess  # nosec B404
import urllib.request
from pathlib import Path


REVISION = "f706ad9c4eb7f219c00d9050e46328518ffb65d2"
SOURCE_URL = "https://github.com/ChaoningZhang/MobileSAM.git"
CHECKPOINT_URL = (
    f"https://github.com/ChaoningZhang/MobileSAM/raw/{REVISION}/weights/mobile_sam.pt"
)
CHECKPOINT_SHA256 = "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f"
CACHE_DIR = Path.home() / ".cache" / "executorch" / "mobilesam" / REVISION
SOURCE_DIR = CACHE_DIR / "source"
CHECKPOINT = CACHE_DIR / "mobile_sam.pt"
PATCH = (
    Path(__file__).parent
    / "patches"
    / "mobile_sam"
    / ("0001-Make-TinyViT-image-size-configurable.patch")
)


def run(*command: str, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)  # nosec B603


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_source() -> None:
    marker = SOURCE_DIR.parent / ".source.executorch-managed"
    if SOURCE_DIR.exists() and not marker.exists():
        raise RuntimeError(f"Refusing to modify unmanaged directory: {SOURCE_DIR}")

    if not SOURCE_DIR.exists():
        SOURCE_DIR.parent.mkdir(parents=True, exist_ok=True)
        run(
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            SOURCE_URL,
            str(SOURCE_DIR),
        )
        marker.write_text(REVISION + "\n")
        run("git", "sparse-checkout", "set", "mobile_sam", cwd=SOURCE_DIR)

    run("git", "fetch", "--quiet", "origin", REVISION, cwd=SOURCE_DIR)
    run("git", "checkout", "--detach", "--force", REVISION, cwd=SOURCE_DIR)
    run("git", "reset", "--hard", REVISION, cwd=SOURCE_DIR)
    run("git", "apply", str(PATCH), cwd=SOURCE_DIR)


def prepare_checkpoint() -> None:
    if not CHECKPOINT.exists():
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        with (
            urllib.request.urlopen(
                CHECKPOINT_URL, timeout=60
            ) as response,  # nosec B310
            CHECKPOINT.open("wb") as file,
        ):
            while chunk := response.read(1024 * 1024):
                file.write(chunk)

    actual_sha256 = sha256(CHECKPOINT)
    if actual_sha256 != CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Checkpoint SHA256 mismatch: expected {CHECKPOINT_SHA256}, "
            f"got {actual_sha256}"
        )


if __name__ == "__main__":
    prepare_source()
    prepare_checkpoint()
    print(f"MobileSAM ready in {CACHE_DIR}")
