# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


def run(command, expected_status=0):
    """Run a probe command, asserting its exit status."""
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode != expected_status:
        raise RuntimeError(
            f"{command} returned {result.returncode}, expected {expected_status}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip()


def has_device_architecture(architecture: str) -> bool:
    """Report whether the SDK identified a device architecture."""
    normalized = architecture.strip().lower()
    return bool(normalized and normalized != "unknown")


def _export_weighted_fixture(directory: Path) -> None:
    from export_bridge_fixture import export_weighted_fixture

    export_weighted_fixture(directory)


def compile_source(source: Path, compiled: Path, architecture: str) -> None:
    """Compile a portable source to an AOT model."""
    run(
        [
            "xcrun",
            "coreai-build",
            "compile",
            str(source),
            "--output",
            str(compiled),
            "--platform",
            "macOS",
            "--architecture",
            architecture,
            "--min-deployment-version",
            "27.0",
        ]
    )
    shutil.rmtree(source)


def run_variant(
    probe: Path, kind: str, source: Path, fixture: Path, bookmark: Path
) -> None:
    """Exercise specialize, restore, and evict for one source variant."""
    arguments = [str(source), str(fixture), str(bookmark)]
    try:
        print(run([str(probe), "specialize", *arguments]), flush=True)
        if source.exists() or not bookmark.is_file():
            raise AssertionError("Expected a saved bookmark and no original source")
        for _ in range(2):
            print(run([str(probe), "restore", *arguments]), flush=True)
    except BaseException:
        try:
            if bookmark.is_file():
                run([str(probe), "evict", *arguments])
        except Exception as cleanup:
            print(f"cleanup evict failed: {cleanup}", flush=True)
        raise
    if bookmark.is_file():
        run([str(probe), "evict", *arguments])
    run([str(probe), "restore", *arguments], expected_status=3)
    print(f"{kind}: persistent cache lifetime and eviction control passed", flush=True)


def main(probe: Path) -> None:
    """Run portable and compiled lifetime probes against a test binary."""
    architecture = run([str(probe), "architecture"])
    with tempfile.TemporaryDirectory(prefix="coreai-lifetime-") as temporary:
        root = Path(temporary)
        for kind in ("portable", "compiled"):
            # AOT compile rejects unknown architectures.
            if kind == "compiled" and not has_device_architecture(architecture):
                print(
                    f"compiled: skipped (unsupported architecture {architecture!r})",
                    flush=True,
                )
                continue
            directory = root / kind
            _export_weighted_fixture(directory)
            source = directory / "model.aimodel"
            if kind == "compiled":
                compiled = directory / "model.aimodelc"
                compile_source(source, compiled, architecture)
                source = compiled
            run_variant(
                probe,
                kind,
                source,
                directory / "fixture.json",
                directory / "saved.bookmark",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("probe", type=Path)
    main(parser.parse_args().probe.resolve())
