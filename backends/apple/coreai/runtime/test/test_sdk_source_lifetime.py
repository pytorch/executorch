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
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode != expected_status:
        raise RuntimeError(
            f"{command} returned {result.returncode}, expected {expected_status}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip()


def main(probe: Path) -> None:
    from export_bridge_fixture import export_weighted_fixture

    architecture = run([str(probe), "architecture"])
    with tempfile.TemporaryDirectory(prefix="coreai-lifetime-") as temporary:
        root = Path(temporary)
        for kind in ("portable", "compiled"):
            directory = root / kind
            export_weighted_fixture(directory)
            source = directory / "model.aimodel"
            if kind == "compiled":
                compiled = directory / "model.aimodelc"
                run(
                    [
                        "xcrun", "coreai-build", "compile", str(source),
                        "--output", str(compiled), "--platform", "macOS",
                        "--architecture", architecture, "--min-deployment-version", "27.0",
                    ]
                )
                shutil.rmtree(source)
                source = compiled
            bookmark = directory / "saved.bookmark"
            arguments = [str(source), str(directory / "fixture.json"), str(bookmark)]
            try:
                print(run([str(probe), "specialize", *arguments]), flush=True)
                if source.exists() or not bookmark.is_file():
                    raise AssertionError("Expected a saved bookmark and no original source")
                for _ in range(2):
                    print(run([str(probe), "restore", *arguments]), flush=True)
            finally:
                if bookmark.is_file():
                    run([str(probe), "evict", *arguments])
            run([str(probe), "restore", *arguments], expected_status=3)
            print(f"{kind}: persistent cache lifetime and eviction control passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("probe", type=Path)
    main(parser.parse_args().probe.resolve())
