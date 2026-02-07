#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
from pathlib import Path
import shutil
import sys


def die(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def write_modulemap(path: Path, module_name: str) -> None:
    content = (
        f"module {module_name} [system] {{\n"
        "  umbrella \".\"\n"
        "  export *\n"
        "  module * { export * }\n"
        "}\n"
    )
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a SwiftPM static-library artifact bundle for ONNX Runtime."
    )
    parser.add_argument("--name", default="onnxruntime", help="Artifact name")
    parser.add_argument("--version", required=True, help="Artifact version string")
    parser.add_argument(
        "--staging-dir",
        required=True,
        help="Staging directory containing include/ and lib/<triple>/...",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory to place <name>.artifactbundle",
    )
    parser.add_argument(
        "--modulemap",
        default="",
        help="Optional path to a module map to copy into include/",
    )
    args = parser.parse_args()

    staging = Path(args.staging_dir).resolve()
    include_dir = staging / "include"
    lib_root = staging / "lib"

    if not include_dir.is_dir():
        die(f"missing include directory: {include_dir}")
    if not lib_root.is_dir():
        die(f"missing lib directory: {lib_root}")

    output_dir = Path(args.output_dir).resolve()
    bundle_dir = output_dir / f"{args.name}.artifactbundle"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    shutil.copytree(include_dir, bundle_dir / "include")
    shutil.copytree(lib_root, bundle_dir / "lib")

    modulemap_path = bundle_dir / "include" / f"{args.name}.modulemap"
    if args.modulemap:
        src = Path(args.modulemap).resolve()
        if not src.is_file():
            die(f"modulemap not found: {src}")
        shutil.copy2(src, modulemap_path)
    else:
        write_modulemap(modulemap_path, args.name)

    variants = []
    for triple_dir in sorted((bundle_dir / "lib").iterdir()):
        if not triple_dir.is_dir():
            continue
        libs = list(triple_dir.glob("*.a")) + list(triple_dir.glob("*.lib"))
        if len(libs) != 1:
            die(
                f"expected exactly one static library in {triple_dir}, found {len(libs)}"
            )
        lib_file = libs[0]
        variants.append(
            {
                "path": str(lib_file.relative_to(bundle_dir)),
                "supportedTriples": [triple_dir.name],
                "staticLibraryMetadata": {
                    "headerPaths": ["include"],
                    "moduleMapPath": f"include/{args.name}.modulemap",
                },
            }
        )

    if not variants:
        die("no library variants found")

    info = {
        "schemaVersion": "1.0",
        "artifacts": {
            args.name: {
                "type": "staticLibrary",
                "version": args.version,
                "variants": variants,
            }
        },
    }

    (bundle_dir / "info.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"wrote {bundle_dir}")


if __name__ == "__main__":
    main()
