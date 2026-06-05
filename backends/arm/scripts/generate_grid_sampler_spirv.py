# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import base64
import shutil
import subprocess  # nosec B404 - required to invoke the shader compiler.
import tempfile
from pathlib import Path


SHADER_DIR = Path(__file__).resolve().parents[1] / "vgf" / "shaders"
DEFAULT_SOURCE = SHADER_DIR / "grid_sampler.glsl"
DEFAULT_OUTPUT = SHADER_DIR / "grid_sampler.spirv.b64"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile the VGF grid_sampler GLSL shader to SPIR-V and write the "
            "base64-encoded payload consumed by the ExecuTorch custom-shader "
            "lowering."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"GLSL source file. Defaults to {DEFAULT_SOURCE}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Base64 SPIR-V output file. Defaults to {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--glslc",
        default="glslc",
        help="Path to glslc. Defaults to resolving glslc from PATH.",
    )
    return parser.parse_args()


def _resolve_glslc(glslc: str) -> str:
    resolved = shutil.which(glslc)
    if resolved is None:
        raise RuntimeError(
            f"Could not find {glslc}. Install the Vulkan SDK or pass --glslc."
        )
    return resolved


def _write_base64_spirv(spirv_path: Path, output_path: Path) -> None:
    encoded = base64.b64encode(spirv_path.read_bytes()).decode("ascii")
    output_path.write_text(encoded + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    glslc = _resolve_glslc(args.glslc)

    with tempfile.TemporaryDirectory() as tmpdir:
        spirv_path = Path(tmpdir) / "grid_sampler.spirv"
        subprocess.run(  # nosec B603 - glslc path is resolved explicitly.
            [glslc, str(args.source), "-o", str(spirv_path)],
            check=True,
        )
        _write_base64_spirv(spirv_path, args.output)


if __name__ == "__main__":
    main()
