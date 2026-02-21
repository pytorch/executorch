#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""Serialization utilities for MLX delegate."""

import subprocess
import sys
from pathlib import Path


def _ensure_generated_files_exist():
    """
    Ensure that auto-generated files exist.

    If the generated files don't exist (e.g., fresh checkout), run generate.py
    to create them. This allows Python imports to work without requiring a
    manual generate step.
    """
    serialization_dir = Path(__file__).parent
    schema_py = serialization_dir / "mlx_graph_schema.py"

    if not schema_py.exists():
        print("MLX delegate: Auto-generating code from schema.fbs...", file=sys.stderr)
        generate_script = serialization_dir / "generate.py"

        # Find executorch root (for working directory)
        executorch_root = serialization_dir.parent.parent.parent.parent

        result = subprocess.run(
            [sys.executable, str(generate_script)],
            cwd=str(executorch_root),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Error generating MLX code: {result.stderr}", file=sys.stderr)
            raise RuntimeError(
                f"Failed to generate MLX delegate code. "
                f"Run 'python {generate_script}' manually to see the error."
            )

        print("MLX delegate: Code generation complete.", file=sys.stderr)


# Auto-generate files if they don't exist
_ensure_generated_files_exist()

# Export serialization functions for convenience
from executorch.backends.mlx.serialization.mlx_graph_serialize import (  # noqa: F401, E501
    deserialize_to_json,
    parse_header,
    serialize_mlx_graph,
)

__all__ = [
    "deserialize_to_json",
    "parse_header",
    "serialize_mlx_graph",
]
