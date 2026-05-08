#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""Serialization utilities for MLX delegate."""

from pathlib import Path


_schema_py = Path(__file__).parent / "mlx_graph_schema.py"
if not _schema_py.exists():
    raise ImportError(
        "MLX delegate generated files not found. "
        "Run 'python install_executorch.py' first."
    )

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
