#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Centralized logging for the MLX backend.

Usage:
    from executorch.backends.mlx._logging import logger

    logger.info("Always visible (e.g., unsupported ops summary)")
    logger.debug("Only visible when ET_MLX_DEBUG=1")
    logger.warning("Always visible")

The logger is set to INFO by default, so logger.info() always prints.
Set ET_MLX_DEBUG=1 to lower the threshold to DEBUG for verbose output
(graph dumps, per-node traces, ops_to_not_decompose lists, etc.).
"""

import logging
import os

_MLX_DEBUG = os.environ.get("ET_MLX_DEBUG", "") not in ("", "0")

logger = logging.getLogger("executorch.backends.mlx")
logger.setLevel(logging.DEBUG if _MLX_DEBUG else logging.INFO)
logger.propagate = False

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
        )
    )
    logger.addHandler(_handler)
