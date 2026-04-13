# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from typing import List


_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_css() -> str:
    """Load base observatory CSS."""

    return _read_file(os.path.join(_TEMPLATE_DIR, "css", "main.css"))


def load_js_chunks() -> List[str]:
    """Load ordered observatory JS runtime chunks."""

    ordered = [
        "00_state.js",
        "01_utils.js",
        "02_layout.js",
        "03_blocks.js",
        "04_actions.js",
        "05_bootstrap_api.js",
    ]

    chunks: List[str] = []
    for filename in ordered:
        path = os.path.join(_TEMPLATE_DIR, "js", filename)
        chunks.append(f"\n// ---- {filename} ----\n")
        chunks.append(_read_file(path))
    return chunks
