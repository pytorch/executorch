# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path


def dynamic_metadata(field: str, settings: dict[str, object]) -> str:
    if field != "version" or settings:
        raise ValueError("Only version metadata without settings is supported")
    return (
        os.environ.get("BUILD_VERSION", "").strip()
        or (Path(__file__).resolve().parents[3] / "version.txt").read_text().strip()
    )
