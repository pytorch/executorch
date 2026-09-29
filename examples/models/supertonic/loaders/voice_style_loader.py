# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

TTL_STYLE_DIMS = (1, 50, 256)
DP_STYLE_DIMS = (1, 8, 16)


@dataclass(frozen=True)
class VoiceStyle:
    ttl: NDArray[np.float32]
    dp: NDArray[np.float32]


def _validate_dimensions(style: dict[str, Any]) -> None:
    ttl_dims = tuple(style["style_ttl"]["dims"])
    dp_dims = tuple(style["style_dp"]["dims"])
    if ttl_dims != TTL_STYLE_DIMS:
        raise ValueError(
            f"style_ttl.dims must be {TTL_STYLE_DIMS}, received {ttl_dims}"
        )
    if dp_dims != DP_STYLE_DIMS:
        raise ValueError(f"style_dp.dims must be {DP_STYLE_DIMS}, received {dp_dims}")


def load_voice_style(voice_style_paths: Sequence[str | Path]) -> VoiceStyle:
    if len(voice_style_paths) == 0:
        raise ValueError("expected at least one voice style path")

    ttl_styles = []
    dp_styles = []
    for style_path in voice_style_paths:
        with Path(style_path).open(encoding="utf-8") as style_file:
            style = json.load(style_file)
        _validate_dimensions(style)
        ttl_styles.append(
            np.asarray(style["style_ttl"]["data"], dtype=np.float32).reshape(
                TTL_STYLE_DIMS[1:]
            )
        )
        dp_styles.append(
            np.asarray(style["style_dp"]["data"], dtype=np.float32).reshape(
                DP_STYLE_DIMS[1:]
            )
        )

    return VoiceStyle(ttl=np.stack(ttl_styles), dp=np.stack(dp_styles))
