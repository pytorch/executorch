# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MessageSample:
    """
    One independent multimodal calibration sample.

    messages uses the HuggingFace message format:
      [{"role": "user"|"assistant", "content": "..."}]
    This makes it directly compatible with HuggingFace chat templates and
    AutoProcessor inputs without any conversion.

    files: image/audio URLs or local paths fed to the encoder.
    messages: full conversation sequence used to build the decoder
              calibration token sequence.
    """

    files: List[str] = field(default_factory=list)
    messages: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        for i, msg in enumerate(self.messages):
            if "role" not in msg or "content" not in msg:
                raise ValueError(
                    f"messages[{i}] must have 'role' and 'content' keys, got: {list(msg.keys())}"
                )
