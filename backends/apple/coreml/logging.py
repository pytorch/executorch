# Copyright © 2023 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import logging
import os
from typing import Optional


def get_coreml_log_level(default_level: int) -> Optional[str]:
    level_str = os.environ.get("ET_COREML_LOG_LEVEL", "").upper()
    if level_str == "":
        return default_level

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    if level_str not in level_map:
        raise ValueError(f"Invalid ET_COREML_LOG_LEVEL: {level_str}")
    return level_map[level_str]
