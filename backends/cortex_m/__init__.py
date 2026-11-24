# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

logger = logging.getLogger(__name__)
if logger.level == logging.NOTSET:
    logger.setLevel(min(logging.INFO, logging.root.getEffectiveLevel()))
    logger.info(
        "Using default logging level %s for %s. Override with "
        "logging.getLogger(%r).setLevel(logging.DEBUG).",
        logging.getLevelName(logger.level),
        __name__,
        __name__,
    )
