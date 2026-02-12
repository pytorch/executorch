# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

logger = logging.getLogger(__name__)

try:
    import test.python.config as test_config  # noqa F401

    logger.debug("Importing from executorch-integration")
except ImportError:
    import executorch.backends.nxp.tests_models.config as test_config  # noqa F401

    logger.debug("Importing from executorch")
