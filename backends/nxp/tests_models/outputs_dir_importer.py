# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

logger = logging.getLogger(__name__)

try:
    import test.python.outputs_dir as outputs_dir
    logger.debug("Importing from executorch-integration")
except ImportError:
    import executorch.backends.nxp.tests_models.outputs_dir as outputs_dir
    logger.debug("Importing from executorch")