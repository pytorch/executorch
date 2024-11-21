# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

from executorch.backends.arm.test.common import arm_test_options, get_option
from torch.export import ExportedProgram

logger = logging.getLogger(__name__)
_model_explorer_installed = False

try:
    # pyre-ignore[21]: We keep track of whether import succeeded manually.
    from model_explorer import config, visualize_from_config, visualize_pytorch

    _model_explorer_installed = True
except ImportError:
    logger.warning("model-explorer is not installed, can't visualize models.")


def is_model_explorer_installed() -> bool:
    return _model_explorer_installed


def get_pytest_option_host() -> str | None:
    host = get_option(arm_test_options.model_explorer_host)
    return str(host) if host else None


def get_pytest_option_port() -> int | None:
    port = get_option(arm_test_options.model_explorer_port)
    return int(port) if port else None


def visualize(
    exported_program: ExportedProgram,
    host: Optional[str] = None,
    port: Optional[int] = None,
):
    """Attempt visualizing exported_program using model-explorer."""

    host = host if host else get_pytest_option_host()
    port = port if port else get_pytest_option_port()

    if not is_model_explorer_installed():
        logger.warning("Can't visualize model since model-explorer is not installed.")
        return

    # If a host is provided, we attempt connecting to an already running server.
    # Note that this needs a modified model-explorer
    if host:
        explorer_config = (
            config()
            .add_model_from_pytorch("ExportedProgram", exported_program)
            .set_reuse_server(server_host=host, server_port=port)
        )
        visualize_from_config(explorer_config)
    else:
        visualize_pytorch(exported_program)
