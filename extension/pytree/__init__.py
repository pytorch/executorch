# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# flake8: noqa: F401

import logging

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


try:
    """
    Internally we link the respective c++ library functions but for the OSS pip
    build we will just use the python library for now. The python library is not
    exactly the same so it will not work for the runtime, but it'll be fine for
    now as in most cases the runtime will not need it.
    """

    # pyre-fixme[21]: Could not find module `executorch.extension.pytree.pybindings`.
    # @manual=//executorch/extension/pytree:pybindings
    from executorch.extension.pytree.pybindings import (
        broadcast_to_and_flatten as broadcast_to_and_flatten,
        from_str as from_str,
        register_custom as register_custom,
        tree_flatten as tree_flatten,
        tree_map as tree_map,
        tree_unflatten as tree_unflatten,
        TreeSpec as TreeSpec,
    )
except:
    logger.info(
        "Unable to import executorch.extension.pytree, using native torch pytree instead."
    )

    from torch.utils._pytree import (
        _broadcast_to_and_flatten,
        _register_pytree_node,
        tree_flatten,
        tree_map,
        tree_unflatten,
        TreeSpec,
        treespec_dumps,
        treespec_loads,
    )

    broadcast_to_and_flatten = _broadcast_to_and_flatten
    from_str = treespec_loads
    register_custom = _register_pytree_node
    TreeSpec.to_str = treespec_dumps  # pyre-ignore
