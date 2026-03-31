# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# This module has moved to executorch.backends.test.graph_builder.
# This re-export exists for backward compatibility.
from executorch.backends.test.graph_builder import GraphBuilder, single_op_builder

__all__ = ["GraphBuilder", "single_op_builder"]
