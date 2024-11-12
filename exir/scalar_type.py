# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from executorch.exir.common_schema import ScalarType  # noqa [F401] 'executorch.exir.common_schema.ScalarType' imported but unused
import warnings
warnings.warn("executorch.exir.scalar_type is deprecated and will be removed in a future release. Please use executorch.exir.common_schema instead", DeprecationWarning)
