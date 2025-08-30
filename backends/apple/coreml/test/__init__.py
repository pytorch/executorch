# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

try:
    import pytest

    # Skip on Windows
    if sys.platform == "win32":
        pytest.skip("Core ML is not available on Windows.", allow_module_level=True)

except ImportError:
    pass
