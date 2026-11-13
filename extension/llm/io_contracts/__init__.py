# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The hf_static I/O contract and its registered Python drivers.

Importing this package registers the runner with Hugging Face ``transformers``
exporters when the contract hook is available.
"""

from executorch.extension.llm.io_contracts.hf_static import (  # noqa: F401
    HfStaticDriver,
    HfStaticRunner,
)
