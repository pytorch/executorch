# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Graph I/O contracts ExecuTorch programs may follow, with their Python drivers.

Importing this package registers each contract's runner with Hugging Face
``transformers`` exporters when it is installed, so ``ExportedGenerator`` drives
programs labelled with the contract.
"""

from executorch.extension.llm.io_contracts.hf_static import (  # noqa: F401
    HfStaticDriver,
    HfStaticRunner,
)
from executorch.extension.llm.io_contracts.kv_cache_io import (  # noqa: F401
    CONTRACT as KV_CACHE_IO,
    io_contract,
    KvCacheIORunner,
)

try:
    from transformers.exporters import register_io_contract
except ImportError:
    pass
else:
    from executorch.extension.llm.io_contracts._transformers import KvCacheIOModelRunner

    register_io_contract(KV_CACHE_IO, KvCacheIOModelRunner)
