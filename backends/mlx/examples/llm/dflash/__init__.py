# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""DFlash draft model package.

Public API:
  - arch: resolve_arch, model_type <-> HF config mapping
  - adapters: DFlashAdapter registry, Qwen3 draft weight loading
  - model: DFlashDraftModel, DFlashConfig, load_dflash_config
  - cache: DFlashDraftKVCache (persistent draft KV cache)
  - export: export entry-point (target + draft -> one .pte)
  - run: speculative decoding loop runner
"""

from executorch.backends.mlx.examples.llm.dflash.adapters import (  # noqa: F401
    get_adapter,
)
from executorch.backends.mlx.examples.llm.dflash.arch import resolve_arch  # noqa: F401
from executorch.backends.mlx.examples.llm.dflash.cache import (  # noqa: F401
    DFlashDraftKVCache,
)
from executorch.backends.mlx.examples.llm.dflash.model import (  # noqa: F401
    DFlashConfig,
    DFlashDraftModel,
    load_dflash_config,
)
