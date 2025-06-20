# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export an LLM with ExecuTorch. Currently follows the following steps:
1. Instantiate our custom PyTorch transformer definition from examples/llama/models/llama_transformer.py.
2. Load weights into the model.
3. Apply source transformations/TorchAO quantization.
4. Export model to intermediate IRs.
5. Graph transformations/PT2E quantization.
6. Partition graph and delegate to backend(s).
7. Export to final ExecuTorch .pte format.

Example usage using full CLI arguments:
python -m extension.llm.export.export_llm \
    base.model_class="llama3" \
    model.use_sdpa_with_kv_cache=True \
    model.use_kv_cache=True \
    debug.verbose=True \
    backend.xnnpack.enabled=True \
    backend.xnnpack.extended_ops=True \
    quantization.qmode="8da4w"
"""

import hydra

from executorch.examples.models.llama.config.llm_config import LlmConfig
from executorch.examples.models.llama.export_llama_lib import export_llama
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

cs = ConfigStore.instance()
cs.store(name="llm_config", node=LlmConfig)


@hydra.main(version_base=None, config_path=None, config_name="llm_config")
def main(llm_config: LlmConfig) -> None:
    export_llama(OmegaConf.to_object(llm_config))


if __name__ == "__main__":
    main()
