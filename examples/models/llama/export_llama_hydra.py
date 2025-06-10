# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Run export_llama using the new Hydra CLI.
"""

import hydra

from executorch.examples.models.llama.config.llm_config import LlmConfig
from executorch.examples.models.llama.export_llama_lib import export_llama
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

cs = ConfigStore.instance()
cs.store(name="llm_config", node=LlmConfig)


@hydra.main(version_base=None, config_name="llm_config")
def main(llm_config: LlmConfig) -> None:
    export_llama(OmegaConf.to_object(llm_config))


if __name__ == "__main__":
    main()
