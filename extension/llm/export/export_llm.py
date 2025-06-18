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

Example usage using config file:
python -m extension.llm.export.export_llm \
    --config example_llm_config.yaml
"""

import argparse
import sys
from typing import Any, List, Tuple

import hydra
import yaml

from executorch.examples.models.llama.config.llm_config import LlmConfig
from executorch.examples.models.llama.export_llama_lib import export_llama
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

cs = ConfigStore.instance()
cs.store(name="llm_config", node=LlmConfig)


def parse_config_arg() -> Tuple[str, List[Any]]:
    """First parse out the arg for whether to use Hydra or the old CLI."""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", type=str, help="Path to the LlmConfig file")
    args, remaining = parser.parse_known_args()
    return args.config, remaining


def pop_config_arg() -> str:
    """
    Removes '--config' and its value from sys.argv.
    Assumes --config is specified and argparse has already validated the args.
    """
    idx = sys.argv.index("--config")
    value = sys.argv[idx + 1]
    del sys.argv[idx : idx + 2]
    return value


@hydra.main(version_base=None, config_name="llm_config")
def hydra_main(llm_config: LlmConfig) -> None:
    export_llama(OmegaConf.to_object(llm_config))


def main() -> None:
    config, remaining_args = parse_config_arg()
    if config:
        # Check if there are any remaining hydra CLI args when --config is specified
        # This might change in the future to allow overriding config file values
        if remaining_args:
            raise ValueError(
                "Cannot specify additional CLI arguments when using --config. "
                f"Found: {remaining_args}. Use either --config file or hydra CLI args, not both."
            )
        
        config_file_path = pop_config_arg()
        default_llm_config = LlmConfig()
        llm_config_from_file = OmegaConf.load(config_file_path)
        # Override defaults with values specified in the .yaml provided by --config.
        merged_llm_config = OmegaConf.merge(default_llm_config, llm_config_from_file)
        export_llama(merged_llm_config)
    else:
        hydra_main()


if __name__ == "__main__":
    main()
