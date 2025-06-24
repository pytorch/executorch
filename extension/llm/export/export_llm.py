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

from executorch.examples.models.llama.config.llm_config import LlmConfig
from executorch.examples.models.llama.export_llama_lib import export_llama
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

cs = ConfigStore.instance()
cs.store(name="llm_config", node=LlmConfig)


# Need this global variable to pass an llm_config from yaml
# into the hydra-wrapped main function.
llm_config_from_yaml = None


def parse_config_arg() -> Tuple[str, List[Any]]:
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
    global llm_config_from_yaml

    # Override the LlmConfig constructed from the provide yaml config file
    # with the CLI overrides.
    if llm_config_from_yaml:
        # Get CLI overrides (excluding defaults list).
        overrides_list: List[str] = list(HydraConfig.get().overrides.get("task", []))
        override_cfg = OmegaConf.from_dotlist(overrides_list)
        merged_config = OmegaConf.merge(llm_config_from_yaml, override_cfg)
        export_llama(merged_config)
    else:
        export_llama(OmegaConf.to_object(llm_config))


def main() -> None:
    # First parse out the arg for whether to use Hydra or the old CLI.
    config, remaining_args = parse_config_arg()
    if config:
        global llm_config_from_yaml
        # Pop out --config and its value so that they are not parsed by
        # Hyra's main.
        config_file_path = pop_config_arg()
        default_llm_config = LlmConfig()
        # Construct the LlmConfig from the config yaml file.
        default_llm_config = LlmConfig()
        from_yaml = OmegaConf.load(config_file_path)
        llm_config_from_yaml = OmegaConf.merge(default_llm_config, from_yaml)
    hydra_main()


if __name__ == "__main__":
    main()
