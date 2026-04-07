#!/usr/bin/env python3

# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from executorch.backends.arm.ethosu import EthosUBackend, EthosUCompileSpec
from executorch.devtools.pte_tool.pte_info import (
    DelegateInfo,
    get_delegate_infos_from_pte,
)


@dataclass(frozen=True)
class EthosUDelegateConfig:
    target: str
    system_config: str
    memory_mode: str


def _extract_flag_value(flags: list[str], prefix: str) -> str | None:
    for flag in flags:
        if flag.startswith(prefix):
            return flag.removeprefix(prefix)
    return None


def _config_from_delegate(delegate_info: DelegateInfo) -> EthosUDelegateConfig:
    compile_spec = EthosUCompileSpec._from_list(delegate_info.compile_specs)
    if compile_spec.target is None:
        raise ValueError("Missing Ethos-U target in delegate compile spec.")

    system_config = _extract_flag_value(compile_spec.compiler_flags, "--system-config=")
    if system_config is None:
        raise ValueError(
            f"Missing --system-config flag in Ethos-U compile spec for {compile_spec.target}."
        )

    memory_mode = _extract_flag_value(compile_spec.compiler_flags, "--memory-mode=")
    if memory_mode is None:
        raise ValueError(
            f"Missing --memory-mode flag in Ethos-U compile spec for {compile_spec.target}."
        )

    return EthosUDelegateConfig(
        target=compile_spec.target,
        system_config=system_config,
        memory_mode=memory_mode,
    )


def get_ethosu_delegate_configs_from_pte(
    pte_path: str | Path,
) -> list[EthosUDelegateConfig]:
    configs: list[EthosUDelegateConfig] = []
    for delegate_info in get_delegate_infos_from_pte(pte_path):
        if delegate_info.delegate_id != EthosUBackend.__name__:
            continue
        configs.append(_config_from_delegate(delegate_info))
    return configs


def get_ethosu_delegate_config_from_pte(
    pte_path: str | Path,
) -> EthosUDelegateConfig | None:
    configs = get_ethosu_delegate_configs_from_pte(pte_path)
    if not configs:
        return None

    unique_configs = sorted(
        {
            (config.target, config.system_config, config.memory_mode)
            for config in configs
        }
    )
    if len(unique_configs) != 1:
        joined = ", ".join(
            f"target={target} system_config={system_config} memory_mode={memory_mode}"
            for target, system_config, memory_mode in unique_configs
        )
        raise ValueError(
            "Found multiple Ethos-U delegate compile spec configurations in "
            f"{pte_path}: {joined}"
        )

    target, system_config, memory_mode = unique_configs[0]
    return EthosUDelegateConfig(
        target=target,
        system_config=system_config,
        memory_mode=memory_mode,
    )


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read the Ethos-U delegate compile spec from a .pte/.bpte file and "
            "print the resolved target, system_config and memory_mode."
        )
    )
    parser.add_argument("pte_path", help="Path to a .pte or .bpte file.")
    parser.add_argument(
        "--format",
        choices=("pretty", "json", "tsv"),
        default="pretty",
        help="Output format. Default: %(default)s.",
    )
    return parser.parse_args()


def _print_config(config: EthosUDelegateConfig, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(asdict(config), sort_keys=True))
        return
    if output_format == "tsv":
        print(f"{config.target}\t{config.system_config}\t{config.memory_mode}")
        return

    print(f"target={config.target}")
    print(f"system_config={config.system_config}")
    print(f"memory_mode={config.memory_mode}")


def main() -> int:
    args = _get_args()
    config = get_ethosu_delegate_config_from_pte(args.pte_path)
    if config is None:
        return 2

    _print_config(config, args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
