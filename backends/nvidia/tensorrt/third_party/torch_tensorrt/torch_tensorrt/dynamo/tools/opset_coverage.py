import dataclasses
import json
import os
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch._prims as prims
import torchgen
from torch._dynamo.variables import BuiltinVariable
from torch._ops import OpOverload
from torch_tensorrt.dynamo.conversion import DYNAMO_CONVERTERS, ConverterRegistry
from torch_tensorrt.dynamo.lowering import get_decompositions
from torchgen.gen import parse_native_yaml


class SupportStatus(Enum):
    CONVERTED = auto()
    LEGACY_CONVERTED = auto()
    LOWERED = auto()
    FALLBACK = auto()

    def __str__(self) -> str:
        return self.name


@dataclass
class OpsetCoverage:
    support_status: Dict[str, Dict[str, str]]
    dynamo_coverage: float
    legacy_coverage: float
    decomposition_coverage: float
    fallback_coverage: float


NATIVE_FUNCTION_YAML_PATH = (
    Path(os.path.dirname(torchgen.__file__))
    / "packaged/ATen/native/native_functions.yaml"
)
TAGS_YAML_PATH = (
    Path(os.path.dirname(torchgen.__file__)) / "packaged/ATen/native/tags.yaml"
)

DYNAMO_REGISTRY_NAME = "Dynamo ATen Converters Registry"
FX_REGISTRY_NAME = "FX ATen Converters Registry"
FX_LEGACY_REGISTRY_NAME = "FX Legacy ATen Converters Registry"


def get_aten_ops() -> List[Tuple[str, str]]:
    parsed_yaml = parse_native_yaml(NATIVE_FUNCTION_YAML_PATH, TAGS_YAML_PATH)
    native_functions = parsed_yaml.native_functions

    aten_ops = OrderedDict()
    for function in native_functions:
        if "core" in function.tags:
            op_name = str(function.func.name)
            aten_ops[op_name] = function

    op_schema_pairs = []
    for key, op in sorted(aten_ops.items()):
        op_name = f"aten.{key}"
        schema = str(op.func).replace("*", r"\*")

        op_schema_pairs.append((op_name, schema))

    return op_schema_pairs


ATEN_OPS = get_aten_ops()


def get_prims_ops() -> List[Tuple[str, str]]:
    op_schema_pairs = []
    for op_name in prims.__all__:
        op_overload = getattr(prims, op_name, None)

        if not isinstance(op_overload, torch._ops.OpOverload):
            continue

        op_overloadpacket = op_overload.overloadpacket

        op_name = str(op_overload).replace(".default", "")
        schema = op_overloadpacket.schema.replace("*", r"\*")

        op_schema_pairs.append((op_name, schema))

    return op_schema_pairs


PRIM_OPS = get_prims_ops()


def get_overloaded_py_ops() -> List[Tuple[str, str]]:
    python_ops = BuiltinVariable._fx_graph_functions()
    op_schema_pairs = []
    for op in python_ops:
        name = op.__name__
        op_schema_pairs.append((f"_operator.{name}", ""))

    return op_schema_pairs


OVERLOADED_PY_OPS = get_overloaded_py_ops()


def opset_coverage(
    opset: List[Tuple[str, str]],
    converter_registry: Optional[ConverterRegistry] = None,
    decomposition_registry: Optional[Dict[OpOverload, Callable[..., Any]]] = None,
) -> OpsetCoverage:
    opset_schemas = dict(opset)
    opset_targets = set(opset_schemas.keys())

    support_status = {}

    # TODO: Could be way less complicated if there is a way to convert from
    # strings to OpOverload
    c_registry = (
        converter_registry if converter_registry is not None else DYNAMO_CONVERTERS
    )
    converter_registry_targets = {
        c_registry.qualified_name_or_str(target)
        .removeprefix("torch.ops.")
        .replace(".default", "")
        for target in c_registry.keys()
    }
    supported_converted_targets = opset_targets.intersection(converter_registry_targets)
    support_count = 0
    legacy_count = 0
    for target in c_registry.keys():
        target_str = (
            c_registry.qualified_name_or_str(target)
            .removeprefix("torch.ops.")
            .replace(".default", "")
        )
        if target_str in opset_targets:
            _, registry_data = c_registry.get_all_converters_with_target(
                target, return_registry_info=True
            )

            if registry_data is not None:
                if (
                    DYNAMO_REGISTRY_NAME in registry_data
                    and registry_data[DYNAMO_REGISTRY_NAME] >= 1
                ):
                    status = SupportStatus.CONVERTED
                    support_count += 1
                elif (
                    FX_REGISTRY_NAME in registry_data
                    and registry_data[FX_REGISTRY_NAME] >= 1
                ) or (
                    FX_LEGACY_REGISTRY_NAME in registry_data
                    and registry_data[FX_LEGACY_REGISTRY_NAME] >= 1
                ):
                    status = SupportStatus.LEGACY_CONVERTED
                    legacy_count += 1
                else:
                    raise Exception(f"Op belongs to unknown registry: {registry_data}")

                support_status[target_str] = {
                    "schema": f"{target_str.split('.')[0]}.{opset_schemas[target_str]}",
                    "status": str(status),
                }
            else:
                warnings.warn(f"No registry data for op: {target_str}")

    l_registry = (
        decomposition_registry
        if decomposition_registry is not None
        else get_decompositions()
    )
    decomp_registry_targets = {
        c_registry.qualified_name_or_str(target)
        .removeprefix("torch.ops.")
        .replace(".default", "")
        for target in l_registry.keys()
    }
    supported_decomp_targets = opset_targets.intersection(decomp_registry_targets)
    decomposition_count = len(supported_decomp_targets)
    for target in supported_decomp_targets:
        support_status[target] = {
            "schema": f"{target.split('.')[0]}.{opset_schemas[target]}",
            "status": str(SupportStatus.LOWERED),
        }

    unsupported_targets = opset_targets.difference(
        supported_converted_targets.union(supported_decomp_targets)
    )
    unsupported_count = len(unsupported_targets)
    for target in unsupported_targets:
        support_status[target] = {
            "schema": f"{target.split('.')[0]}.{opset_schemas[target]}",
            "status": str(SupportStatus.FALLBACK),
        }

    return OpsetCoverage(
        support_status,
        dynamo_coverage=support_count / len(opset),
        legacy_coverage=legacy_count / len(opset),
        decomposition_coverage=decomposition_count / len(opset),
        fallback_coverage=unsupported_count / len(opset),
    )


def get_coverage_status(opset: List[Tuple[str, str]], name: str) -> OpsetCoverage:
    coverage = opset_coverage(opset)
    return coverage


ATEN_COVERAGE = get_coverage_status(ATEN_OPS, "ATen")
PRIMS_COVERAGE = get_coverage_status(PRIM_OPS, "prim")
PY_OVERLOAD_COVERAGE = get_coverage_status(OVERLOADED_PY_OPS, "py_overload")

if __name__ == "__main__":

    def find_coverage_status(opset: List[Tuple[str, str]], name: str) -> None:
        coverage = opset_coverage(opset)
        print(f"{name}:")
        print(f"    - Dynamo converters: {coverage.dynamo_coverage:.2%}")
        print(f"    - Decomposed: {coverage.decomposition_coverage:.2%}")
        print(f"    - Legacy FX converters: {coverage.legacy_coverage:.2%}")
        print(f"    - Ops to fallback to Torch: {coverage.fallback_coverage:.2%}")
        print(
            f"Per op coverage status saved to /tmp/{name.lower()}_coverage_status.json"
        )

        with open(f"/tmp/{name.lower()}_coverage_status.json", "w") as f:
            json.dump(dataclasses.asdict(coverage), f)

    print("-------- OPERATOR SET COVERAGE --------")
    find_coverage_status(ATEN_OPS, "ATen")
    find_coverage_status(PRIM_OPS, "prim")
    find_coverage_status(OVERLOADED_PY_OPS, "py_overload")
