# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile

from dataclasses import dataclass
from enum import IntEnum, unique

from importlib.resources import files
from typing import Dict, Optional

from executorch.exir._serialize._dataclass import _DataclassEncoder, _json_to_dataclass
from executorch.exir._serialize._flatbuffer import _flatc_compile, _flatc_decompile
from executorch.exir._warnings import experimental
from executorch.exir.backend.backend_details import CompileSpec


@unique
class SamsungChipset(IntEnum):
    UNDEFINED_CHIP_V = 0
    E9955 = 9955
    E9965 = 9965


@experimental(
    "This API is experimental. If you use this mode, you should verify pte file on device farm which can be used on "
    "exynos developer society site ( https://soc-developer.semiconductor.samsung.com/)"
)
@unique
class PerformanceMode(IntEnum):
    DEFAULT = 0
    HIGH_PERFORMANCE = 1


@unique
class WeightSharingFlag(IntEnum):
    WEIGHT_SHARING_NONE = 0
    WEIGHT_SHARING_GEN = 1
    WEIGHT_SHARING_USE = 2


@dataclass
class EnnExecuTorchOptions:
    chipset: SamsungChipset = SamsungChipset.UNDEFINED_CHIP_V
    perf_mode: PerformanceMode = PerformanceMode.DEFAULT
    weight_sharing_flag: WeightSharingFlag = WeightSharingFlag.WEIGHT_SHARING_NONE


ENN_COMPILE_OPTION_TITLE = "enn_compile_options"
ENN_COMPILE_WEIGHT_BUFFER_HASH_ID = "enn_weight_buffer_hash_id"
COMPILE_OPTION_SCHEMA_NAME = "compile_options_def"


def gen_samsung_backend_compile_spec_core(options: EnnExecuTorchOptions) -> CompileSpec:
    with tempfile.TemporaryDirectory() as d:
        # schema
        schema_path = os.path.join(d, "{}.fbs".format(COMPILE_OPTION_SCHEMA_NAME))

        schema_content = (
            files(__package__)
            .joinpath(f"{COMPILE_OPTION_SCHEMA_NAME}.fbs")
            .read_bytes()
        )

        with open(schema_path, "wb") as schema_file:
            schema_file.write(schema_content)
        # dump json
        json_path = os.path.join(d, "{}.json".format(COMPILE_OPTION_SCHEMA_NAME))
        enn_options_json = json.dumps(options, cls=_DataclassEncoder, indent=4)
        with open(json_path, "wb") as json_file:
            json_file.write(enn_options_json.encode("ascii"))

        _flatc_compile(d, schema_path, json_path)
        output_path = os.path.join(d, "{}.eeto".format(COMPILE_OPTION_SCHEMA_NAME))
        with open(output_path, "rb") as output_file:
            return CompileSpec(ENN_COMPILE_OPTION_TITLE, output_file.read())


def gen_samsung_backend_compile_spec(
    chipset: str,
    perf_mode: Optional[PerformanceMode] = None,
    weight_sharing_flag: Optional[WeightSharingFlag] = None,
):
    """
    A function to generate an ExecuTorch binary for Samsung Backend.

    Attributes:
        chipset (str): chipset name in SamsungChipset. For example, E9955 or E9965 (case-insensitive).

    Returns:
        CompileSpec: key is COMPILE_OPTION_SCHEMA_NAME, value is serialization binary of fb schema
    """

    perf_mode = PerformanceMode.DEFAULT if perf_mode is None else perf_mode
    weight_sharing_flag = (
        WeightSharingFlag.WEIGHT_SHARING_NONE
        if weight_sharing_flag is None
        else weight_sharing_flag
    )

    option = EnnExecuTorchOptions(
        getattr(SamsungChipset, chipset.upper()),
        perf_mode,
        weight_sharing_flag,
    )

    return gen_samsung_backend_compile_spec_core(option)


def gen_samsung_backend_compile_weight_spec():
    return CompileSpec(ENN_COMPILE_WEIGHT_BUFFER_HASH_ID, b"")


def _convert_int_enum_values(json_dict: Dict, cls: type) -> Dict:
    """
    Convert integer values to enum names for IntEnum fields in the JSON dict.

    This is needed because _json_to_dataclass expects enum names (strings) but
    the serialized JSON contains integer values for IntEnum fields.

    Args:
        json_dict: The JSON dictionary loaded from the deserialized data.
        cls: The target dataclass type.

    Returns:
        Modified JSON dictionary with IntEnum values converted to names.
    """
    from dataclasses import fields
    from typing import get_args, get_origin, Union

    result = {}
    for field in fields(cls):
        key = field.name
        if key not in json_dict:
            continue

        value = json_dict[key]
        T = field.type

        # Handle Optional types
        if get_origin(T) is Union:
            args = get_args(T)
            if len(args) > 0 and args[-1] is type(None):
                T = args[0]

        # Check if this is an IntEnum field
        if isinstance(T, type) and issubclass(T, IntEnum):
            # Convert integer value to enum name
            if isinstance(value, int):
                try:
                    result[key] = T(value).name
                except ValueError:
                    # If the integer doesn't match any enum value, keep it as-is
                    result[key] = value
            else:
                result[key] = value
        else:
            result[key] = value

    return result


def deserialize_compile_options(serialized_data: bytes) -> EnnExecuTorchOptions:
    """
    Deserialize FlatBuffers binary data back to EnnExecuTorchOptions dataclass.

    This function reverses the serialization process performed by
    gen_samsung_backend_compile_spec_core(), converting the binary
    FlatBuffers format back to a Python dataclass.

    Args:
        serialized_data: The FlatBuffers serialized binary bytes.

    Returns:
        EnnExecuTorchOptions dataclass containing the deserialized options.
    """
    with tempfile.TemporaryDirectory() as d:
        schema_filename = f"{COMPILE_OPTION_SCHEMA_NAME}.fbs"
        binary_filename = f"{COMPILE_OPTION_SCHEMA_NAME}.bin"
        json_filename = f"{COMPILE_OPTION_SCHEMA_NAME}.json"

        schema_path = os.path.join(d, schema_filename)
        binary_path = os.path.join(d, binary_filename)
        json_path = os.path.join(d, json_filename)

        # Write the FlatBuffers schema file from package resources
        schema_bytes = files(__package__).joinpath(schema_filename).read_bytes()
        with open(schema_path, "wb") as f:
            f.write(schema_bytes)

        # Write the serialized binary data
        with open(binary_path, "wb") as f:
            f.write(serialized_data)

        # Execute flatc decompilation to convert binary to JSON
        _flatc_decompile(d, schema_path, binary_path, ["--raw-binary"])

        # Read the generated JSON and convert to dataclass
        with open(json_path, "rb") as f:
            json_dict = json.load(f)
            # Convert IntEnum integer values to enum names before deserialization
            json_dict = _convert_int_enum_values(json_dict, EnnExecuTorchOptions)
            return _json_to_dataclass(json_dict, EnnExecuTorchOptions)


def get_weight_sharing_flag_from_compile_spec(
    option_spec: CompileSpec,
) -> WeightSharingFlag:
    """
    Extract the weight_sharing_flag value from a CompileSpec.

    Args:
        option_spec: CompileSpec with key ENN_COMPILE_OPTION_TITLE
                     containing serialized EnnExecuTorchOptions.

    Returns:
        WeightSharingFlag enum value indicating the weight sharing mode.

    Raises:
        ValueError: If the CompileSpec does not contain valid data.
    """
    if not option_spec or not option_spec.value:
        raise ValueError("CompileSpec value is empty")

    options = deserialize_compile_options(option_spec.value)
    return options.weight_sharing_flag
