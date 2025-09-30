# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.resources
import json
import os
import tempfile

from dataclasses import dataclass
from enum import IntEnum, unique

from executorch.exir._serialize._dataclass import _DataclassEncoder
from executorch.exir._serialize._flatbuffer import _flatc_compile
from executorch.exir.backend.backend_details import CompileSpec


@unique
class SamsungChipset(IntEnum):
    UNDEFINED_CHIP_V = 0
    E9955 = 9955


@dataclass
class EnnExecuTorchOptions:
    chipset: SamsungChipset = SamsungChipset.UNDEFINED_CHIP_V


ENN_COMPILE_OPTION_TITLE = "enn_compile_options"
COMPILE_OPTION_SCHEMA_NAME = "compile_options_def"


def gen_samsung_backend_compile_spec_core(options: EnnExecuTorchOptions) -> CompileSpec:
    with tempfile.TemporaryDirectory() as d:
        # schema
        schema_name = f"{COMPILE_OPTION_SCHEMA_NAME}.fbs"
        schema_path = os.path.join(d, schema_name)
        resource_file = importlib.resources.files(__package__) / f"{schema-name}"
        schema_content = resource_file.read_bytes()
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
):
    """
    A function to generate an ExecuTorch binary for Samsung Backend.

    Attributes:
        chipset (str): chipset name in SamsungChipset. For example, E9955 or e9955 both work.

    Returns:
        CompileSpec: key is COMPILE_OPTION_SCHEMA_NAME, value is serialization binary of fb schema
    """
    option = EnnExecuTorchOptions(
        getattr(SamsungChipset, chipset.upper()),
    )

    return gen_samsung_backend_compile_spec_core(option)
