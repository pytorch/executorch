# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import importlib.resources
import os
import re
import shutil
import subprocess

import tempfile

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

# If this environment variable is set to true, save the flatc input files when
# serialization fails.
_SAVE_FLATC_ENV: str = "ET_EXIR_SAVE_FLATC_INPUTS_ON_FAILURE"


def _is_valid_alignment(alignment: int) -> bool:
    """Returns True if the alignment is valid, or is None."""
    if alignment is None:
        return True
    return alignment > 0 and (alignment & (alignment - 1)) == 0


def _patch_schema_alignment(
    schema: bytes,
    constant_tensor_alignment: Optional[int],
    delegate_alignment: Optional[int],
) -> bytes:
    """Modifies annotated "force_align" values in a flatbuffer schema.

    Args:
        schema: The flatbuffer schema to modify.
        constant_tensor_alignment: If provided, the alignment to use for lines annotated
            with "@executorch-tensor-alignment". If not provided, does not patch
            tensor alignment.
        delegate_alignment: If provided, the alignment to use for lines
            annotated with "@executorch-delegate-alignment". If not provided,
            does not patch delegate alignment.

    Returns:
        The possibly-modified flatbuffer schema.
    """

    def assert_valid_alignment(alignment: Optional[int], name: str) -> None:
        if not (alignment is None or _is_valid_alignment(alignment)):
            raise ValueError(f"Bad {name} {alignment}")

    assert_valid_alignment(constant_tensor_alignment, "constant_tensor_alignment")
    assert_valid_alignment(delegate_alignment, "delegate_alignment")

    def patch_alignment(line: bytes, alignment: int) -> bytes:
        """Replaces an existing alignment with a new alignment."""
        return re.sub(
            rb"\(\s*force_align\s*:\s*\d+\s*\)",
            f"(force_align: {alignment})".encode("utf-8"),
            line,
        )

    lines = []
    for line in schema.splitlines():
        if constant_tensor_alignment and b"@executorch-tensor-alignment" in line:
            lines.append(patch_alignment(line, constant_tensor_alignment))
        elif delegate_alignment and b"@executorch-delegate-alignment" in line:
            lines.append(patch_alignment(line, delegate_alignment))
        else:
            lines.append(line)
    return b"\n".join(lines)


class _SchemaMaxAlignmentGetter:
    """Finds the largest (force_align: N) N value in flatbuffer schemas."""

    def __init__(self) -> None:
        self.max_alignment: int = 0

    def __call__(self, schema: bytes) -> bytes:
        """Finds all `(force_align: N)` instances and updates max_alignment.

        Returns the input schema unmodified.
        """
        regex = re.compile(rb"\(\s*force_align\s*:\s*(\d+)\s*\)")
        matches = regex.findall(schema)
        for alignment in [int(match) for match in matches]:
            if alignment > self.max_alignment:
                self.max_alignment = alignment
        return schema


class _ResourceFiles:
    """Manages a collection of python resources that will be written to files."""

    def __init__(self, resource_names: Sequence[str]) -> None:
        """Load the resources with the provided names."""
        # Map each name to its contents.
        self._files: Dict[str, bytes] = {}
        for name in resource_names:
            self._files[name] = importlib.resources.read_binary(__package__, name)

    def patch_files(self, patch_fn: Callable[[bytes], bytes]) -> None:
        """Uses the provided patching function to update the contents of all
        files. `patch_fn` takes the current contents of a file as input and
        returns the new contents.
        """
        for name in self._files.keys():
            self._files[name] = patch_fn(self._files[name])

    def write_to(self, out_dir: str) -> None:
        """Writes the files to the specified directory. File names are based on
        the original resource names.
        """
        for name, data in self._files.items():
            with open(os.path.join(out_dir, name), "wb") as fp:
                fp.write(data)


@dataclass
class _SchemaInfo:
    # Path to a file containing the root schema. Other included schema files may
    # be present in the same directly.
    root_path: str

    # An alignment value that can satisfy all "force_align" entries found in the
    # schema files.
    max_alignment: int


def _prepare_schema(
    out_dir: str,
    constant_tensor_alignment: Optional[int] = None,
    delegate_alignment: Optional[int] = None,
) -> _SchemaInfo:
    """Returns the path to the program schema file after copying it and its deps
    into out_dir. May patch the schema contents depending on the parameters to
    this function.
    """
    program_schema = "program.fbs"
    # Included by the root program schema; must also be present.
    deps = ["scalar_type.fbs"]

    schemas = _ResourceFiles([program_schema] + deps)

    # Update annotated alignments in the schema files.
    schemas.patch_files(
        lambda data: _patch_schema_alignment(
            schema=data,
            constant_tensor_alignment=constant_tensor_alignment,
            delegate_alignment=delegate_alignment,
        ),
    )
    # Find the largest alignment used in the patched schema files.
    get_alignments = _SchemaMaxAlignmentGetter()
    schemas.patch_files(get_alignments)

    # Write the patched schema files to the filesystem.
    schemas.write_to(out_dir)

    return _SchemaInfo(
        root_path=os.path.join(out_dir, program_schema),
        max_alignment=get_alignments.max_alignment,
    )


@dataclass
class _FlatbufferResult:
    # Serialized flatbuffer data.
    data: bytes

    # The maximum "force_align" value from the schema used to serialize the data.
    max_alignment: int


# Name of an optional resource containing the `flatc` executable.
_FLATC_RESOURCE_NAME: str = "flatbuffers-flatc"


def _run_flatc(args: Sequence[str]) -> None:
    """Runs the `flatc` command with the provided args.

    If a resource matching _FLATC_RESOURCE_NAME exists, uses that executable.
    Otherwise, expects the `flatc` tool to be available on the system path.
    """
    if importlib.resources.is_resource(__package__, _FLATC_RESOURCE_NAME):
        # Use the provided flatc binary.
        with importlib.resources.path(__package__, _FLATC_RESOURCE_NAME) as flatc_path:
            subprocess.run([flatc_path] + list(args), check=True)
    else:
        # Expect the `flatc` tool to be on the system path or set as an env var.
        flatc_path = os.getenv("FLATC_EXECUTABLE")
        if not flatc_path:
            flatc_path = "flatc"
        subprocess.run([flatc_path] + list(args), check=True)


def _flatc_compile(output_dir: str, schema_path: str, json_path: str) -> None:
    """Serializes JSON data to a binary flatbuffer file.

    Args:
        output_dir: Directory under which to create the binary flatbuffer file.
        schema_path: Path to the flatbuffer schema to use for serialization.
            If the schema inclues other schema files, they must be present in
            the same directory.
        json_path: Path to the data to serialize, as JSON data whose structure
            matches the schema.
    """
    _run_flatc(
        [
            "--binary",
            "-o",
            output_dir,
            schema_path,
            json_path,
        ]
    )


def _flatc_decompile(
    output_dir: str,
    schema_path: str,
    bin_path: str,
    flatc_additional_args: Optional[List[str]] = None,
) -> None:
    """Deserializes binary flatbuffer data to a JSON file.

    Args:
        output_dir: Directory under which to create the JSON file.
        schema_path: Path to the flatbuffer schema to use for deserialization.
            If the schema inclues other schema files, they must be present in
            the same directory.
        bin_path: Path to the data to deserialize, as binary data compatible
            with the schema.
    """
    flatc_additional_args = flatc_additional_args if flatc_additional_args else []
    _run_flatc(
        flatc_additional_args
        + [
            "--json",
            "--defaults-json",
            "--strict-json",
            "-o",
            output_dir,
            schema_path,
            "--",
            bin_path,
        ]
    )


def _program_json_to_flatbuffer(
    program_json: str,
    *,
    constant_tensor_alignment: Optional[int] = None,
    delegate_alignment: Optional[int] = None,
) -> _FlatbufferResult:
    """Converts Program-compatible JSON into binary flatbuffer data.

    Args:
        program_json: The JSON to convert. Must be compatible with the root
            table type of //executorch/schema/program.fbs.
        constant_tensor_alignment: If provided, the alignment to use for tensor
            data embedded in the output flatbuffer data. If not provided, uses
            the alignment in the schema.
        delegate_alignment: If provided, the alignment to use for delegate
            data embedded in the output flatbuffer data. If not provided, uses
            the alignment in the schema.

    Returns: The flatbuffer data and associated metadata.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        schema_info = _prepare_schema(
            out_dir=temp_dir,
            constant_tensor_alignment=constant_tensor_alignment,
            delegate_alignment=delegate_alignment,
        )
        file_stem = "data"
        json_path = os.path.join(temp_dir, file_stem + ".json")
        output_path = os.path.join(temp_dir, file_stem + ".pte")

        with open(json_path, "wb") as json_file:
            json_file.write(program_json.encode("ascii"))

        try:
            _flatc_compile(temp_dir, schema_info.root_path, json_path)
        except Exception as err:
            # It's helpful to save the breaking files for debugging. Optionally
            # move them out of the auto-deleting temporary directory. Don't do
            # this by default because some input files can be many GB in size,
            # and these copies won't be auto-deleted.
            should_save = os.getenv(_SAVE_FLATC_ENV, "").strip() not in {"", "0"}
            extra_message = ""
            if should_save:
                try:
                    saved_dir = tempfile.mkdtemp(prefix="exir-saved-flatc-")
                    for f in os.listdir(temp_dir):
                        shutil.move(src=os.path.join(temp_dir, f), dst=saved_dir)
                    extra_message += f" Moved input files to '{saved_dir}'."
                except Exception as err2:
                    extra_message += (
                        f" (Failed to save input files for debugging: {err2})"
                    )
            else:
                extra_message += (
                    f" Set {_SAVE_FLATC_ENV}=1 to save input files on failure."
                )

            raise RuntimeError(
                f"Failed to compile {json_path} to {output_path}." + extra_message
            ) from err
        with open(output_path, "rb") as output_file:
            return _FlatbufferResult(
                data=output_file.read(), max_alignment=schema_info.max_alignment
            )


def _replace_infinity_in_json_file(content: bytes) -> bytes:
    """Replace -inf and inf with "inf" and "-inf" in the JSON file. program.fbs
    is used to convert from flatbuffer to JSON. +-inf float values are not
    supported by JSON, so we replace them with the string equivalent. When
    converting from JSON to python dataclasses, the string is read as a Union
    of float and string (see schema.py).
    """
    content = re.sub(
        rb'"double_val"\s*:\s*(-)?inf', rb'"double_val": "\g<1>inf"', content
    )
    return content


def _program_flatbuffer_to_json(program_flatbuffer: bytes) -> bytes:
    """Converts binary flatbuffer data into Program-compatible JSON.

    The binary is parsed using the schema in //executorch/schema/program.fbs.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # No need to patch the alignment when reading. "force_align" is only
        # used during serialization.
        schema_info = _prepare_schema(temp_dir)
        file_stem = "data"
        bin_path = os.path.join(temp_dir, file_stem + ".bin")
        json_path = os.path.join(temp_dir, file_stem + ".json")

        with open(bin_path, "wb") as bin_file:
            bin_file.write(program_flatbuffer)

        _flatc_decompile(temp_dir, schema_info.root_path, bin_path)
        with open(json_path, "rb") as output_file:
            json_data = output_file.read()
            return _replace_infinity_in_json_file(json_data)
