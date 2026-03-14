# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import importlib.resources
import os
import re
import subprocess

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


class _SchemaFileIdentifierGetter:
    """Finds the file_identifier value in flatbuffer schemas."""

    def __init__(self) -> None:
        self.file_identifier: Optional[bytes] = None

    def __call__(self, schema: bytes) -> bytes:
        identifiers = re.findall(rb'file_identifier\s+"([^"]+)"', schema)
        for file_identifier in identifiers:
            if len(file_identifier) != 4:
                raise ValueError(
                    f"Invalid file_identifier length {len(file_identifier)} in schema"
                )
            if self.file_identifier is None:
                self.file_identifier = file_identifier
            elif self.file_identifier != file_identifier:
                raise ValueError(
                    f"Mismatched file_identifier {file_identifier} != {self.file_identifier}"
                )
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

    def get(self, name: str) -> bytes:
        """Returns the current contents of the named file."""
        return self._files[name]

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

    # The file identifier declared in the root schema.
    file_identifier: bytes

    # The alignment for constant tensor data in the schema.
    tensor_alignment: int

    # The alignment for delegate data in the schema.
    delegate_alignment: int


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
    get_file_identifier = _SchemaFileIdentifierGetter()
    schemas.patch_files(get_file_identifier)
    if get_file_identifier.file_identifier is None:
        raise ValueError("Missing file_identifier in schema files.")

    def extract_alignment(schema: bytes, marker: bytes) -> int:
        for line in schema.splitlines():
            if marker in line:
                match = re.search(rb"force_align\s*:\s*(\d+)", line)
                if match:
                    return int(match.group(1))
        raise RuntimeError(f"Failed to find marker {marker!r} in program.fbs")

    program_schema_data = schemas.get(program_schema)
    tensor_alignment = extract_alignment(
        program_schema_data, b"@executorch-tensor-alignment"
    )
    effective_delegate_alignment = extract_alignment(
        program_schema_data, b"@executorch-delegate-alignment"
    )

    # Write the patched schema files to the filesystem.
    schemas.write_to(out_dir)

    return _SchemaInfo(
        root_path=os.path.join(out_dir, program_schema),
        max_alignment=get_alignments.max_alignment,
        file_identifier=get_file_identifier.file_identifier,
        tensor_alignment=tensor_alignment,
        delegate_alignment=effective_delegate_alignment,
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
    flatc_resource = importlib.resources.files(__package__).joinpath(
        _FLATC_RESOURCE_NAME
    )
    if flatc_resource.is_file():
        # Use the provided flatc binary.
        with importlib.resources.as_file(flatc_resource) as flatc_path:
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
