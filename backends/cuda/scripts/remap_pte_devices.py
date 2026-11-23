# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rewrite the CUDA device ids recorded in a PTE so it loads on other GPUs.

The ids come from the `target_device` compile spec (default
`cuda:0`, so only an explicit non-zero spec records one) and land in
three places that must agree: the planned-memory table, tensor
`extra_tensor_info`, and the spec itself. This tool rewrites all
three with one mapping -- dense by default (3, 5, 7 become 0, 1, 2)
-- and warns whenever the result is not entirely on cuda:0, since
the runtime currently runs every delegate on the current device
(pytorch/executorch#24890). Distinct-preserving mappings load like a
native export; collapsing mappings keep the bounces and input
copies a native single-device export would elide. Re-serialization
uses the default alignments, so non-default-alignment exports may
be repacked. See the README section in examples/cuda for the longer
discussion. The companion .ptd needs no changes: tensor data
carries no device ids.

Usage:
    python -m executorch.backends.cuda.scripts.remap_pte_devices \
        --input-pte model.pte --output-pte model_dense.pte
    python -m executorch.backends.cuda.scripts.remap_pte_devices \
        --input-pte model.pte --output-pte model_cuda0.pte \
        --mapping 3=0,5=0,7=0
    python -m executorch.backends.cuda.scripts.remap_pte_devices \
        --input-pte model.pte --dry-run
"""

import argparse
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Set, Tuple, Union

from executorch.backends.cuda.scripts._atomic_output import write_atomically
from executorch.exir._serialize._program import (
    deserialize_pte_binary,
    PTEFile,
    serialize_pte_binary,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.passes.propagate_device_pass import (
    _parse_device_spec_value,
    TARGET_DEVICE_COMPILE_SPEC_KEY,
)
from executorch.exir.schema import (
    DeviceType,
    ExtraTensorInfo,
    NonConstBufferDevice,
    Program,
    Tensor,
)

# device_index is a flatbuffer byte in both places it is stored
# (schema/program.fbs), so no recorded or target index can exceed it.
_MAX_DEVICE_INDEX = 127

_DeviceSite = Union[NonConstBufferDevice, ExtraTensorInfo, CompileSpec]


@dataclass(frozen=True)
class DeviceRemapReport:
    """What one remap changed, per place device ids live."""

    mapping: Tuple[Tuple[int, int], ...]
    planned_buffers_changed: int
    tensor_infos_changed: int
    compile_specs_changed: int


def _parse_target_device(value: bytes) -> Optional[int]:
    """The CUDA index a target_device compile spec names.

    Returns None for a spec that names a non-CUDA device, which is not
    this tool's to move; an unrecognized device type (another
    backend's spec that happens to share the key) counts as non-CUDA.
    Malformed CUDA values raise, exactly as they do in the pass that
    derived placement from the spec at export.
    """
    type_name = value.decode("utf-8", errors="replace").strip().lower()
    if type_name.split(":", 1)[0] not in {dt.name.lower() for dt in DeviceType}:
        return None
    device_type, device_index = _parse_device_spec_value(value)
    if device_type != DeviceType.CUDA:
        return None
    return device_index


def _cuda_device_sites(program: Program) -> Iterator[_DeviceSite]:
    """Every object that records a CUDA device index, in one walk.

    Collecting and rewriting share this walk so the two can never
    disagree about where device ids live.
    """
    for plan in program.execution_plan:
        for buffer_device in plan.non_const_buffer_device or []:
            if buffer_device.device_type == DeviceType.CUDA:
                yield buffer_device
        for evalue in plan.values:
            tensor = evalue.val
            if not isinstance(tensor, Tensor):
                continue
            info = tensor.extra_tensor_info
            if info is not None and info.device_type == DeviceType.CUDA:
                yield info
        for delegate in plan.delegates:
            for spec in delegate.compile_specs:
                if spec.key != TARGET_DEVICE_COMPILE_SPEC_KEY:
                    continue
                if _parse_target_device(spec.value) is not None:
                    yield spec


def _site_device_index(site: _DeviceSite) -> int:
    if isinstance(site, CompileSpec):
        index = _parse_target_device(site.value)
        assert index is not None  # filtered by _cuda_device_sites
        return index
    return site.device_index


def _set_site_device_index(site: _DeviceSite, new_index: int) -> None:
    if isinstance(site, CompileSpec):
        site.value = f"cuda:{new_index}".encode()
    else:
        site.device_index = new_index


def collect_cuda_device_indices(program: Program) -> Set[int]:
    """Every CUDA device index recorded anywhere in the program."""
    return {_site_device_index(site) for site in _cuda_device_sites(program)}


def dense_device_mapping(indices: Set[int]) -> Dict[int, int]:
    """Map sorted device indices to 0..n-1, keeping their order."""
    return {old: new for new, old in enumerate(sorted(indices))}


def parse_device_mapping(text: str) -> Dict[int, int]:
    """Parse an explicit "3=0,5=1" mapping."""
    mapping: Dict[int, int] = {}
    for pair in text.split(","):
        pair = pair.strip()
        if not pair:
            continue
        old, _, new = pair.partition("=")
        try:
            source, target = int(old), int(new)
        except ValueError:
            raise ValueError(
                f"device mapping entry {pair!r} is not of the form OLD=NEW"
            ) from None
        if source in mapping:
            raise ValueError(f"device mapping repeats source device {source}: {text!r}")
        mapping[source] = target
    if not mapping:
        raise ValueError("device mapping is empty")
    return mapping


def remap_program_devices(
    program: Program, mapping: Dict[int, int]
) -> DeviceRemapReport:
    """Apply one device mapping to every place the program records one.

    Raises ValueError if the mapping names a target outside 0..=127
    (device_index is stored in a byte), if it misses a recorded
    device, or if it names one the program does not record, so a file
    is never rewritten only in part and a mapping meant for another
    file fails loudly.
    """
    if any(not 0 <= target <= _MAX_DEVICE_INDEX for target in mapping.values()):
        raise ValueError(
            f"device mapping targets must be within 0..={_MAX_DEVICE_INDEX}: "
            f"{sorted(mapping.items())}"
        )
    # Several delegates can share one CompileSpec object in an
    # emitted in-memory program; rewrite each object only once.
    sites = list({id(site): site for site in _cuda_device_sites(program)}.values())
    recorded = {_site_device_index(site) for site in sites}
    missing = recorded - mapping.keys()
    if missing:
        raise ValueError(
            f"device mapping does not cover cuda device(s) {sorted(missing)} "
            f"recorded in the program; mapping covers {sorted(mapping)}"
        )
    unknown = mapping.keys() - recorded
    if unknown:
        raise ValueError(
            f"device mapping names cuda device(s) {sorted(unknown)} that "
            f"the program does not record; it records {sorted(recorded)}"
        )

    changed = {"buffers": 0, "tensors": 0, "specs": 0}
    for site in sites:
        old_index = _site_device_index(site)
        new_index = mapping[old_index]
        if new_index == old_index:
            continue
        _set_site_device_index(site, new_index)
        if isinstance(site, NonConstBufferDevice):
            changed["buffers"] += 1
        elif isinstance(site, ExtraTensorInfo):
            changed["tensors"] += 1
        else:
            changed["specs"] += 1

    return DeviceRemapReport(
        mapping=tuple(sorted(mapping.items())),
        planned_buffers_changed=changed["buffers"],
        tensor_infos_changed=changed["tensors"],
        compile_specs_changed=changed["specs"],
    )


def _remap_pte_file(
    pte_file: PTEFile,
    mapping: Optional[Dict[int, int]],
    warn_stacklevel: int = 3,
) -> DeviceRemapReport:
    """Remap a deserialized PTE in place, dense by default.

    Warns whenever the result is not entirely on cuda:0, with the
    dense default or an explicit mapping alike.
    """
    indices = collect_cuda_device_indices(pte_file.program)
    if not indices:
        raise ValueError("the program records no CUDA device ids to remap")
    if mapping is None:
        mapping = dense_device_mapping(indices)
    report = remap_program_devices(pte_file.program, mapping)
    targets = {new for _, new in report.mapping}
    if targets != {0}:
        warnings.warn(
            "this program does not land entirely on cuda:0 after "
            f"remapping (targets: {sorted(targets)}); every remapped "
            "device must exist on the target machine, and the CUDA "
            "runtime currently runs every delegate on the current "
            "device (see pytorch/executorch#24890). Collapse them onto "
            "cuda:0 with an explicit mapping (for example "
            + ",".join(f"{device}=0" for device in sorted(indices))
            + ") to run everything on one GPU.",
            UserWarning,
            stacklevel=warn_stacklevel,
        )
    return report


def remap_pte_bytes(
    pte_data: bytes, mapping: Optional[Dict[int, int]] = None
) -> Tuple[bytes, DeviceRemapReport]:
    """Remap one PTE's device ids, returning the new bytes and a report.

    With no mapping, the dense mapping of the devices found in the
    file is used. Emits a warning whenever the result is not
    entirely on cuda:0.
    """
    pte_file = deserialize_pte_binary(pte_data)
    report = _remap_pte_file(pte_file, mapping)
    return (
        bytes(serialize_pte_binary(pte_file, extract_delegate_segments=True)),
        report,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite the CUDA device ids recorded in a PTE (planned-memory "
            "devices, tensor devices, and delegate target_device specs) with "
            "one mapping, dense by default"
        )
    )
    parser.add_argument("--input-pte", required=True, type=Path)
    parser.add_argument("--output-pte", type=Path)
    parser.add_argument(
        "--mapping",
        help='explicit device mapping like "3=0,5=1"; sources may share a '
        "target to collapse everything onto one device; default compresses "
        "the device ids found in the file to 0..n-1",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the mapping and what would change, without writing",
    )
    return parser


def main() -> None:
    """Command-line entry point for CUDA PTE device remapping."""
    parser = _build_parser()
    args = parser.parse_args()
    if not args.dry_run and args.output_pte is None:
        parser.error("--output-pte is required unless --dry-run is given")
    if args.dry_run and args.output_pte is not None:
        parser.error("--output-pte is not written with --dry-run; drop one")
    try:
        mapping = (
            parse_device_mapping(args.mapping) if args.mapping is not None else None
        )
        pte_file = deserialize_pte_binary(args.input_pte.read_bytes())
        report = _remap_pte_file(pte_file, mapping, warn_stacklevel=2)
        # Serialize here too, so a serialization failure is reported the
        # same way and --dry-run exercises it instead of deferring it to
        # a write that never happens.
        serialized = serialize_pte_binary(pte_file, extract_delegate_segments=True)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))

    print(f"CUDA device mapping: {dict(report.mapping)}")
    print(
        f"changed: {report.planned_buffers_changed} planned buffer(s), "
        f"{report.tensor_infos_changed} tensor info(s), "
        f"{report.compile_specs_changed} delegate target_device spec(s)"
    )
    if args.dry_run:
        print("dry run: no file written")
        return

    try:
        write_atomically(
            args.output_pte,
            serialized.write_to_file,
            mode_from=args.input_pte,
        )
    except OSError as error:
        parser.error(str(error))
    print(f"wrote {args.output_pte}")


if __name__ == "__main__":
    main()
