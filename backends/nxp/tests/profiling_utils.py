# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
import subprocess

from executorch.devtools.etdump.serialize import deserialize_from_etdump_flatcc


def get_neutron_driver_version(etdump_path: str) -> str:
    """
    Extract the Neutron Driver version from an ETDump file.

    The Neutron Driver version is stored in the metadata of the last Neutron
    delegate event. This event is emitted when the profiling dump is generated.
    The version is encoded as a 16-bit value in little-endian format:
    - 4 bits  - major version
    - 4 bits  - minor version
    - 4 bits  - patch version
    - 4 bits  - reserved

    :param etdump_path: Path to the ETDump binary file.
    :return: Neutron Driver version string (e.g. "1.2.3") if successfully decoded,
    otherwise empty string. Errors are logged instead of raised.
    """

    try:
        with open(etdump_path, "rb") as f:
            data = f.read()
        etdump = deserialize_from_etdump_flatcc(data)
    except Exception as e:
        logging.exception("Failed to load ETDump: %s", e)
        return ""

    events = []
    try:
        for run in etdump.run_data:
            for event in run.events:
                profile_event = getattr(event, "profile_event", None)
                if (
                    profile_event is not None
                    and getattr(profile_event, "delegate_debug_id_int", 0) > 0
                ):
                    events.append(event)
    except Exception as e:
        logging.exception("Failed while processing events: %s", e)
        return ""

    try:
        metadata = events[-1].profile_event.delegate_debug_metadata
        if not metadata or len(metadata) < 2:
            logging.error("Invalid delegate_debug_metadata")
            return ""

        major, minor, patch = [
            (int.from_bytes(metadata, "little") >> shift) & 0xF for shift in (8, 4, 0)
        ]
        return f"{major}.{minor}.{patch}"

    except Exception as e:
        logging.exception("Failed to extract version from metadata: %s", e)
        return ""


def get_neutron_converter_version() -> str:
    """
    Get the Neutron Converter version reported by the neutron_converter tool.

    Executes `neutron_converter --version` and returns the version as
    {major}.{minor}.{patch} string.

    :return: The version string returned by neutron_converter, or empty string if
    the command fails, times out, or the executable is not available.
    Errors are logged instead of being raised.
    """

    try:
        proc = subprocess.Popen(
            ["neutron_converter", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate(timeout=10)
        if proc.returncode != 0:
            logging.error(
                "Failed to get converter version: %s",
                stderr.strip(),
            )
            return ""
        version_match = re.search(r"version\s(\d+\.\d+\.\d+)", stdout)
        if version_match:
            return version_match.group(1)
        else:
            logging.exception(
                "Unexpected error while getting neutron converter version"
            )
            return ""
    except Exception:
        logging.exception("Error while getting neutron converter version")
        return ""


def get_neutron_kernel_kinds(target: str = "imxrt700") -> dict[int, str]:
    """
    Retrieve kernel kinds supported by neutron_converter for the specified target.

    Executes the neutron_converter command with the --show-kernel-kinds option,
    parses its output, and returns a dictionary mapping kernel IDs to kernel
    names.

    :param target: Target platform for which kernel kinds should be queried.
    Defaults to "imxrt700".
    :return: Returns empty dict if neutron_converter exits with an error.
    Otherwise, a dictionary where:
            - key: kernel ID (int)
            - value: kernel name (str)
    """
    proc = subprocess.Popen(
        ["neutron_converter", "--target", target, "--show-kernel-kinds"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = proc.communicate(timeout=10)
    if proc.returncode != 0:
        logging.error(
            "Failed to get kernrl kinds from neutron_converter: %s",
            stderr.strip(),
        )
        return {}
    return {
        int(op_id): name
        for op_id, name in re.findall(r"\[\s*(\d+)\s*\]\s+(.+)", stdout)
    }
