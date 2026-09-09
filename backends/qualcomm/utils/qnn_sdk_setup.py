# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Making the Qualcomm SDK usable, on the paths that need it.

Deliberately not in the package's ``__init__``, where it used to run: loading the package needs
no SDK, because the native adaptor links no QNN library and resolves those symbols with
``dlopen`` when a backend starts. So setup belongs on the calls that start one.
"""

import logging
import os
import platform
import threading

_sdk_ready = False
# Guards the flag above. Two threads starting a backend at the same time would otherwise both
# run an installer that downloads and rewrites the environment.
_sdk_lock = threading.Lock()

_vendor_is_amd = None
_vendor_lock = threading.Lock()


def setup_qnn_sdk() -> None:
    """Make the Qualcomm SDK usable in this process, once.

    Safe to call repeatedly and from more than one thread. After a success, later calls return
    immediately. After a failure nothing is remembered, so the next caller tries again, which is
    deliberate: a download can fail for a reason that goes away, such as a dropped network.
    """
    if _sdk_ready:
        return
    with _sdk_lock:
        # Another thread may have finished while this one waited.
        if _sdk_ready:
            return
        _setup_qnn_sdk_locked()


def _setup_qnn_sdk_locked() -> None:
    global _sdk_ready

    # The wheel build imports this package to collect its files, and has no use for an SDK.
    if os.getenv("EXECUTORCH_BUILDING_WHEEL", "0").lower() in ("1", "true", "yes"):
        _sdk_ready = True
        return

    # An empty value counts as set, because a caller that exports the variable at all has taken
    # charge of the SDK, often supplying it through LD_LIBRARY_PATH instead. Treating that as
    # unset downloads a second copy and rewrites both variables underneath them.
    qnn_sdk_root = os.getenv("QNN_SDK_ROOT")
    if qnn_sdk_root is not None:
        # Reported differently when empty, because naming it as a path would read as though a
        # location had been found.
        if qnn_sdk_root:
            logging.info("[QNN] Using QNN SDK at %s (from QNN_SDK_ROOT)", qnn_sdk_root)
        else:
            logging.info(
                "[QNN] QNN_SDK_ROOT is set but empty, so the SDK is left to the caller"
            )
        _sdk_ready = True
        return

    # Decided here rather than by asking the downloader, so a host with no published SDK returns
    # without needing a module some builds do not package.
    if not _is_linux_x86():
        _sdk_ready = True
        return

    # Read before the attempt, because the installer writes both variables while working and a
    # failure has to leave neither behind. It treats the QNN libraries being findable through
    # LD_LIBRARY_PATH as proof that a usable SDK is present, so a leftover value makes the next
    # call report success with no SDK path set at all.
    sdk_root_before = os.environ.get("QNN_SDK_ROOT")
    ld_path_before = os.environ.get("LD_LIBRARY_PATH")

    # In a finally, because the installer raises as well as returning False: a missing unpacking
    # tool and an unrecognised archive both escape it.
    installed = False
    try:
        installed = _install_qnn_sdk()
    finally:
        if not installed:
            for name, previous in (
                ("QNN_SDK_ROOT", sdk_root_before),
                ("LD_LIBRARY_PATH", ld_path_before),
            ):
                if previous is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = previous

    if not installed:
        from executorch.backends.qualcomm.scripts.download_qnn_sdk import QNN_ZIP_URL

        raise RuntimeError(
            "Failed to set up QNN SDK.\n\n"
            "To resolve, try one of:\n"
            "  1. Download the SDK manually from:\n"
            f"       {QNN_ZIP_URL}\n"
            "     Or go to step 2 if QNN SDK already exists.\n"
            "  2. Set QNN_SDK_ROOT to an existing SDK installation:\n"
            "       export QNN_SDK_ROOT=/path/to/qualcomm/sdk\n"
            "       export LD_LIBRARY_PATH="
            "$QNN_SDK_ROOT/lib/x86_64-linux-clang/:$LD_LIBRARY_PATH"
        )

    _sdk_ready = True


def _is_linux_x86() -> bool:
    """True when a prebuilt Qualcomm SDK is published for this platform."""
    return platform.system().lower() == "linux" and platform.machine().lower() in (
        "x86_64",
        "amd64",
        "i386",
        "i686",
    )


def _install_qnn_sdk() -> bool:
    # Imported here rather than at module scope because the downloader lives in a directory some
    # builds do not package, and it imports the network stack.
    try:
        from executorch.backends.qualcomm.scripts.download_qnn_sdk import (
            install_qnn_sdk,
        )
    except ModuleNotFoundError as error:
        # Only when the downloader itself is absent. A dependency missing from inside it is a
        # different problem and is left to speak for itself.
        if not (error.name or "").startswith("executorch.backends.qualcomm.scripts"):
            raise
        # Treated like a platform with no published SDK rather than an error, because a build that
        # leaves the downloader out supplies the libraries some other way, through the loader path
        # its own build rules set up. Raising here fails a lowering that would otherwise work, and
        # a genuinely missing library still reports itself when the backend starts.
        logging.info(
            "[QNN] This build does not package the SDK downloader, so the SDK is left to it. "
            "Set QNN_SDK_ROOT to choose an installation explicitly."
        )
        return True

    return install_qnn_sdk()


def disable_mkldnn_on_amd() -> None:
    """Turn off PyTorch's MKLDNN backend on an AMD host.

    MKLDNN core dumps on some AMD hosts, on a plain convolution with nothing from this backend
    involved.

    This changes a global PyTorch setting, so it is applied by the calls that start a backend
    rather than at import, where it would also change how unrelated models run in the same
    interpreter.

    The setting is re-applied on every call, since a caller may have turned it back on in
    between, but the vendor behind the decision is read only once. Reading it spawns a
    subprocess, and the call sites reach here more than once per lowering.
    """
    if not _host_is_amd():
        return

    import torch

    if not torch.backends.flags_frozen():
        torch.backends.mkldnn.enabled = False


def _host_is_amd() -> bool:
    global _vendor_is_amd
    if _vendor_is_amd is not None:
        return _vendor_is_amd
    with _vendor_lock:
        if _vendor_is_amd is None:
            _vendor_is_amd = _read_cpu_vendor().lower().find("amd") != -1
    return _vendor_is_amd


def _read_cpu_vendor() -> str:
    try:
        import cpuinfo
    except ImportError:
        # Warned about rather than raised, because this runs on every lowering and the guard it
        # feeds only matters on an AMD host. Aborting here would fail a lowering on an Intel or
        # ARM machine over a dependency that machine has no use for.
        logging.warning(
            "[QNN] py-cpuinfo is not installed, so the CPU vendor cannot be read and the AMD "
            "MKLDNN workaround is skipped. Install it with: pip install py-cpuinfo"
        )
        return ""

    # Releases of py-cpuinfo disagree on the key: older ones report "vendor_id", newer ones
    # "vendor_id_raw". Reading only one of them finds nothing on the other, silently.
    info = cpuinfo.get_cpu_info()
    return info.get("vendor_id_raw") or info.get("vendor_id") or ""
