import os
import platform
import sys
import threading

# The Qualcomm SDK setup below is deferred rather than run here, so that importing this
# package has no side effects. It used to run at import time, which meant that merely
# importing the package downloaded the SDK and libc++ over the network, re-executed the
# interpreter under a staged loader, rewrote QNN_SDK_ROOT and LD_LIBRARY_PATH, and raised
# when a machine had no network. It also disabled PyTorch's MKLDNN backend for the whole
# process on any AMD host, which affects every other model in that interpreter.
#
# Nothing about loading this package needs any of that. The native adaptor does not link
# the SDK; it resolves QNN symbols with dlopen when a model is actually compiled, and says
# so plainly when they are missing. So setup happens on the first call that needs it.

# Resolved on first use rather than at import, because the downloader lives in a sibling
# directory that some builds do not package, and importing this package must not require it.
# A module level __getattr__ keeps these as ordinary attributes, so callers and tests can still
# reach and replace them.
_LAZY_NAMES = ("install_qnn_sdk", "QNN_ZIP_URL")


def is_linux_x86() -> bool:
    """True when a prebuilt Qualcomm SDK is published for this platform."""
    return platform.system().lower() == "linux" and platform.machine().lower() in (
        "x86_64",
        "amd64",
        "i386",
        "i686",
    )


def __getattr__(name):
    if name in _LAZY_NAMES:
        from .scripts.download_qnn_sdk import install_qnn_sdk, QNN_ZIP_URL

        globals()["install_qnn_sdk"] = install_qnn_sdk
        globals()["QNN_ZIP_URL"] = QNN_ZIP_URL
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_sdk_ready = False
# Guards the flag above. Python's import lock does not, because the setup is now called from
# several modules rather than from one package __init__, and two threads importing two of them
# concurrently would otherwise both run an installer that downloads and rewrites the
# environment.
_sdk_lock = threading.Lock()


def setup_qnn_sdk() -> None:
    """Make the Qualcomm SDK usable in this process, once.

    Safe to call repeatedly and from more than one thread: the work happens on the first call and
    later calls return immediately. Called by the code paths that need the SDK, so a caller does
    not have to.
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

    # A preinstalled SDK is used as it is.
    qnn_sdk_root = os.getenv("QNN_SDK_ROOT", None)
    if qnn_sdk_root:
        print(f"[QNN] Using QNN SDK at {qnn_sdk_root} (from QNN_SDK_ROOT)", flush=True)
        _sdk_ready = True
        return

    # Downloading a prebuilt SDK is only possible for the platform it is published for.
    if not sys.modules[__name__].is_linux_x86():
        _sdk_ready = True
        return

    module = sys.modules[__name__]
    if not module.install_qnn_sdk():
        raise RuntimeError(
            "Failed to set up QNN SDK.\n\n"
            "To resolve, try one of:\n"
            "  1. Download the SDK manually from:\n"
            f"       {module.QNN_ZIP_URL}\n"
            "     Or go to step 2 if QNN SDK already exists.\n"
            "  2. Set QNN_SDK_ROOT to an existing SDK installation:\n"
            "       export QNN_SDK_ROOT=/path/to/qualcomm/sdk\n"
            "       export LD_LIBRARY_PATH="
            "$QNN_SDK_ROOT/lib/x86_64-linux-clang/:$LD_LIBRARY_PATH"
        )

    _sdk_ready = True


def disable_mkldnn_on_amd() -> None:
    """Turn off PyTorch's MKLDNN backend on an AMD host.

    MKLDNN crashes on some AMD hosts, which is why this exists. The original comment described
    it as producing wrong results; what was measured is a core dump, from a plain convolution
    with nothing from this backend involved.

    This changes a global PyTorch setting, so it is applied by the QNN compile paths rather than
    at import, where it would also change how unrelated models run in the same interpreter.
    """
    import torch

    try:
        import cpuinfo
    except ImportError:
        raise ImportError("Please install the cpuinfo with pip install py-cpuinfo.")

    vendor = cpuinfo.get_cpu_info().get("vendor_id_raw", "") or ""
    if "amd" in vendor.lower():
        torch.backends.mkldnn.enabled = False
