import os

import cpuinfo
import torch

from .scripts.download_qnn_sdk import install_qnn_sdk, is_linux_x86, QNN_ZIP_URL

env_flag = os.getenv("EXECUTORCH_BUILDING_WHEEL", "0").lower()
# If users have preinstalled QNN_SDK_ROOT, we will use it.
qnn_sdk_root_flag = os.getenv("QNN_SDK_ROOT", None)

if env_flag not in ("1", "true", "yes"):
    if qnn_sdk_root_flag:
        print(
            f"[QNN] Using QNN SDK at {qnn_sdk_root_flag} (from QNN_SDK_ROOT)",
            flush=True,
        )
    elif is_linux_x86():
        ok = install_qnn_sdk()
        if not ok:
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

info = cpuinfo.get_cpu_info()
vendor = info.get("vendor_id_raw", "").lower()
if "amd" in vendor:
    torch.backends.mkldnn.enabled = False
