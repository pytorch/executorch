import os

from .scripts.download_qnn_sdk import check_glibc_exist, install_qnn_sdk, is_linux_x86


env_flag = os.getenv("EXECUTORCH_BUILDING_WHEEL", "0").lower()
# If users have preinstalled QNN_SDK_ROOT, we will use it.
qnn_sdk_root_flag = os.getenv("QNN_SDK_ROOT", None)

if (
    env_flag not in ("1", "true", "yes")
    and not qnn_sdk_root_flag
    and is_linux_x86()
    and check_glibc_exist()
):
    ok = install_qnn_sdk()

    if not ok:
        raise RuntimeError("Failed to install QNN SDK. Please check the logs above.")
