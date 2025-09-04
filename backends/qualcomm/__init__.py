import os

from .scripts.download_qnn_sdk import install_qnn_sdk

env_flag = os.getenv("EXECUTORCH_BUILDING_WHEEL", "0").lower()
# If users have preinstalled QNN_SDK_ROOT, we will use it.
qnn_sdk_root_flag = os.getenv("QNN_SDK_ROOT", None)

if not env_flag in ("1", "true", "yes") and not qnn_sdk_root_flag:
    install_qnn_sdk()
