import ctypes
import os
import pathlib

from .scripts.download_qnn_sdk import install_qnn_sdk

print("Running executorch/backends/qualcomm/__init__.py")

env_flag = os.getenv("EXECUTORCH_INSTALL_QNN_SDK", "0").lower()

if env_flag in ("1", "true", "yes"):
    print("[QCOM init] EXECUTORCH_INSTALL_QNN_SDK is set -> installing Qualcomm SDK...")
else:
    install_qnn_sdk()
    print(f"[QCOM init] Skipping SDK install (EXECUTORCH_INSTALL_QNN_SDK={env_flag!r})")
# install_qnn_sdk()
