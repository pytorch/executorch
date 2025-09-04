import os

from .scripts.download_qnn_sdk import install_qnn_sdk

print("Running executorch/backends/qualcomm/__init__.py")

env_flag = os.getenv("EXECUTORCH_INSTALL_QNN_SDK", "0").lower()
print(f"[QCOM init] EXECUTORCH_INSTALL_QNN_SDK = {env_flag!r}")

if env_flag in ("1", "true", "yes"):
    print(
        "[QCOM init] EXECUTORCH_INSTALL_QNN_SDK is set -> skipping Qualcomm SDK install."
    )
else:
    print("[QCOM init] SDK install flag not set -> installing Qualcomm SDK...")
    install_qnn_sdk()
