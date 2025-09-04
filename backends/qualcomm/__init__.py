import os

from .scripts.download_qnn_sdk import install_qnn_sdk

print("Running executorch/backends/qualcomm/__init__.py")

env_flag = os.getenv("EXECUTORCH_BUILDING_WHEEL", "0").lower()
print(f"[QCOM init] EXECUTORCH_BUILDING_WHEEL = {env_flag!r}")

if env_flag in ("1", "true", "yes"):
    print(
        "[QCOM init] EXECUTORCH_BUILDING_WHEEL is set -> skipping Qualcomm SDK install."
    )
else:
    print("[QCOM init] SDK install flag not set -> installing Qualcomm SDK...")
    install_qnn_sdk()
