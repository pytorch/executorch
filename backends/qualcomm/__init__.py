import os
import pathlib

from .scripts.download_qnn_sdk import _download_qnn_sdk, SDK_DIR

# -----------------------------------------------------------------------------
# Main SDK setup
# -----------------------------------------------------------------------------
qnn_root = os.environ.get("QNN_SDK_ROOT")
if qnn_root:
    SDK_DIR = pathlib.Path(qnn_root)
else:
    if not SDK_DIR.exists():
        print("Qualcomm SDK not found. Downloading...")
        _download_qnn_sdk()
    os.environ["QNN_SDK_ROOT"] = str(SDK_DIR)
