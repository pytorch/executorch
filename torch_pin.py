"""Dependency and wheel-train versions shared by installation and release tooling."""

# The newest complete PyTorch wheel train used by binary installs on main.
PYTORCH_VERSION = "2.15.0.dev20260922"
PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/nightly"
TORCHVISION_VERSION = "0.30.0.dev20260922"
TORCHAUDIO_VERSION = "2.11.0.dev20260922"

TORCHAO_INDEX_URL = PYTORCH_INDEX_URL
TORCHAO_NIGHTLY_VERSION = "0.19.0.dev20260922"
# PyTorch no longer publishes current ROCm nightlies. These jobs remain on the
# newest compatible test-index wheel instead of silently weakening the main pin.
ROCM_PYTORCH_VERSION = "2.14.0"

# CUDA wheel trains supported by main. Release preparation owns any temporary
# filtering needed for a particular release.
CUDA_WHEEL_VERSIONS = ["cu132", "cu134"]
