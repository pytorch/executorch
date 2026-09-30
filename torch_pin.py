# PyTorch release used by development installs and release-wheel metadata.
TORCH_VERSION = "2.14.0"
TORCHVISION_VERSION = "0.29.0"
TORCHAUDIO_VERSION = "2.11.0"
# Date used by the main-branch weekly pin bot. Release preparation resolves the
# selected TORCH_VERSION tag and updates the source commit separately.
NIGHTLY_VERSION = "dev20260913"

# Changed to True by scripts/release/apply-release-changes.sh. Release wheels
# declare the PyTorch release above; development and minimal wheels do not.
RELEASE_WHEEL = False

# Changed to True after stable third-party artifacts and submodule tags exist.
RELEASE_DEPENDENCIES_FINALIZED = False
