# PyTorch release used by development installs and release-wheel metadata.
TORCH_VERSION = "2.14.0"
NIGHTLY_VERSION = "dev20260913"

# Changed to True by scripts/release/apply-release-changes.sh. Release wheels
# declare the PyTorch release above; development and minimal wheels do not.
RELEASE_WHEEL = False
