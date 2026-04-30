# CHANNEL selects the wheel source for torch and its domain libraries.
#   "nightly" — dev builds from /whl/nightly. NIGHTLY_VERSION is appended to
#               every package spec, and CI source-builds pytorch from the
#               pinned SHA in pytorch.txt to catch upstream regressions.
#   "test"    — release candidates from /whl/test.
#   "release" — stable releases from /whl.
# For "test" and "release", NIGHTLY_VERSION is ignored and CI installs the
# published wheels directly (no source build).
#
# Example — pinning to a 2.12 release candidate when nightly is broken:
#   1. Set CHANNEL = "test".
#   2. Set the four version constants to the RC's major.minor.patch
#      (look up matching versions on https://download.pytorch.org/whl/test/).
#   3. Re-run install_requirements.sh; commit. The pre-commit hook calls
#      .github/scripts/update_pytorch_pin.py, which writes torch_branch()
#      (e.g. "release/2.12") into .ci/docker/ci_commit_pins/pytorch.txt and
#      re-syncs grafted c10 headers.
CHANNEL = "test"

TORCH_VERSION = "2.11.0"
TORCHAUDIO_VERSION = "2.11.0"
TORCHCODEC_VERSION = "0.11.0"
TORCHVISION_VERSION = "0.26.0"

NIGHTLY_VERSION = "dev20260318"


def _spec(name: str, version: str) -> str:
    if CHANNEL == "nightly":
        return f"{name}=={version}.{NIGHTLY_VERSION}"
    return f"{name}=={version}"


def torch_spec() -> str:
    return _spec("torch", TORCH_VERSION)


def torchaudio_spec() -> str:
    return _spec("torchaudio", TORCHAUDIO_VERSION)


def torchcodec_spec() -> str:
    return _spec("torchcodec", TORCHCODEC_VERSION)


def torchvision_spec() -> str:
    return _spec("torchvision", TORCHVISION_VERSION)


def torch_index_url_base() -> str:
    if CHANNEL == "release":
        return "https://download.pytorch.org/whl"
    return f"https://download.pytorch.org/whl/{CHANNEL}"


def torch_branch() -> str:
    # PyTorch uses "release/M.N" branches; derive from the pinned version.
    # Used by update_pytorch_pin.py to write into pytorch.txt for test/release.
    return f"release/{TORCH_VERSION.rsplit('.', 1)[0]}"
