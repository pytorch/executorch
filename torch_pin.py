# CHANNEL selects the wheel source for torch and its domain libraries.
#   "nightly" - dev builds from /whl/nightly. NIGHTLY_VERSION is appended to
#               every package spec.
#   "test"    - release candidates from /whl/test.
#   "release" - stable releases from /whl.
# For "test" and "release", NIGHTLY_VERSION is ignored and CI installs the
# published wheels directly.
#
# Example: pinning to a release candidate when nightly is broken:
#   1. Set CHANNEL = "test".
#   2. Set the four version constants to the RC's major.minor.patch
#      (look up matching versions on https://download.pytorch.org/whl/test/).
#   3. Re-run install_requirements.sh; commit. The pre-commit hook calls
#      .github/scripts/update_pytorch_pin.py, which writes torch_branch()
#      (e.g. "release/2.12") into .ci/docker/ci_commit_pins/pytorch.txt and
#      re-syncs grafted c10 headers.
CHANNEL = "release"

TORCH_VERSION = "2.12.0"
TORCHAUDIO_VERSION = "2.11.0"
TORCHCODEC_VERSION = "0.13.0"
TORCHVISION_VERSION = "0.27.0"

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


def pip_cache_args() -> list[str]:
    if CHANNEL == "test":
        return ["--no-cache-dir"]
    return []


def _release_branch(version: str) -> str:
    return f"release/{version.rsplit('.', 1)[0]}"


def torch_branch() -> str:
    return _release_branch(TORCH_VERSION)


def torchaudio_branch() -> str:
    return _release_branch(TORCHAUDIO_VERSION)


def torchvision_branch() -> str:
    return _release_branch(TORCHVISION_VERSION)
