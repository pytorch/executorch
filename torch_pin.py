# Since ExecuTorch often uses main-branch features of pytorch, only the nightly
# pip versions will have the required features.
#
# NOTE: If a newly-fetched version of the executorch repo changes the value of
# NIGHTLY_VERSION, you should re-run install_executorch.sh script to install the necessary
# package versions.
#
# NOTE: If you're changing, make the corresponding change in .ci/docker/ci_commit_pins/pytorch.txt
# by picking the hash from the same date in https://hud.pytorch.org/hud/pytorch/pytorch/nightly/
#
# NOTE: If you're changing, make the corresponding supported CUDA versions in
# SUPPORTED_CUDA_VERSIONS above if needed.
TORCH_VERSION = "2.10.0"
NIGHTLY_VERSION = "dev20251004"
SUPPORTED_CUDA_VERSIONS = (
    (12, 6),
    (12, 8),
    (13, 0),
)
