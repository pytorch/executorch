# Docker images for ExecuTorch CI

This directory contains everything needed to build the Docker images
that are used in ExecuTorch CI. The content of this directory are copied
from PyTorch CI https://github.com/pytorch/pytorch/tree/main/.ci/docker.
It also uses the same directory structure as PyTorch.

## Contents

* `build.sh` -- dispatch script to launch all builds
* `common` -- scripts used to execute individual Docker build stages
* `ubuntu` -- Dockerfile for Ubuntu image for CPU build and test jobs

## Usage

```bash
# Generic usage
./build.sh "${IMAGE_NAME}" "${DOCKER_BUILD_PARAMETERS}"

# Build a specific image
./build.sh executorch-ubuntu-22.04-clang12 -t myimage:latest

# Set CLANG version (see build.sh) and build image
CLANG_VERSION=11 ./build.sh executorch-ubuntu-22.04-clang11 -t myimage:latest
```

Images that set `TORCHCI_REPORTS=1` in `build.sh` (clang12 and gcc11) build
the published November 2 torch, torchvision and torchaudio source commits in
`ci_commit_pins/*-torchci.txt`, whose `torch.testing._internal.torchci`
reporter CI uses to upload test results to HUD. The shared macOS pin and other
images' sources are unchanged. The build checks the installed torch identity
and schema-1.0 reporter after installing the domain libraries.

These pins participate in the existing global Linux image hash. A pin change
can require other Linux images to rebuild, even if their pins are unchanged.
Validate the rebuilt image before enabling test-report uploads.

## Arm FVP cache

Arm and Zephyr SDK images cache the Corstone-300, Corstone-320, and Corstone-1000
archives in `/opt/arm-fvp-cache`. Download URLs, versions, checksums, and model
directories are maintained only in `ci_commit_pins/arm-fvp.sh`. Update FVP pins
in that file for both developer setup and CI.

`backends/arm/scripts/fvp_utils.sh` and `common/install_arm.sh` both source
`common/arm_fvp_cache.sh`, which sources the shared pins. Docker builds and
`install_arm.sh` install copies of these same source files under `/opt/executorch`
so image prewarming uses the same pins as developer setup. Setup validates cached
archives before installation.

Prewarming all three archives is mandatory for Arm and Zephyr SDK image builds.
If any archive is still missing or checksum-invalid after retries, the image
build fails, even when a consuming job does not need FVPs or Corstone-1000.
This replaces best-effort FVP prewarming so runtime jobs can require a complete
cache without falling back to the CDN.

`examples/arm/setup.sh --fvp corstone300` selects one platform. Repeat `--fvp`
to select several, or use `--disable-fvps` for jobs that do not run a simulator.
The default selection remains Corstone-300 and Corstone-320.

Setup automatically uses the image cache and forbids downloads on a cache miss.
A changed pin or missing archive therefore requires rebuilding the image;
runtime jobs do not fall back to Arm's CDN. Changes under `.ci/docker` produce
new image tags. Apply the `ciflow/docker` label to a PR that changes the images,
wait for `docker-builds` to finish, and rerun dependent jobs.

Outside these images, set `ARM_FVP_CACHE_DIR` to a persistent writable directory
to reuse archives across checkouts or jobs. The default is
`${XDG_CACHE_HOME:-$HOME/.cache}/executorch/fvps`. Set `ARM_FVP_CACHE_ONLY=1`
after prewarming to fail on cache misses; image builds explicitly use
`ARM_FVP_CACHE_ONLY=0` to populate their cache.

For migration, setup also checks the legacy scratch archive and
`ARM_SETUP_DOWNLOAD_CACHE/${ARM_FVP_URL##*/}` after a miss in the new cache.
Only checksum-valid archives are reused. Writable caches import them atomically;
cache-only operation uses them directly. The original archives are preserved.

Run the cache regression tests on Linux without downloading FVPs:

```bash
python3 -m unittest discover -s .ci/docker/tests -p test_arm_fvp_cache.py
```
