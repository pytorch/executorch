
#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/utils.sh"

# Double check if the NDK version is set
[ -n "${ZEPHYR_SDK}" ]

install_prerequiresites() {
    rm /var/lib/dpkg/info/libc-bin.*
    apt-get clean
    apt-get -y update
    apt-get install -y libc-bin
    apt-get -y update
    apt-get clean
    apt-get install --no-install-recommends -y dos2unix
    apt-get install --no-install-recommends -y ca-certificates
    apt-get install -y --reinstall libc-bin
    apt-get install --no-install-recommends -y file
    apt-get install --no-install-recommends -y locales
    apt-get install --no-install-recommends -y git
    apt-get install --no-install-recommends -y build-essential
    apt-get install --no-install-recommends -y cmake
    apt-get install --no-install-recommends -y ninja-build gperf
    apt-get install --no-install-recommends -y device-tree-compiler
    apt-get install --no-install-recommends -y wget
    apt-get install --no-install-recommends -y curl
    apt-get install --no-install-recommends -y xz-utils
    apt-get install --no-install-recommends -y dos2unix
    apt-get install --no-install-recommends -y vim
    apt-get install --no-install-recommends -y nano
    apt-get install --no-install-recommends -y mc
    apt-get install --no-install-recommends -y openssh-server
    apt-get install -y gdb

    # Zephyr SDK relies on python 3.12
    apt install software-properties-common -y
    add-apt-repository ppa:deadsnakes/ppa -y
    apt update
    apt install -y python3.12 python3.12-dev python3.12-venv python3-pip
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

    # Upgrade cmake ot 3.24
    apt update
    apt install -y software-properties-common lsb-release
    apt update
    test -f /usr/share/doc/kitware-archive-keyring/copyright || \
        wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/kitware.list > /dev/null
    apt update
    apt install -y cmake

    # Install additional required software for Zephyr
    apt install --no-install-recommends -y ccache \
        dfu-util \
        python3-setuptools \
        python3-tk \
        python3-wheel \
        make \
        gcc \
        libsdl2-dev \
        libmagic1 \
        xterm \
        telnet \
        net-tools
    apt install --no-install-recommends -y gcc-multilib g++-multilib
    apt-get clean -y
    apt-get autoremove --purge -y
    rm -rf /var/lib/apt/lists/*
    wget https://apt.kitware.com/kitware-archive.sh && \
        chmod +x kitware-archive.sh && \
        ./kitware-archive.sh && \
        rm -f kitware-archive.sh
    pip_install --no-cache-dir west
    pip_install pyelftools

    # Cache the Zephyr SDK release assets in the image. CI still runs
    # `west sdk install` in its workspace, but the local proxy fallback can serve
    # these files without downloading them on every job.
    local zephyr_sdk_version
    zephyr_sdk_version="$(python3 "${SCRIPT_DIR}/zephyr_sdk_release_proxy.py" --print-version)"
    local zephyr_sdk_cache_dir="${ZEPHYR_SDK_RELEASE_PROXY_CACHE_DIR:-/opt/zephyr-sdk-cache/v${zephyr_sdk_version}}"
    python3 "${SCRIPT_DIR}/zephyr_sdk_release_proxy.py" \
        --version "${zephyr_sdk_version}" \
        --cache-dir "${zephyr_sdk_cache_dir}" \
        --populate-cache
}

install_prerequiresites
