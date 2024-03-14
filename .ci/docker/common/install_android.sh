#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# Double check if the NDK version is set
[ -n "${ANDROID_NDK_VERSION}" ]

install_prerequiresites() {
  apt-get update

  apt-get install -y --no-install-recommends \
    openjdk-11-jdk \
    ca-certificates-java \
    ant

  # Cleanup package manager
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

download_and_install_ndk() {
  NDK_INSTALLATION_DIR=/opt/ndk
  mkdir -p "${NDK_INSTALLATION_DIR}"

  pushd /tmp
  # The NDK installation is cached on ossci-android S3 bucket
  curl -Os --retry 3 "https://ossci-android.s3.amazonaws.com/android-ndk-${ANDROID_NDK_VERSION}-linux.zip"
  unzip -qo "/tmp/android-ndk-${ANDROID_NDK_VERSION}-linux.zip"

  # Print the content for manual verification
  ls -lah "android-ndk-${ANDROID_NDK_VERSION}"
  mv "android-ndk-${NDK_INSTALLATION_DIR}"/* "${NDK_INSTALLATION_DIR}"

  popd
}

install_prerequiresites
download_and_install_ndk

# NB: We might also need to install Android SDK and some additional tools like
# Gradle here
