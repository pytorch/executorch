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
  OS=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
  case "$OS" in
    amzn)
      # https://docs.aws.amazon.com/corretto/latest/corretto-17-ug/amazon-linux-install.html
      yum install -y java-17-amazon-corretto \
        ca-certificates \
        ant
      ;;
    *)
      apt-get update

      # NB: Need OpenJDK 17 at the minimum
      apt-get install -y --no-install-recommends \
        openjdk-17-jdk \
        ca-certificates-java \
        ant

      # Cleanup package manager
      apt-get autoclean && apt-get clean
      rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
    ;;
  esac
}

install_ndk() {
  NDK_INSTALLATION_DIR=/opt/ndk
  rm -rf "${NDK_INSTALLATION_DIR}" && mkdir -p "${NDK_INSTALLATION_DIR}"

  pushd /tmp
  # The NDK installation is cached on ossci-android S3 bucket
  curl -Os --retry 3 "https://ossci-android.s3.amazonaws.com/android-ndk-${ANDROID_NDK_VERSION}-linux.zip"
  unzip -qo "android-ndk-${ANDROID_NDK_VERSION}-linux.zip"

  # Print the content for manual verification
  ls -lah "android-ndk-${ANDROID_NDK_VERSION}"
  mv "android-ndk-${ANDROID_NDK_VERSION}"/* "${NDK_INSTALLATION_DIR}"

  popd
}

install_cmdtools() {
  CMDTOOLS_FILENAME=commandlinetools-linux-11076708_latest.zip

  pushd /tmp
  # The file is cached on ossci-android S3 bucket
  curl -Os --retry 3 "https://ossci-android.s3.us-west-1.amazonaws.com/${CMDTOOLS_FILENAME}"
  unzip -qo "${CMDTOOLS_FILENAME}" -d /opt

  ls -lah /opt/cmdline-tools/bin
  popd
}

install_sdk() {
  SDK_INSTALLATION_DIR=/opt/android/sdk
  rm -rf "${SDK_INSTALLATION_DIR}" && mkdir -p "${SDK_INSTALLATION_DIR}"

  # These are the tools needed to build Android apps
  yes | /opt/cmdline-tools/bin/sdkmanager --sdk_root="${SDK_INSTALLATION_DIR}" --install "platforms;android-34"
  yes | /opt/cmdline-tools/bin/sdkmanager --sdk_root="${SDK_INSTALLATION_DIR}" --install "build-tools;33.0.1"
  # And some more tools for future emulator tests
  yes | /opt/cmdline-tools/bin/sdkmanager --sdk_root="${SDK_INSTALLATION_DIR}" --install "platform-tools"
  yes | /opt/cmdline-tools/bin/sdkmanager --sdk_root="${SDK_INSTALLATION_DIR}" --install "tools"
}

install_prerequiresites
install_ndk
install_cmdtools
install_sdk
