#!/bin/bash

set -ex

install_ubuntu() {
  apt-get update

  apt-get install -y --no-install-recommends clang-"$CLANG_VERSION"
  apt-get install -y --no-install-recommends llvm-"$CLANG_VERSION"

  # Use update-alternatives to make this version the default
  update-alternatives --install /usr/bin/clang clang /usr/bin/clang-"$CLANG_VERSION" 50
  update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-"$CLANG_VERSION" 50
  # Override cc/c++ to clang as well
  update-alternatives --install /usr/bin/cc cc /usr/bin/clang 50
  update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++ 50

  # CLANG's packaging is a little messed up (the runtime libs aren't
  # added into the linker path), so give it a little help
  CLANG_LIBS=("/usr/lib/llvm-$CLANG_VERSION/lib/clang/"*"/lib/linux")
  for CLANG_LIB in "${CLANG_LIB[@]}"
  do
    echo "${CLANG_LIB}" > /etc/ld.so.conf.d/clang.conf
  done
  ldconfig

  # Cleanup package manager
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

if [ -n "$CLANG_VERSION" ]; then
  # Install base packages depending on the base OS
  ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
  case "$ID" in
    ubuntu)
      install_ubuntu
      ;;
    *)
      echo "Unable to determine OS..."
      exit 1
      ;;
  esac
fi
