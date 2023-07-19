#!/bin/bash

set -ex

install_ubuntu() {
  apt-get update

  # TODO: Setup buck will come later in a separate PR
  wget -q https://github.com/facebook/buck/releases/download/v2021.01.12.01/buck.2021.01.12.01_all.deb
  apt install -y ./buck.2021.01.12.01_all.deb

  rm buck.2021.01.12.01_all.deb
  # Cleanup package manager
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

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
