#!/bin/bash

set -ex

install_ubuntu() {
  apt-get update

  apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    wget \
    sudo \
    vim \
    jq \
    vim \
    unzip \
    gdb

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
