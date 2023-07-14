#!/bin/bash

set -ex

IMAGE_NAME="$1"
shift

OS="ubuntu"
OS_VERSION=22.04
CLANG_VERSION=12

docker build \
  --no-cache \
  --progress=plain \
  --build-arg "IMAGE_NAME=${IMAGE_NAME}" \
  --build-arg "OS_VERSION=${OS_VERSION}" \
  --build-arg "CLANG_VERSION=${CLANG_VERSION}" \
  -f "${OS}"/Dockerfile \
  "$@" \
  .
