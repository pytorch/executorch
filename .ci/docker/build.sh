#!/bin/bash

set -ex

OS="ubuntu"
OS_VERSION=22.04
CLANG_VERSION=12

docker build \
  --no-cache \
  --progress=plain \
  --build-arg "OS_VERSION=${OS_VERSION}" \
  --build-arg "CLANG_VERSION=${CLANG_VERSION}" \
  -f "${OS}"/Dockerfile \
  "$@" \
  .
