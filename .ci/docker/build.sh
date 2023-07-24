#!/bin/bash

set -ex

IMAGE_NAME="$1"
shift

echo "Building ${IMAGE_NAME} Docker image"

OS=ubuntu
OS_VERSION=22.04
CLANG_VERSION=12
PYTHON_VERSION=3.10
MINICONDA_VERSION='23.5.1-0'

docker build \
  --no-cache \
  --progress=plain \
  --build-arg "OS_VERSION=${OS_VERSION}" \
  --build-arg "CLANG_VERSION=${CLANG_VERSION}" \
  --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
  --build-arg "MINICONDA_VERSION=${MINICONDA_VERSION}" \
  -f "${OS}"/Dockerfile \
  "$@" \
  .
