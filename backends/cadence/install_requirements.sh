#!/usr/bin/env bash

SCRIPT_DIR_PATH="$(
    cd -- "$(dirname "$0")" >/dev/null 2>&1
    pwd -P
)"

red=`tput setaf 1`
green=`tput setaf 2`

EXECUTORCH_ROOT_PATH=$(realpath "$SCRIPT_DIR_PATH/../../")
CADENCE_DIR_PATH="$EXECUTORCH_ROOT_PATH/backends/cadence"
HIFI4_DIR_PATH="$CADENCE_DIR_PATH/hifi/third-party/nnlib/nnlib-hifi4"
HIFI5_DIR_PATH="$CADENCE_DIR_PATH/hifi/third-party/nnlib/nnlib-hifi5"
FUSION_DIR_PATH="$CADENCE_DIR_PATH/fusion_g3/third-party/nnlib/nnlib-FusionG3"
FACTO_DIR_PATH="$CADENCE_DIR_PATH/utils/FACTO"

cd "$EXECUTORCH_ROOT_PATH"

## HiFi

rm -rf "$HIFI4_DIR_PATH"
mkdir -p "$HIFI4_DIR_PATH"

echo "${green}ExecuTorch: Cloning hifi4 nnlib"
git clone "https://github.com/foss-xtensa/nnlib-hifi4.git" $HIFI4_DIR_PATH
cd $HIFI4_DIR_PATH
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "${red}ExecuTorch: Failed to clone hifi4 nnlib."
    exit 1
fi

rm -rf "$HIFI5_DIR_PATH"
mkdir -p "$HIFI5_DIR_PATH"

echo "${green}ExecuTorch: Cloning hifi5 nnlib"
git clone "https://github.com/foss-xtensa/nnlib-hifi5.git" $HIFI5_DIR_PATH
cd $HIFI5_DIR_PATH
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "${red}ExecuTorch: Failed to clone hifi5 nnlib."
    exit 1
fi

## Fusion G3

rm -rf "$FUSION_DIR_PATH"
mkdir -p "$FUSION_DIR_PATH"

echo "${green}ExecuTorch: Cloning fusion g3"
git clone "https://github.com/foss-xtensa/nnlib-FusionG3.git" $FUSION_DIR_PATH
cd $FUSION_DIR_PATH
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "${red}ExecuTorch: Failed to clone fusion g3."
    exit 1
fi

git checkout 11230f47b587b074ba0881deb28beb85db566ac2


## FACTO
pip install -e $FACTO_DIR_PATH
