#!/bin/bash
# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if [[ -z $HEXAGON_SDK_ROOT || -z $QNN_SDK_ROOT ]]; then
  echo "please export HEXAGON_SDK_ROOT and QNN_SDK_ROOT"
  exit -1
fi

usage() { 
    echo "Usage: Sign the LPAI library for a given LPAI architecture"
    echo "e.g.: executorch$ $0 --lpai_arch v6"
    exit 1;
}

short=l:,h
long=lpai_arch:,help
args=$(getopt -a -o $short -l $long -n $0 -- $@)
eval set -- $args

lpai_arch=""
while true; do
  case $1 in
    -l | --lpai_arch) lpai_arch=$2; shift 2;;
    -h | --help) usage;;
    --) shift; break;;
    *) echo "unknown keyword: $1"; usage;;
  esac
done

if [[ -z $lpai_arch ]]; then
  echo "please specify lpai version"
  usage
fi

signed_folder=$QNN_SDK_ROOT/lib/lpai-$lpai_arch/signed
signer=$HEXAGON_SDK_ROOT/tools/elfsigner/elfsigner.py
mkdir -p $signed_folder

yes 2>/dev/null | python $signer -i $QNN_SDK_ROOT/lib/lpai-$lpai_arch/unsigned/libQnnLpaiSkel.so -o $signed_folder
