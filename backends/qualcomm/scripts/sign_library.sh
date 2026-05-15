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
    echo ""
    echo "Non-direct mode (default), e.g.:"
    echo "  executorch$ $0 --lpai_arch v6"
    echo ""
    echo "Direct mode, e.g.:"
    echo "  executorch$ $0 --direct_mode --htp_arch v81 --lpai_arch v6"
    exit 1;
}

short=l:,c:,d,h
long=lpai_arch:,htp_arch:,direct_mode,help
args=$(getopt -a -o $short -l $long -n $0 -- $@)
eval set -- $args

lpai_arch=""
htp_arch=""
direct_mode=false
while true; do
  case $1 in
    -l | --lpai_arch) lpai_arch=$2; shift 2;;
    -c | --htp_arch) htp_arch=$2; shift 2;;
    -d | --direct_mode) direct_mode=true; shift;;
    -h | --help) usage;;
    --) shift; break;;
    *) echo "unknown keyword: $1"; usage;;
  esac
done

if [[ -z $lpai_arch ]]; then
  echo "please specify lpai version"
  usage
fi

if [ "$direct_mode" = true ]; then
  if [[ -z $htp_arch ]]; then
    echo "please specify htp_arch for direct mode"
    usage
  fi
fi

SCRIPT_DIR=$(cd -- "$(dirname "$0")" && pwd)
PRJ_ROOT=$SCRIPT_DIR/../../..

signed_folder=$QNN_SDK_ROOT/lib/lpai-$lpai_arch/signed
signer=$HEXAGON_SDK_ROOT/tools/elfsigner/elfsigner.py
mkdir -p $signed_folder

if [ "$direct_mode" = true ]; then
  yes 2>/dev/null | python $signer -i $QNN_SDK_ROOT/lib/lpai-$lpai_arch/unsigned/libQnnLpai.so -o $signed_folder
  yes 2>/dev/null | python $signer -i $QNN_SDK_ROOT/lib/hexagon-$htp_arch/unsigned/libQnnSystem.so -o $signed_folder
  yes 2>/dev/null | python $signer -i $HEXAGON_TOOLS_ROOT/Tools/target/hexagon/lib/$htp_arch/G0/pic/libc++abi.so.1 -o $signed_folder
  yes 2>/dev/null | python $signer -i $HEXAGON_TOOLS_ROOT/Tools/target/hexagon/lib/$htp_arch/G0/pic/libc++.so.1 -o $signed_folder
  yes 2>/dev/null | python $signer -i $PRJ_ROOT/build-direct/backends/qualcomm/qnn_executorch/direct_mode/libqnn_executorch_skel.so -o $signed_folder
  yes 2>/dev/null | python $signer -i $PRJ_ROOT/build-direct/backends/qualcomm/libqnn_executorch_backend.so -o $signed_folder
else
  yes 2>/dev/null | python $signer -i $QNN_SDK_ROOT/lib/lpai-$lpai_arch/unsigned/libQnnLpaiSkel.so -o $signed_folder
fi
