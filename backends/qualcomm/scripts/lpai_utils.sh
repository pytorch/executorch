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

usage() { echo "usage: $0 [--serial abc] [--workspace /data/tmp/local/xxx] [--direct] [--lpai v6] [--hexagon v81] [--artifact /path/to/artifacts]" 1>&2; exit 1; }

short=H:,s:,w:,l:,x:,a:,d,h,
long=host:,serial:,workspace:,lpai:,hexagon:,artifact:,direct,help
args=$(getopt -a -o $short -l $long -n $0 -- $@)
eval set -- $args

host=""
serial=""
workspace=""
mode="hlos"
lpai=""
hexagon=""
artifact=""
while true; do
  case $1 in
    -H | --host) host=$2; shift 2;;
    -s | --serial) serial=$2; shift 2;;
    -w | --workspace) workspace=$2; shift 2;;
    -l | --lpai) lpai=$2; shift 2;;
    -x | --hexagon) hexagon=$2; shift 2;;
    -a | --artifact) artifact=$2; shift 2;;
    -d | --direct) mode="direct"; shift;;
    -h | --help) usage;;
    --) shift; break;;
    *) echo "unknown keyword: $1"; usage;;
  esac
done

if [[ -z $lpai ]]; then
  echo "please specify lpai version"
  usage
elif [[ $mode == "direct" && -z $workspace ]]; then
  echo "please provide device serial and workspace while using direct mode"
  usage
fi

signed_folder=$QNN_SDK_ROOT/lib/lpai-$lpai/signed
signer=$HEXAGON_SDK_ROOT/tools/elfsigner/elfsigner.py
mkdir -p $signed_folder

if [[ $mode == "hlos" ]]; then
  yes 2>/dev/null | python $signer -i $QNN_SDK_ROOT/lib/lpai-$lpai/unsigned/libQnnLpaiSkel.so -o $signed_folder
else
  if [[ -z $hexagon ]]; then
    echo "please specify hexagon arch"
  fi
  adb_args=""
  if [[ ! -z $host ]]; then
    adb_args="$adb_args -H $host"
  fi
  if [[ ! -z $serial ]]; then
    adb_args="$adb_args -s $serial"
  fi
  adb $adb_args shell mkdir -p $workspace
  yes 2>/dev/null | python $signer -i $QNN_SDK_ROOT/lib/lpai-$lpai/unsigned/libQnnLpai.so -o $signed_folder
  yes 2>/dev/null | python $signer -i $QNN_SDK_ROOT/lib/hexagon-$hexagon/unsigned/libQnnSystem.so -o $signed_folder
  yes 2>/dev/null | python $signer -i $HEXAGON_TOOLS_ROOT/Tools/target/hexagon/lib/$hexagon/G0/pic/libc++.so.1.0 -o $signed_folder
  yes 2>/dev/null | python $signer -i build-hexagon/backends/qualcomm/qnn_executorch/fastrpc/libqnn_executorch_skel.so -o $signed_folder
  yes 2>/dev/null | python $signer -i build-hexagon/backends/qualcomm/libqnn_executorch_backend.so -o $signed_folder
  if [[ ! -z $artifact ]]; then
    adb $adb_args push $(find $QNN_SDK_ROOT/lib/lpai-$lpai/signed/ -name 'lib*' ! -name '*LpaiSkel*') $workspace
    adb $adb_args push build-android/backends/qualcomm/qnn_executorch/fastrpc/libqnn_executorch_stub.so $workspace
    adb $adb_args push build-android/backends/qualcomm/qnn_executorch/fastrpc/qnn_executor_runner $workspace
    adb $adb_args shell "cd $workspace && rm -f libc++.so.1 && ln -s libc++.so.1.0 libc++.so.1"
    pte=$(find $artifact -type f -name "*.pte")
    input_list=$(find $artifact -type f -name "input*.txt")
    input_data=$(find $artifact -type f -name "input*.raw")
    output_folder=output_direct
    adb $adb_args shell "rm -rf $workspace/$output_folder && mkdir -p $workspace/$output_folder"
    adb $adb_args push $pte $input_list $input_data $workspace
    adb $adb_args shell "cd $workspace && \
      export LD_LIBRARY_PATH=. && export ADSP_LIBRARY_PATH=. \
      && echo 0x0C > qnn_executor_runner.farf && logcat -c \
      && ./qnn_executor_runner --model_path $(basename $pte) --output_folder_path $output_folder"
    adb $adb_args pull $workspace/$output_folder $artifact
  fi
fi
