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
    echo ""
    echo "Additionally sign a custom op package, e.g.:"
    echo "  executorch$ $0 --direct_mode --htp_arch v81 --lpai_arch v6 \\"
    echo "      --op_package_dir examples/qualcomm/custom_op/example_op_package_lpai/ExampleLpaiOpPackage"
    echo ""
    echo "The DSP target the op package was built for defaults to"
    echo "$default_op_package_arch and can be overridden with --op_package_arch."
    exit 1;
}

short=l:,c:,d,p:,a:,h
long=lpai_arch:,htp_arch:,direct_mode,op_package_dir:,op_package_arch:,help
args=$(getopt -a -o $short -l $long -n $0 -- $@)
eval set -- $args

lpai_arch=""
htp_arch=""
direct_mode=false
op_package_dir=""
# Hexagon target the op package was compiled for. This is *not* $htp_arch and
# not $lpai_arch either: on SM8850 the HTP is V81 and the LPAI hardware version
# is V6, while the op package builds for hexagon-v79. The three version numbers
# are independent, and v79 is currently the only usable value because the QNN
# SDK ships a single LPAI op package makefile,
# share/QNN/OpPackageGenerator/makefiles/LPAI/Makefile.hexagon-v79, which
# hardcodes QNN_TARGET, -mv79 and the computev79 QuRT headers. The option exists
# so that this script does not have to change once the SDK ships more of them.
default_op_package_arch=v79
op_package_arch=$default_op_package_arch
while true; do
  case $1 in
    -l | --lpai_arch) lpai_arch=$2; shift 2;;
    -c | --htp_arch) htp_arch=$2; shift 2;;
    -d | --direct_mode) direct_mode=true; shift;;
    -p | --op_package_dir) op_package_dir=$2; shift 2;;
    -a | --op_package_arch) op_package_arch=$2; shift 2;;
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

# Sign one library, failing loudly if the signer did not.
#
# The exit status has to be checked explicitly: without it a failing signer (a
# missing input, an expired certificate, or an interpreter too new for
# elfsigner.py, which imports the `imp` module removed in Python 3.12) leaves
# this script exiting 0 while the signed library is missing or stale. The
# failure then only shows up on device as an opaque library load error.
#
# `yes` feeds the signer's interactive confirmation prompt and is expected to
# die with SIGPIPE once the signer exits, so PIPESTATUS[1] rather than $? is the
# status of interest here.
sign_library() {
  local lib=$1
  if [[ ! -f $lib ]]; then
    echo "no such library: $lib"
    exit 1
  fi
  yes 2>/dev/null | python $signer -i $lib -o $signed_folder
  local signer_status=${PIPESTATUS[1]}
  if [[ $signer_status -ne 0 ]]; then
    echo "failed to sign $lib (elfsigner.py exited $signer_status)"
    echo "note that elfsigner.py imports 'imp' and needs python < 3.12"
    exit 1
  fi
}

if [ "$direct_mode" = true ]; then
  sign_library $QNN_SDK_ROOT/lib/lpai-$lpai_arch/unsigned/libQnnLpai.so
  sign_library $QNN_SDK_ROOT/lib/hexagon-$htp_arch/unsigned/libQnnSystem.so
  sign_library $HEXAGON_TOOLS_ROOT/Tools/target/hexagon/lib/$htp_arch/G0/pic/libc++abi.so.1
  sign_library $HEXAGON_TOOLS_ROOT/Tools/target/hexagon/lib/$htp_arch/G0/pic/libc++.so.1
  sign_library $PRJ_ROOT/build-direct/backends/qualcomm/qnn_executorch/direct_mode/libqnn_executorch_skel.so
  sign_library $PRJ_ROOT/build-direct/backends/qualcomm/libqnn_executorch_backend.so
else
  sign_library $QNN_SDK_ROOT/lib/lpai-$lpai_arch/unsigned/libQnnLpaiSkel.so
fi

# A custom op package brings its own DSP code, which must be signed just like
# the runtime libraries above or the DSP refuses to load it. Both objects built
# by the hexagon target carry the kernel and both have to be signed: the op
# package itself and libLpaiOpPackageIsland.so, which is linked for
# always-resident (island) memory. Which one the runtime resolves depends on
# island mode, so sign both rather than guessing.
if [[ -n $op_package_dir ]]; then
  hexagon_libs=$op_package_dir/libs/hexagon-$op_package_arch
  if [[ ! -d $hexagon_libs ]]; then
    echo "no such directory: $hexagon_libs"
    echo "build the op package first, e.g. 'make lpai_hexagon_$op_package_arch' in $op_package_dir"
    exit 1
  fi
  # A glob that matches nothing expands to itself, which would hand the signer
  # the literal "*.so"; guard with -e so an unbuilt directory is reported here.
  signed_any=false
  for lib in $hexagon_libs/*.so; do
    [[ -e $lib ]] || continue
    sign_library $lib
    signed_any=true
  done
  if [ "$signed_any" = false ]; then
    echo "no .so found in $hexagon_libs"
    echo "build the op package first, e.g. 'make lpai_hexagon_$op_package_arch' in $op_package_dir"
    exit 1
  fi
fi
