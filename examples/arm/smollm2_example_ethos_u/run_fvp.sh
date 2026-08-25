#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)
fvp_bin="${repo_root}/examples/arm/arm-scratch/FVP-corstone320/models/Linux64_GCC-9.3/FVP_Corstone_SSE-320"
runner=""

usage() {
  cat <<EOF
Usage: $(basename "$0") --runner=PATH [options]

Options:
  --runner=PATH   Runner ELF to execute.
  --fvp=PATH      FVP binary. Default: ${fvp_bin}
EOF
}

for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
    --runner=*) runner="${arg#*=}" ;;
    --fvp=*) fvp_bin="${arg#*=}" ;;
    *)
      echo "Unknown option: ${arg}" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${runner}" ]]; then
  echo "--runner is required" >&2
  exit 1
fi

exec "${fvp_bin}" \
  -C mps4_board.subsystem.ethosu.num_macs=256 \
  -C mps4_board.visualisation.disable-visualisation=1 \
  -C vis_hdlcd.disable_visualisation=1 \
  -C mps4_board.telnetterminal0.start_telnet=0 \
  -C mps4_board.uart0.out_file='-' \
  -C mps4_board.uart0.unbuffered_output=1 \
  -C mps4_board.uart0.shutdown_on_eot=1 \
  -a "${runner}" \
  -C mps4_board.subsystem.ethosu.extra_args="--fast"
