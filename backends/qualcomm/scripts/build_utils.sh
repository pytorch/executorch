#!/usr/bin/env bash
# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Resolve SoC model to HTP arch version and optionally LPAI hardware version.
# Sets HTP_ARCH (always) and LPAI_HW_VER (only when DSP_TYPE=0) in caller's scope.
# Arguments:
#   $1 - PYTHON_EXECUTABLE
#   $2 - SOC_MODEL
#   $3 - DSP_TYPE
resolve_soc_info() {
    local python_exec="$1"
    local soc_model="$2"
    local dsp_type="$3"

    HTP_ARCH=$($python_exec -c "
import sys, os
devnull = open(os.devnull, 'w')
old_stdout = sys.stdout
sys.stdout = devnull
from executorch.backends.qualcomm.utils.utils import get_soc_to_htp_arch_map
sys.stdout = old_stdout
m = get_soc_to_htp_arch_map()
if '${soc_model}' not in m:
    sys.exit(1)
print(m['${soc_model}'].value)
" 2>/dev/null) || {
        echo "Error: SoC model '${soc_model}' not found in HTP arch map."
        echo "Check supported models in executorch/backends/qualcomm/utils/utils.py get_soc_to_htp_arch_map()."
        exit 1
    }

    if [ "$dsp_type" = "0" ]; then
        LPAI_HW_VER=$($python_exec -c "
import sys, os
devnull = open(os.devnull, 'w')
old_stdout = sys.stdout
sys.stdout = devnull
from executorch.backends.qualcomm.utils.utils import get_soc_to_lpai_hw_ver_map
sys.stdout = old_stdout
m = get_soc_to_lpai_hw_ver_map()
if '${soc_model}' not in m:
    sys.exit(1)
print(m['${soc_model}'].value)
" 2>/dev/null) || {
            echo "Error: SoC model '${soc_model}' not found in LPAI hardware version map."
            echo "Check supported models in executorch/backends/qualcomm/utils/utils.py get_soc_to_lpai_hw_ver_map()."
            exit 1
        }
    fi
}
