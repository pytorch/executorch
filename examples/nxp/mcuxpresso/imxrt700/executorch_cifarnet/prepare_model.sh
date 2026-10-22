#!/bin/bash
# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ue

pushd "$(dirname "$0")/../../../../.."

./install_executorch.sh
./devtools/install_requirements.sh

pip install -r backends/nxp/requirements-eiq.txt

python3 -m examples.nxp.aot_neutron_compile -m cifar10 -d -q --use_channels_last_dim_order --remove-quant-io-ops
mv cifar10_nxp_delegate.pte model.pte

popd

cat > model_pte.h <<'EOF'
#ifdef __MCUXPRESSO
#define __PLACEMENT __attribute__((section(".data.$modeldata")))
#else
#define __PLACEMENT __attribute__((section(".modeldata")))
#endif
 
static const uint8_t model_pte[] __ALIGNED(16) __PLACEMENT = {
EOF


xxd -i "$(dirname "$0")/../../../../../model.pte" | grep -v unsigned >> model_pte.h
