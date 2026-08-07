#!/bin/bash
# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -u
EIQ_PYPI_URL="${EIQ_PYPI_URL:-https://eiq.nxp.com/repository}"

pushd "$(dirname "$0")/../../../../.."

./install_executorch.sh
./devtools/install_requirements.sh

pip install --index-url ${EIQ_PYPI_URL} eiq-neutron-sdk==3.2.0

python3 -m examples.nxp.aot_neutron_compile -m cifar10 -d -q --use_channels_last_dim_order --remove-quant-io-ops
mv cifar10_nxp_delegate.pte model.pte
xxd -i model.pte > model_pte.h

popd
cd "$(dirname "$0")"

echo '#ifdef __MCUXPRESSO' > model_pte.h
echo '#define __PLACEMENT __attribute__((section(".data.$modeldata")))' >> model_pte.h
echo '#else' >> model_pte.h
echo '#define __PLACEMENT __attribute__((section(".modeldata")))' >> model_pte.h
echo '#endif' >> model_pte.h
echo >> model_pte.h
echo 'static const uint8_t model_pte[] __ALIGNED(16) __PLACEMENT = {' >> model_pte.h

cat ../../../../../model_pte.h | grep -v unsigned >> model_pte.h
