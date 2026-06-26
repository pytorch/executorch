/*
 * Copyright (c) 2026 iote.ai
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Simple RNN — multi-subgraph AXON delegation example.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(simple_rnn, LOG_LEVEL_INF);

extern int run_inference(void);

int main(void)
{
    LOG_INF("Simple RNN - ExecuTorch multi-subgraph AXON delegation");
    LOG_INF("Board: %s", CONFIG_BOARD_TARGET);

#if defined(CONFIG_NRF_AXON) && CONFIG_NRF_AXON
    LOG_INF("AXON NPU: enabled");
#else
    LOG_INF("AXON NPU: not available (CPU only)");
#endif

    int ret = run_inference();
    if (ret != 0) {
        LOG_ERR("Inference failed: %d", ret);
    }

    return 0;
}
