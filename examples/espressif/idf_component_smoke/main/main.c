/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Runs the add/multiply model from export_smoke_model.py through et_model
 * and checks both outputs exactly.
 */

#include <stdbool.h>
#include <stdio.h>

#include <et_model.h>

extern const uint8_t smoke_pte_start[];
extern const uint8_t smoke_pte_end[];

static const float kX[8] = {-4.0f, -1.5f, 0.0f, 0.25f, 1.0f, 2.5f, 7.0f, 16.0f};
static const float kY[8] = {1.0f, 3.0f, -2.0f, 0.25f, -0.5f, 4.0f, -9.0f, 0.125f};

static bool run_once(et_model_t* m) {
  if (!et_model_set_input(m, 0, kX, sizeof(kX)) ||
      !et_model_set_input(m, 1, kY, sizeof(kY)) || !et_model_execute(m)) {
    return false;
  }
  size_t sum_bytes = 0;
  size_t prod_bytes = 0;
  const float* sum = et_model_output_data(m, 0, &sum_bytes);
  const float* prod = et_model_output_data(m, 1, &prod_bytes);
  if (sum == NULL || prod == NULL || sum_bytes != sizeof(kX) ||
      prod_bytes != sizeof(kX)) {
    return false;
  }
  for (int i = 0; i < 8; i++) {
    if (sum[i] != kX[i] + kY[i] || prod[i] != kX[i] * kY[i]) {
      return false;
    }
  }
  return true;
}

void app_main(void) {
  et_model_config_t cfg = ET_MODEL_CONFIG_DEFAULT(
      smoke_pte_start, (size_t)(smoke_pte_end - smoke_pte_start));
  et_model_t* m = et_model_create(&cfg);
  bool ok = m != NULL && et_model_load_method(m, NULL);
  for (int i = 0; ok && i < 3; i++) {
    ok = run_once(m);
  }
  if (ok) {
    printf("ET_COMPONENT_SMOKE: PASS\n");
  } else {
    printf(
        "ET_COMPONENT_SMOKE: FAIL (%s)\n",
        m != NULL ? et_model_last_error(m) : "et_model_create failed");
  }
  et_model_destroy(m);
  printf("Program complete.\n");
}
