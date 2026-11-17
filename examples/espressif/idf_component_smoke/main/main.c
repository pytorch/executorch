/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Runs each embedded model (see main/CMakeLists.txt) through et_model on the
 * forward test cases in its .io file (the inputs and expected outputs of a
 * bundled program's test cases), and compares the outputs with the expected
 * ones.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <et_model.h>

#include "sdkconfig.h"

typedef struct {
  const char* name;
  const uint8_t* pte;
  const uint8_t* pte_end;
  const uint8_t* io;
} smoke_model_t;

extern const smoke_model_t smoke_models[];
extern const uint32_t smoke_model_count;

enum { kIoAlignment = 16, kScalarTypeFloat = 6 };

static uint32_t read_u32(const uint8_t** cursor) {
  uint32_t value;
  memcpy(&value, *cursor, sizeof(value));
  *cursor += sizeof(value);
  return value;
}

/* Returns the next tensor's data and byte size, and moves past it. */
static const uint8_t* next_tensor(
    const uint8_t* io, const uint8_t** cursor, uint32_t* nbytes) {
  *nbytes = read_u32(cursor);
  *cursor += (kIoAlignment - (size_t)(*cursor - io) % kIoAlignment) %
      kIoAlignment;
  const uint8_t* data = *cursor;
  *cursor += *nbytes;
  return data;
}

static bool check_outputs(
    const smoke_model_t* model,
    et_model_t* m,
    uint32_t index,
    const uint8_t** cursor) {
  float atol;
  float rtol;
  memcpy(&atol, model->io, sizeof(atol));
  memcpy(&rtol, model->io + sizeof(atol), sizeof(rtol));
  uint32_t count = read_u32(cursor);
  if (count != et_model_num_outputs(m)) {
    printf("%s: case %u has %u expected outputs, model has %u\n", model->name,
           (unsigned)index, (unsigned)count,
           (unsigned)et_model_num_outputs(m));
    return false;
  }
  for (uint32_t i = 0; i < count; i++) {
    uint32_t nbytes;
    const uint8_t* expected = next_tensor(model->io, cursor, &nbytes);
    size_t actual_nbytes = 0;
    const uint8_t* actual = et_model_output_data(m, i, &actual_nbytes);
    et_tensor_info_t info;
    if (actual == NULL || actual_nbytes != nbytes ||
        !et_model_output_info(m, i, &info)) {
      printf("%s: case %u output %u is missing or has the wrong size\n",
             model->name, (unsigned)index, (unsigned)i);
      return false;
    }
    if (info.scalar_type != kScalarTypeFloat) {
      if (memcmp(actual, expected, nbytes) != 0) {
        printf("%s: case %u output %u differs\n", model->name,
               (unsigned)index, (unsigned)i);
        return false;
      }
      continue;
    }
    for (uint32_t j = 0; j < nbytes / sizeof(float); j++) {
      float a;
      float b;
      memcpy(&a, actual + j * sizeof(float), sizeof(float));
      memcpy(&b, expected + j * sizeof(float), sizeof(float));
      if (!(fabsf(a - b) <= atol + rtol * fabsf(b))) {
        printf("%s: case %u output %u[%u] is %g, expected %g\n", model->name,
               (unsigned)index, (unsigned)i, (unsigned)j, (double)a,
               (double)b);
        return false;
      }
    }
  }
  return true;
}

/* Runs the cases in order on one loaded method, so a stateful model's cases
 * continue from the previous ones. */
static bool run_model(const smoke_model_t* model) {
  et_model_config_t cfg = ET_MODEL_CONFIG_DEFAULT(
      model->pte, (size_t)(model->pte_end - model->pte));
#if CONFIG_ET_PROGRAM_VERIFICATION
  cfg.verify = true;
#endif
  et_model_t* m = et_model_create(&cfg);
  bool api_ok = m != NULL && et_model_load_method(m, "forward");
  const uint8_t* cursor = model->io + 2 * sizeof(float);
  uint32_t cases = read_u32(&cursor);
  uint32_t passed = 0;
  for (; api_ok && passed < cases; passed++) {
    uint32_t count = read_u32(&cursor);
    for (uint32_t i = 0; api_ok && i < count; i++) {
      uint32_t nbytes;
      const uint8_t* data = next_tensor(model->io, &cursor, &nbytes);
      api_ok = et_model_set_input(m, i, data, nbytes);
    }
    api_ok = api_ok && et_model_execute(m);
    if (!api_ok || !check_outputs(model, m, passed, &cursor)) {
      break;
    }
  }
  if (!api_ok && m != NULL) {
    printf("%s: %s\n", model->name, et_model_last_error(m));
  }
  bool ok = api_ok && passed == cases;
  printf("%s: %s, %u/%u cases\n", model->name, ok ? "PASS" : "FAIL",
         (unsigned)passed, (unsigned)cases);
  et_model_destroy(m);
  return ok;
}

void app_main(void) {
  bool ok = true;
  for (uint32_t i = 0; i < smoke_model_count; i++) {
    ok = run_model(&smoke_models[i]) && ok;
  }
  printf("ET_COMPONENT_SMOKE: %s\n", ok ? "PASS" : "FAIL");
  printf("Program complete.\n");
}
