/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#pragma once

#include <stddef.h>
#include <strings.h>

namespace example {
namespace llm_helper {

typedef enum { INT4, INT8, INT16, FP16, INT32, FP32, INVALID } LLMType;

inline size_t getLLMTypeSize(const LLMType llm_type) {
#define RETURN_LLMTYPE_SIZE(type, size) \
  case type:                            \
    return size;

  switch (llm_type) {
    RETURN_LLMTYPE_SIZE(INT4, 1) // min 1 byte
    RETURN_LLMTYPE_SIZE(INT8, 1)
    RETURN_LLMTYPE_SIZE(INT16, 2)
    RETURN_LLMTYPE_SIZE(FP16, 2)
    RETURN_LLMTYPE_SIZE(INT32, 4)
    RETURN_LLMTYPE_SIZE(FP32, 4)
    default:
      return 0; // invalid type
  }

#undef RETURN_LLMTYPE_SIZE
}

inline LLMType getLLMTypeFromName(const char* llm_type_name) {
#define RETURN_LLMTYPE(type)               \
  if (!strcasecmp(llm_type_name, #type)) { \
    return LLMType::type;                  \
  }

  RETURN_LLMTYPE(INT4)
  RETURN_LLMTYPE(INT8)
  RETURN_LLMTYPE(INT16)
  RETURN_LLMTYPE(FP16)
  RETURN_LLMTYPE(INT32)
  RETURN_LLMTYPE(FP32)

#undef RETURN_LLMTYPE

  return LLMType::INVALID;
}

inline const char* getLLMTypeName(const LLMType llm_type) {
#define RETURN_TYPENAME_STR(type) \
  case LLMType::type:             \
    return #type;

  switch (llm_type) {
    RETURN_TYPENAME_STR(INT4)
    RETURN_TYPENAME_STR(INT8)
    RETURN_TYPENAME_STR(INT16)
    RETURN_TYPENAME_STR(FP16)
    RETURN_TYPENAME_STR(FP32)
    RETURN_TYPENAME_STR(INT32)
    RETURN_TYPENAME_STR(INVALID)
  }

#undef RETURN_TYPENAME_STR
}

} // namespace llm_helper
} // namespace example
