/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#pragma once

#include <cstdint>

struct __attribute__((packed)) NeuronPayloadHeader {
  unsigned char Version;

  uint32_t InputCount;

  uint32_t OutputCount;

  uint32_t DataLen;
};

struct NeuronPayload {
  NeuronPayload(const void* payload, size_t size)
      : Header(*(struct NeuronPayloadHeader*)payload),
        CompiledNetwork((char*)payload + sizeof(struct NeuronPayloadHeader)) {}

  NeuronPayloadHeader Header;

  void* CompiledNetwork = nullptr;
};
