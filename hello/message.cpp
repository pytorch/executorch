/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hello/message.h"
#include <flatbuffers/flatbuffers.h>
#include "schema/program_generated.h"

namespace executorch {
  std::string message() {
    auto builder = flatbuffers::FlatBufferBuilder{1024};
    auto msg = builder.CreateString("🏎️ move fast!");
    auto string_offset = executorch_flatbuffer::CreateString(builder, msg);
    builder.Finish(string_offset);
    auto buffer = builder.GetBufferPointer();
    auto string_obj = flatbuffers::GetRoot<executorch_flatbuffer::String>(buffer);
    return string_obj->string_val()->str();
  }
}
