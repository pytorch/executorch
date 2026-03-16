/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/freeable_buffer.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>

using namespace ::testing;

using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;

struct FreeCallArgs {
  size_t calls;
  std::variant<const void*, uint64_t> data;
  size_t size;
};

void RecordFree(void* context, void* data, size_t size) {
  auto* call = reinterpret_cast<FreeCallArgs*>(context);
  call->calls++;
  call->data = data;
  call->size = size;
}

void RecordInt64Free(void* context, uint64_t data, size_t size) {
  auto* call = reinterpret_cast<FreeCallArgs*>(context);
  call->calls++;
  call->data = data;
  call->size = size;
}

TEST(FreeableBufferTest, EmptyTest) {
  FreeableBuffer fb;
  EXPECT_EQ(fb.data(), nullptr);
  EXPECT_EQ(fb.data_safe().error(), Error::Ok);
  EXPECT_EQ(fb.data_safe().get(), nullptr);
  EXPECT_EQ(fb.size(), 0);
}

TEST(FreeableBufferTest, DataAndSizeTest) {
  int i;
  FreeableBuffer fb(
      /*data=*/&i,
      /*size=*/sizeof(i),
      /*free_fn=*/nullptr);

  // It should return the ctor params unmodified.
  EXPECT_EQ(fb.size(), sizeof(i));
  EXPECT_EQ(fb.data(), &i);
  EXPECT_EQ(fb.data_safe().error(), Error::Ok);
  EXPECT_EQ(fb.data_safe().get(), &i);

  // Freeing should clear them, even though free_fn is nullptr.
  fb.Free();
  EXPECT_EQ(fb.size(), 0);
  EXPECT_EQ(fb.data(), nullptr);
  EXPECT_EQ(fb.data_safe().error(), Error::Ok);
  EXPECT_EQ(fb.data_safe().get(), nullptr);

  // Use uint64_t constructor.
  const uint64_t i64 = 1;
  FreeableBuffer fb2(
      /*data_uint64=*/i64,
      /*size=*/sizeof(i64),
      /*free_fn=*/nullptr);

  // It should return the ctor params unmodified.
  EXPECT_EQ(fb2.size(), sizeof(i64));
  EXPECT_EQ(fb2.data_uint64_type().error(), Error::Ok);
  EXPECT_EQ(fb2.data_uint64_type().get(), i64);

  // Freeing should clear them, even though free_fn is nullptr.
  fb2.Free();
  EXPECT_EQ(fb2.size(), 0);
  EXPECT_EQ(fb2.data_uint64_type().error(), Error::Ok);
  EXPECT_EQ(fb2.data_uint64_type().get(), 0);
}

TEST(FreeableBufferTest, FreeTest) {
  // Updated when RecordFree() is called.
  FreeCallArgs call = {};

  {
    // Create a FreeableBuffer with a free_fn that records when it's called.
    int i;
    FreeableBuffer fb(
        /*data=*/&i,
        /*size=*/sizeof(i),
        /*free_fn=*/RecordFree,
        /*free_fn_context=*/&call);

    // Not called during construction.
    EXPECT_EQ(call.calls, 0);

    // Called once during Free() with the expected data/size.
    fb.Free();
    EXPECT_EQ(call.calls, 1);
    EXPECT_EQ(std::get<const void*>(call.data), &i);
    EXPECT_EQ(call.size, sizeof(i));

    // A second call to Free() should not call the function again.
    fb.Free();
    EXPECT_EQ(call.calls, 1);
  }

  // The destructor should not have called the function again.
  EXPECT_EQ(call.calls, 1);

  // Test with uint64_t constructor and free function.
  FreeCallArgs call2 = {};
  {
    uint64_t i64 = 1;
    FreeableBuffer fb(
        /*data_uint64=*/i64,
        /*size=*/sizeof(i64),
        /*free_fn=*/RecordInt64Free,
        /*free_fn_context=*/&call2);

    // Not called during construction.
    EXPECT_EQ(call2.calls, 0);

    // Called once during Free() with the expected data/size.
    fb.Free();
    EXPECT_EQ(call2.calls, 1);
    EXPECT_EQ(std::get<uint64_t>(call2.data), i64);
    EXPECT_EQ(call2.size, sizeof(i64));

    // A second call to Free() should not call the function again.
    fb.Free();
    EXPECT_EQ(call2.calls, 1);
  }
  EXPECT_EQ(call2.calls, 1);
}

TEST(FreeableBufferTest, DestructorTest) {
  // Updated when RecordFree() is called.
  FreeCallArgs call = {};
  int i;

  {
    // Create a FreeableBuffer with a free_fn that records when it's called.
    FreeableBuffer fb(
        /*data=*/&i,
        /*size=*/sizeof(i),
        /*free_fn=*/RecordFree,
        /*free_fn_context=*/&call);

    // Not called during construction.
    EXPECT_EQ(call.calls, 0);
  }

  // The destructor should have freed the data.
  EXPECT_EQ(call.calls, 1);
  EXPECT_EQ(std::get<const void*>(call.data), &i);
  EXPECT_EQ(call.size, sizeof(i));

  // Test with uint64_t constructor and free function.
  FreeCallArgs call2 = {};
  uint64_t i64 = 1;
  {
    FreeableBuffer fb2(
        /*data_uint64=*/i64,
        /*size=*/sizeof(i),
        /*free_fn=*/RecordInt64Free,
        /*free_fn_context=*/&call2);
    EXPECT_EQ(call2.calls, 0);
  }
  // The destructor should have freed the data.
  EXPECT_EQ(call2.calls, 1);
  EXPECT_EQ(std::get<uint64_t>(call2.data), i64);
  EXPECT_EQ(call2.size, sizeof(i));
}

TEST(FreeableBufferTest, MoveTest) {
  // Updated when RecordFree() is called.
  FreeCallArgs call = {};
  int i;

  // Create a FreeableBuffer with some data.
  FreeableBuffer fb_src(
      /*data=*/&i,
      /*size=*/sizeof(i),
      /*free_fn=*/RecordFree,
      /*free_fn_context=*/&call);
  EXPECT_EQ(fb_src.size(), sizeof(i));
  EXPECT_EQ(fb_src.data(), &i);

  // Move it into a second FreeableBuffer.
  FreeableBuffer fb_dst(std::move(fb_src));

  // The source FreeableBuffer should now be empty.
  EXPECT_EQ(fb_src.size(), 0); // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(fb_src.data(), nullptr); // NOLINT(bugprone-use-after-move)

  // The destination FreeableBuffer should have the data.
  EXPECT_EQ(fb_dst.size(), sizeof(i));
  EXPECT_EQ(fb_dst.data(), &i);
  // Freeing the source FreeableBuffer should not call the free function.
  fb_src.Free();
  EXPECT_EQ(call.calls, 0);

  // Freeing the destination FreeableBuffer should call the free function.
  fb_dst.Free();
  EXPECT_EQ(call.calls, 1);
  EXPECT_EQ(call.size, sizeof(i));

  // Test with uint64_t constructor and free function.
  FreeCallArgs call2 = {};
  const uint64_t i64 = 1;
  FreeableBuffer fb_src2(
      /*data_uint64=*/i64,
      /*size=*/sizeof(i64),
      /*free_fn=*/RecordInt64Free,
      /*free_fn_context=*/&call2);
  EXPECT_EQ(fb_src2.size(), sizeof(i64));
  EXPECT_EQ(fb_src2.data_uint64_type().error(), Error::Ok);
  EXPECT_EQ(fb_src2.data_uint64_type().get(), i64);

  // Move it into a second FreeableBuffer.
  FreeableBuffer fb_dst2(std::move(fb_src2));

  // The source FreeableBuffer should now be empty.
  EXPECT_EQ(fb_src2.size(), 0); // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(
      fb_src2.data_uint64_type().error(),
      Error::Ok); // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(
      fb_src2.data_uint64_type().get(), 0); // NOLINT(bugprone-use-after-move)

  // The destination FreeableBuffer should have the data.
  EXPECT_EQ(fb_dst2.size(), sizeof(i64));
  EXPECT_EQ(fb_dst2.data_uint64_type().error(), Error::Ok);
  EXPECT_EQ(fb_dst2.data_uint64_type().get(), i64);
  // Freeing the source FreeableBuffer should not call the free function.
  fb_src2.Free();
  EXPECT_EQ(call2.calls, 0);

  // Freeing the destination FreeableBuffer should call the free function.
  fb_dst2.Free();
  EXPECT_EQ(call2.calls, 1);
  EXPECT_EQ(call2.size, sizeof(i64));
}

TEST(FreeableBufferTest, APIMisuseDeathTest) {
  executorch::runtime::pal_init();
  int i;
  FreeableBuffer fb(
      /*data=*/&i,
      /*size=*/sizeof(i),
      /*free_fn=*/nullptr);
  EXPECT_EQ(fb.data_uint64_type().error(), Error::InvalidType);

  uint64_t i64 = 1;
  FreeableBuffer fb2(
      /*data_uint64=*/i64,
      /*size=*/sizeof(i64),
      /*free_fn=*/nullptr);
  EXPECT_EQ(fb2.data_safe().error(), Error::InvalidType);
  ET_EXPECT_DEATH(fb2.data(), ".*");
}
