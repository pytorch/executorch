/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/freeable_buffer.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::runtime::FreeableBuffer;

struct FreeCallArgs {
  size_t calls;
  void* data;
  size_t size;
};

void RecordFree(void* context, void* data, size_t size) {
  auto* call = reinterpret_cast<FreeCallArgs*>(context);
  call->calls++;
  call->data = data;
  call->size = size;
}

TEST(FreeableBufferTest, EmptyTest) {
  FreeableBuffer fb;
  EXPECT_EQ(fb.data(), nullptr);
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

  // Freeing should clear them, even though free_fn is nullptr.
  fb.Free();
  EXPECT_EQ(fb.size(), 0);
  EXPECT_EQ(fb.data(), nullptr);
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
    EXPECT_EQ(call.data, &i);
    EXPECT_EQ(call.size, sizeof(i));

    // A second call to Free() should not call the function again.
    fb.Free();
    EXPECT_EQ(call.calls, 1);
  }

  // The destructor should not have called the function again.
  EXPECT_EQ(call.calls, 1);
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
  EXPECT_EQ(call.data, &i);
  EXPECT_EQ(call.size, sizeof(i));
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
  EXPECT_EQ(call.data, &i);
  EXPECT_EQ(call.size, sizeof(i));
}
