#include <gtest/gtest.h>

#include <executorch/platform/Platform.h>

TEST(ExecutorPalTest, Initialization) {
  /*
   * Ensure `et_pal_init` can be called multiple times.
   * It has already been called once in the main() function.
   */
  et_pal_init();
}

TEST(ExecutorPalTest, TimestampCoherency) {
  et_pal_init();

  et_timestamp_t time_a = et_pal_current_ticks();
  ASSERT_TRUE(time_a >= 0);

  et_timestamp_t time_b = et_pal_current_ticks();
  ASSERT_TRUE(time_b >= time_a);
}
