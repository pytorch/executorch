#include <executorch/core/OperatorRegistry.h>
#include <executorch/core/Runtime.h>
#include <executorch/core/kernel_types/kernel_types.h>
#include <executorch/test/utils/DeathTest.h>
#include <gtest/gtest.h>
#include <string>

using namespace ::testing;

namespace torch {
namespace executor {

class KernelDoubleRegistrationTest : public ::testing::Test {
 public:
  void SetUp() override {
    torch::executor::runtime_init();
  }
};

TEST_F(KernelDoubleRegistrationTest, Basic) {
  Kernel kernels[] = {Kernel(
      "aten::add.out",
      "v1/7;0,1,2,3|7;0,1,2,3|7;0,1,2,3",
      [](RuntimeContext&, EValue**) {})};
  ArrayRef<Kernel> kernels_array = ArrayRef<Kernel>(kernels);
  Error err = Error::InvalidArgument;

  ET_EXPECT_DEATH(
      { auto res = register_kernels(kernels_array); },
      std::to_string(static_cast<uint32_t>(err)));
}

} // namespace executor
} // namespace torch
