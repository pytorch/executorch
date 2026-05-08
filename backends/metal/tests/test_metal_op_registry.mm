/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 */
#import <Metal/Metal.h>
#include <gtest/gtest.h>
#include <executorch/backends/metal/ops/registry/MetalOpRegistry.h>
#include <executorch/backends/metal/ops/registry/MetalOp.h>
#include <memory>

using executorch::backends::metal_v2::MetalOpRegistry;
using executorch::backends::metal_v2::MetalOp;

namespace {

// Minimal MetalOp implementation for testing the registry. We don't
// need a real op — just something with a stable name() string.
class StubOp : public MetalOp {
 public:
  explicit StubOp(const char* n) : name_(n) {}
  const char* name() const override { return name_; }
  bool supports(::executorch::aten::ScalarType /*dtype*/) const override {
    return true;  // stub op accepts any dtype
  }
  const char* kernelSource() const override { return ""; }
  void dispatch(
      executorch::backends::metal_v2::MetalStream*,
      ::executorch::runtime::Span<::executorch::runtime::EValue*>,
      ::executorch::runtime::Span<::executorch::runtime::EValue*>) override {}
 private:
  const char* name_;
};

TEST(MetalOpRegistryTest, SharedReturnsSameInstance) {
  EXPECT_EQ(&MetalOpRegistry::shared(), &MetalOpRegistry::shared());
}

TEST(MetalOpRegistryTest, GetUnknownReturnsNull) {
  EXPECT_EQ(MetalOpRegistry::shared().get("aten::nonexistent_op_xyz"), nullptr);
}

TEST(MetalOpRegistryTest, HasOpFalseForUnknown) {
  EXPECT_FALSE(MetalOpRegistry::shared().hasOp("aten::nonexistent_op_xyz"));
}

TEST(MetalOpRegistryTest, RegisterAndGetByName) {
  const char* k = "test::stub_register_and_get";
  // Production code only registers once at startup; if a prior test
  // registered the same key, get() returns the existing one — that's
  // fine, test just verifies the lookup path.
  if (!MetalOpRegistry::shared().hasOp(k)) {
    MetalOpRegistry::shared().registerOp(std::make_unique<StubOp>(k));
  }
  MetalOp* found = MetalOpRegistry::shared().get(k);
  ASSERT_NE(found, nullptr);
  EXPECT_STREQ(found->name(), k);
}

TEST(MetalOpRegistryTest, HasOpTrueAfterRegister) {
  const char* k = "test::stub_has_op";
  if (!MetalOpRegistry::shared().hasOp(k)) {
    MetalOpRegistry::shared().registerOp(std::make_unique<StubOp>(k));
  }
  EXPECT_TRUE(MetalOpRegistry::shared().hasOp(k));
}

TEST(MetalOpRegistryTest, GetByStringOverload) {
  const char* k = "test::stub_string_overload";
  if (!MetalOpRegistry::shared().hasOp(k)) {
    MetalOpRegistry::shared().registerOp(std::make_unique<StubOp>(k));
  }
  std::string skey(k);
  MetalOp* found = MetalOpRegistry::shared().get(skey);
  ASSERT_NE(found, nullptr);
}

}  // namespace
