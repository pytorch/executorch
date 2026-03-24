#include <gtest/gtest.h>

#include <executorch/backends/xnnpack/runtime/executor/shape_env.h>
#include <executorch/backends/xnnpack/runtime/graph/tensor_spec.h>

#include <cstdint>

using namespace executorch::backends::xnnpack::core;
using namespace executorch::backends::xnnpack::executor;
using namespace executorch::backends::xnnpack::graph;

static TensorSpec make_spec(std::vector<DimSizeSpec> sizes) {
    return TensorSpec { .dtype = DType::Float32, .sizes = std::move(sizes) };
}

static Tensor make_tensor(std::vector<uint64_t> sizes) {
    return Tensor { .dtype = DType::Float32, .sizes = std::move(sizes) };
}

template <typename... Args>
static std::vector<Tensor> make_tensors(Args&&... args) {
    std::vector<Tensor> v;
    v.reserve(sizeof...(args));
    (v.push_back(std::forward<Args>(args)), ...);
    return v;
}

TEST(TestShapeEnv, single_sym_solves) {
    // spec: [s0], tensor: [42] => s0 = 42
    ShapeEnv env(1);
    std::vector<TensorSpec> specs = { make_spec({ DimSizeSpec::sym(0) }) };
    auto values = make_tensors(make_tensor({ 42 }));

    EXPECT_TRUE(env.specialize(specs, values));
    EXPECT_EQ(env.specialized_bounds[0].min, 42u);
    ASSERT_TRUE(env.specialized_bounds[0].max.has_value());
    EXPECT_EQ(*env.specialized_bounds[0].max, 42u);
}

TEST(TestShapeEnv, constant_dim_valid) {
    // spec: [const(10)], tensor: [10] => ok
    ShapeEnv env(0);
    std::vector<TensorSpec> specs = { make_spec({ DimSizeSpec::constant(10) }) };
    auto values = make_tensors(make_tensor({ 10 }));

    EXPECT_TRUE(env.specialize(specs, values));
}

TEST(TestShapeEnv, constant_dim_mismatch) {
    // spec: [const(10)], tensor: [5] => fail
    ShapeEnv env(0);
    std::vector<TensorSpec> specs = { make_spec({ DimSizeSpec::constant(10) }) };
    auto values = make_tensors(make_tensor({ 5 }));

    EXPECT_FALSE(env.specialize(specs, values));
}

TEST(TestShapeEnv, sym_with_offset) {
    // spec: [1*s0 + 3], tensor: [13] => s0 = 10
    ShapeEnv env(1);
    DimSizeSpec dim = { .coeffs = {{ .sym = 0, .coefficient = 1 }}, .offset = 3 };
    std::vector<TensorSpec> specs = { make_spec({ dim }) };
    auto values = make_tensors(make_tensor({ 13 }));

    EXPECT_TRUE(env.specialize(specs, values));
    EXPECT_EQ(env.specialized_bounds[0].min, 10u);
    ASSERT_TRUE(env.specialized_bounds[0].max.has_value());
    EXPECT_EQ(*env.specialized_bounds[0].max, 10u);
}

TEST(TestShapeEnv, same_sym_consistent) {
    // Two dims both use s0, both give s0 = 5 => ok, bounds tightened to [5, 5]
    ShapeEnv env(1);
    std::vector<TensorSpec> specs = {
        make_spec({ DimSizeSpec::sym(0) }),
        make_spec({ DimSizeSpec::sym(0) }),
    };
    auto values = make_tensors(make_tensor({ 5 }), make_tensor({ 5 }));

    EXPECT_TRUE(env.specialize(specs, values));
    EXPECT_EQ(env.specialized_bounds[0].min, 5u);
    ASSERT_TRUE(env.specialized_bounds[0].max.has_value());
    EXPECT_EQ(*env.specialized_bounds[0].max, 5u);
}

TEST(TestShapeEnv, same_sym_contradictory) {
    // Two dims both use s0, one gives s0=5, other gives s0=7 => fail
    ShapeEnv env(1);
    std::vector<TensorSpec> specs = {
        make_spec({ DimSizeSpec::sym(0) }),
        make_spec({ DimSizeSpec::sym(0) }),
    };
    auto values = make_tensors(make_tensor({ 5 }), make_tensor({ 7 }));

    EXPECT_FALSE(env.specialize(specs, values));
}

TEST(TestShapeEnv, multiple_syms) {
    // spec: [s0, s1], tensor: [3, 7] => s0=3, s1=7
    ShapeEnv env(2);
    std::vector<TensorSpec> specs = {
        make_spec({ DimSizeSpec::sym(0), DimSizeSpec::sym(1) }),
    };
    auto values = make_tensors(make_tensor({ 3, 7 }));

    EXPECT_TRUE(env.specialize(specs, values));
    EXPECT_EQ(env.specialized_bounds[0].min, 3u);
    EXPECT_EQ(*env.specialized_bounds[0].max, 3u);
    EXPECT_EQ(env.specialized_bounds[1].min, 7u);
    EXPECT_EQ(*env.specialized_bounds[1].max, 7u);
}

TEST(TestShapeEnv, spec_value_count_mismatch) {
    ShapeEnv env(1);
    std::vector<TensorSpec> specs = {
        make_spec({ DimSizeSpec::sym(0) }),
        make_spec({ DimSizeSpec::sym(0) }),
    };
    auto values = make_tensors(make_tensor({ 5 }));

    EXPECT_FALSE(env.specialize(specs, values));
}

TEST(TestShapeEnv, dim_count_mismatch) {
    // spec has 2 dims, tensor has 1 dim
    ShapeEnv env(1);
    std::vector<TensorSpec> specs = {
        make_spec({ DimSizeSpec::sym(0), DimSizeSpec::constant(10) }),
    };
    auto values = make_tensors(make_tensor({ 5 }));

    EXPECT_FALSE(env.specialize(specs, values));
}

TEST(TestShapeEnv, multi_term_skipped) {
    // Multi-term dim: 2*s0 + 3*s1 + 0 is left unsolved, bounds stay at defaults
    ShapeEnv env(2);
    DimSizeSpec dim = {
        .coeffs = {{ .sym = 0, .coefficient = 2 }, { .sym = 1, .coefficient = 3 }},
        .offset = 0
    };
    std::vector<TensorSpec> specs = { make_spec({ dim }) };
    auto values = make_tensors(make_tensor({ 100 }));

    EXPECT_TRUE(env.specialize(specs, values));
    // Bounds should remain at defaults (min=1, max=nullopt)
    EXPECT_EQ(env.specialized_bounds[0].min, 1u);
    EXPECT_FALSE(env.specialized_bounds[0].max.has_value());
    EXPECT_EQ(env.specialized_bounds[1].min, 1u);
    EXPECT_FALSE(env.specialized_bounds[1].max.has_value());
}

TEST(TestShapeEnv, non_unit_coefficient_skipped) {
    // Single term but coefficient != 1: 2*s0, left unsolved
    ShapeEnv env(1);
    DimSizeSpec dim = { .coeffs = {{ .sym = 0, .coefficient = 2 }}, .offset = 0 };
    std::vector<TensorSpec> specs = { make_spec({ dim }) };
    auto values = make_tensors(make_tensor({ 10 }));

    EXPECT_TRUE(env.specialize(specs, values));
    EXPECT_EQ(env.specialized_bounds[0].min, 1u);
    EXPECT_FALSE(env.specialized_bounds[0].max.has_value());
}

TEST(TestShapeEnv, repeated_specialize_resets) {
    // Calling specialize twice should reset bounds from first call
    ShapeEnv env(1);
    std::vector<TensorSpec> specs = { make_spec({ DimSizeSpec::sym(0) }) };
    auto values1 = make_tensors(make_tensor({ 42 }));
    auto values2 = make_tensors(make_tensor({ 7 }));

    EXPECT_TRUE(env.specialize(specs, values1));
    EXPECT_EQ(env.specialized_bounds[0].min, 42u);

    EXPECT_TRUE(env.specialize(specs, values2));
    EXPECT_EQ(env.specialized_bounds[0].min, 7u);
    EXPECT_EQ(*env.specialized_bounds[0].max, 7u);
}
