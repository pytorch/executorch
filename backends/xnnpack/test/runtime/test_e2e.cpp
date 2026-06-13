#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <memory>

#include <executorch/backends/xnnpack/runtime/core/quant_params.h>
#include <executorch/backends/xnnpack/runtime/core/tensor.h>
#include <executorch/backends/xnnpack/runtime/executor/executor.h>
#include <executorch/backends/xnnpack/runtime/graph/graph_builder.h>

using namespace executorch::backends::xnnpack::core;
using namespace executorch::backends::xnnpack::executor;
using namespace executorch::backends::xnnpack::graph;

static Storage make_owned(size_t size_in_bytes) {
  return std::move(Storage::create_owned(size_in_bytes).get());
}

TEST(TestE2E, add) {
  // Build graph: output = input_a + input_b
  // Shape: [1, 4] float32, static sizes.
  auto builder = GraphBuilder();

  auto spec = TensorSpec{
      .dtype = DType::Float32,
      .sizes = {DimSizeSpec::constant(1), DimSizeSpec::constant(4)}};

  auto a = builder.createInput(spec);
  auto b = builder.createInput(spec);
  auto add = builder.createOperator(Operator::Add, spec, a, b);
  builder.createOutput(add);

  auto graph = builder.build();
  auto executor_result = Executor::build(graph);
  ASSERT_TRUE(executor_result.ok());
  auto& executor = *executor_result;

  // Create input tensors.
  Tensor ta;
  ta.dtype = DType::Float32;
  ta.sizes = {1, 4};
  ta.storage = make_owned(4 * sizeof(float));
  float* da = ta.data_mut<float>();
  da[0] = 1.0f;
  da[1] = 2.0f;
  da[2] = 3.0f;
  da[3] = 4.0f;

  Tensor tb;
  tb.dtype = DType::Float32;
  tb.sizes = {1, 4};
  tb.storage = make_owned(4 * sizeof(float));
  float* db = tb.data_mut<float>();
  db[0] = 10.0f;
  db[1] = 20.0f;
  db[2] = 30.0f;
  db[3] = 40.0f;

  std::vector<Tensor> inputs;
  inputs.push_back(std::move(ta));
  inputs.push_back(std::move(tb));

  auto outputs_result = executor.run({inputs.data(), inputs.size()});
  ASSERT_TRUE(outputs_result.ok());
  auto& outputs = *outputs_result;

  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(outputs[0].sizes, (std::vector<uint64_t>{1, 4}));

  auto* out = outputs[0].data_const<float>();
  EXPECT_FLOAT_EQ(out[0], 11.0f);
  EXPECT_FLOAT_EQ(out[1], 22.0f);
  EXPECT_FLOAT_EQ(out[2], 33.0f);
  EXPECT_FLOAT_EQ(out[3], 44.0f);
}

/*
 * Helper: build and run a single binary op over two [1,n] float32 inputs.
 * Returns the executor (owns the arena) and the output tensors (alias it).
 */
struct BinaryOpResult {
  Executor executor;
  std::vector<Tensor> outputs;
};

BinaryOpResult
run_binary_op(Operator op, const float* a, const float* b, size_t n) {
  auto builder = GraphBuilder();
  auto spec = TensorSpec{
      .dtype = DType::Float32,
      .sizes = {
          DimSizeSpec::constant(1),
          DimSizeSpec::constant(static_cast<int64_t>(n))}};

  auto ia = builder.createInput(spec);
  auto ib = builder.createInput(spec);
  auto out = builder.createOperator(op, spec, ia, ib);
  builder.createOutput(out);

  auto graph = builder.build();
  auto executor_result = Executor::build(graph);
  EXPECT_TRUE(executor_result.ok());
  auto& executor = *executor_result;

  Tensor ta;
  ta.dtype = DType::Float32;
  ta.sizes = {1, static_cast<uint64_t>(n)};
  ta.storage = make_owned(n * sizeof(float));
  std::memcpy(ta.data_mut<float>(), a, n * sizeof(float));

  Tensor tb;
  tb.dtype = DType::Float32;
  tb.sizes = {1, static_cast<uint64_t>(n)};
  tb.storage = make_owned(n * sizeof(float));
  std::memcpy(tb.data_mut<float>(), b, n * sizeof(float));

  std::vector<Tensor> inputs;
  inputs.push_back(std::move(ta));
  inputs.push_back(std::move(tb));

  auto outputs_result = executor.run({inputs.data(), inputs.size()});
  EXPECT_TRUE(outputs_result.ok());
  return {std::move(executor), std::move(*outputs_result)};
}

TEST(TestE2E, subtract) {
  float a[] = {10, 20, 30, 40};
  float b[] = {1, 2, 3, 4};
  auto [executor, outputs] = run_binary_op(Operator::Subtract, a, b, 4);
  auto* d = outputs[0].data_const<float>();
  EXPECT_FLOAT_EQ(d[0], 9.0f);
  EXPECT_FLOAT_EQ(d[1], 18.0f);
  EXPECT_FLOAT_EQ(d[2], 27.0f);
  EXPECT_FLOAT_EQ(d[3], 36.0f);
}

TEST(TestE2E, multiply) {
  float a[] = {2, 3, 4, 5};
  float b[] = {10, 20, 30, 40};
  auto [executor, outputs] = run_binary_op(Operator::Multiply, a, b, 4);
  auto* d = outputs[0].data_const<float>();
  EXPECT_FLOAT_EQ(d[0], 20.0f);
  EXPECT_FLOAT_EQ(d[1], 60.0f);
  EXPECT_FLOAT_EQ(d[2], 120.0f);
  EXPECT_FLOAT_EQ(d[3], 200.0f);
}

TEST(TestE2E, divide) {
  float a[] = {10, 20, 30, 40};
  float b[] = {2, 4, 5, 8};
  auto [executor, outputs] = run_binary_op(Operator::Divide, a, b, 4);
  auto* d = outputs[0].data_const<float>();
  EXPECT_FLOAT_EQ(d[0], 5.0f);
  EXPECT_FLOAT_EQ(d[1], 5.0f);
  EXPECT_FLOAT_EQ(d[2], 6.0f);
  EXPECT_FLOAT_EQ(d[3], 5.0f);
}

TEST(TestE2E, maximum) {
  float a[] = {1, 20, 3, 40};
  float b[] = {10, 2, 30, 4};
  auto [executor, outputs] = run_binary_op(Operator::Maximum, a, b, 4);
  auto* d = outputs[0].data_const<float>();
  EXPECT_FLOAT_EQ(d[0], 10.0f);
  EXPECT_FLOAT_EQ(d[1], 20.0f);
  EXPECT_FLOAT_EQ(d[2], 30.0f);
  EXPECT_FLOAT_EQ(d[3], 40.0f);
}

TEST(TestE2E, minimum) {
  float a[] = {1, 20, 3, 40};
  float b[] = {10, 2, 30, 4};
  auto [executor, outputs] = run_binary_op(Operator::Minimum, a, b, 4);
  auto* d = outputs[0].data_const<float>();
  EXPECT_FLOAT_EQ(d[0], 1.0f);
  EXPECT_FLOAT_EQ(d[1], 2.0f);
  EXPECT_FLOAT_EQ(d[2], 3.0f);
  EXPECT_FLOAT_EQ(d[3], 4.0f);
}

TEST(TestE2E, copysign) {
  float a[] = {5, -5, 5, -5};
  float b[] = {1, -1, -1, 1};
  auto [executor, outputs] = run_binary_op(Operator::CopySign, a, b, 4);
  auto* d = outputs[0].data_const<float>();
  EXPECT_FLOAT_EQ(d[0], 5.0f);
  EXPECT_FLOAT_EQ(d[1], -5.0f);
  EXPECT_FLOAT_EQ(d[2], -5.0f);
  EXPECT_FLOAT_EQ(d[3], 5.0f);
}

TEST(TestE2E, squared_difference) {
  float a[] = {5, 10, 3, 8};
  float b[] = {2, 7, 1, 4};
  auto [executor, outputs] =
      run_binary_op(Operator::SquaredDifference, a, b, 4);
  auto* d = outputs[0].data_const<float>();
  EXPECT_FLOAT_EQ(d[0], 9.0f);
  EXPECT_FLOAT_EQ(d[1], 9.0f);
  EXPECT_FLOAT_EQ(d[2], 4.0f);
  EXPECT_FLOAT_EQ(d[3], 16.0f);
}

TEST(TestE2E, prelu) {
  float a[] = {1, -2, 3, -4};
  float b[] = {0.5f, 0.5f, 0.5f, 0.5f};
  auto [executor, outputs] = run_binary_op(Operator::PReLU, a, b, 4);
  auto* d = outputs[0].data_const<float>();
  EXPECT_FLOAT_EQ(d[0], 1.0f);
  EXPECT_FLOAT_EQ(d[1], -1.0f);
  EXPECT_FLOAT_EQ(d[2], 3.0f);
  EXPECT_FLOAT_EQ(d[3], -2.0f);
}

TEST(TestE2E, add_dynamic_shape) {
  // Build graph: output = A + B with shape [1, s0] (dynamic second dim).
  auto builder = GraphBuilder();
  auto s0 = builder.createSymInt();

  auto spec = TensorSpec{
      .dtype = DType::Float32,
      .sizes = {DimSizeSpec::constant(1), DimSizeSpec::sym(s0)}};

  auto a = builder.createInput(spec);
  auto b = builder.createInput(spec);
  auto add = builder.createOperator(Operator::Add, spec, a, b);
  builder.createOutput(add);

  auto graph = builder.build();
  auto executor_result = Executor::build(graph);
  ASSERT_TRUE(executor_result.ok());
  auto& executor = *executor_result;

  // First run: n=4
  {
    float da[] = {1, 2, 3, 4};
    float db[] = {10, 20, 30, 40};

    Tensor ta;
    ta.dtype = DType::Float32;
    ta.sizes = {1, 4};
    ta.storage = make_owned(4 * sizeof(float));
    std::memcpy(ta.data_mut<float>(), da, sizeof(da));

    Tensor tb;
    tb.dtype = DType::Float32;
    tb.sizes = {1, 4};
    tb.storage = make_owned(4 * sizeof(float));
    std::memcpy(tb.data_mut<float>(), db, sizeof(db));

    std::vector<Tensor> inputs;
    inputs.push_back(std::move(ta));
    inputs.push_back(std::move(tb));

    auto outputs_result = executor.run({inputs.data(), inputs.size()});
    ASSERT_TRUE(outputs_result.ok());
    auto& outputs = *outputs_result;
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0].sizes, (std::vector<uint64_t>{1, 4}));
    auto* out = outputs[0].data_const<float>();
    EXPECT_FLOAT_EQ(out[0], 11.0f);
    EXPECT_FLOAT_EQ(out[1], 22.0f);
    EXPECT_FLOAT_EQ(out[2], 33.0f);
    EXPECT_FLOAT_EQ(out[3], 44.0f);
  }

  // Second run: n=2 (different dynamic size, same executor)
  {
    float da[] = {100, 200};
    float db[] = {5, 6};

    Tensor ta;
    ta.dtype = DType::Float32;
    ta.sizes = {1, 2};
    ta.storage = make_owned(2 * sizeof(float));
    std::memcpy(ta.data_mut<float>(), da, sizeof(da));

    Tensor tb;
    tb.dtype = DType::Float32;
    tb.sizes = {1, 2};
    tb.storage = make_owned(2 * sizeof(float));
    std::memcpy(tb.data_mut<float>(), db, sizeof(db));

    std::vector<Tensor> inputs;
    inputs.push_back(std::move(ta));
    inputs.push_back(std::move(tb));

    auto outputs_result = executor.run({inputs.data(), inputs.size()});
    ASSERT_TRUE(outputs_result.ok());
    auto& outputs = *outputs_result;
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0].sizes, (std::vector<uint64_t>{1, 2}));
    auto* out = outputs[0].data_const<float>();
    EXPECT_FLOAT_EQ(out[0], 105.0f);
    EXPECT_FLOAT_EQ(out[1], 206.0f);
  }
}

TEST(TestE2E, three_adds_two_outputs) {
  auto builder = GraphBuilder();

  auto spec = TensorSpec{
      .dtype = DType::Float32,
      .sizes = {DimSizeSpec::constant(1), DimSizeSpec::constant(4)}};

  auto a = builder.createInput(spec);
  auto b = builder.createInput(spec);
  auto c = builder.createInput(spec);
  auto add0 = builder.createOperator(Operator::Add, spec, a, b);
  auto add1 = builder.createOperator(Operator::Add, spec, add0, c);
  auto add2 = builder.createOperator(Operator::Add, spec, add0, add1);
  builder.createOutput(add0);
  builder.createOutput(add2);

  auto graph = builder.build();
  auto executor_result = Executor::build(graph);
  ASSERT_TRUE(executor_result.ok());
  auto& executor = *executor_result;

  Tensor ta;
  ta.dtype = DType::Float32;
  ta.sizes = {1, 4};
  ta.storage = make_owned(4 * sizeof(float));
  float* da = ta.data_mut<float>();
  da[0] = 1.0f;
  da[1] = 2.0f;
  da[2] = 3.0f;
  da[3] = 4.0f;

  Tensor tb;
  tb.dtype = DType::Float32;
  tb.sizes = {1, 4};
  tb.storage = make_owned(4 * sizeof(float));
  float* db = tb.data_mut<float>();
  db[0] = 10.0f;
  db[1] = 20.0f;
  db[2] = 30.0f;
  db[3] = 40.0f;

  Tensor tc;
  tc.dtype = DType::Float32;
  tc.sizes = {1, 4};
  tc.storage = make_owned(4 * sizeof(float));
  float* dc = tc.data_mut<float>();
  dc[0] = 100.0f;
  dc[1] = 200.0f;
  dc[2] = 300.0f;
  dc[3] = 400.0f;

  std::vector<Tensor> inputs;
  inputs.push_back(std::move(ta));
  inputs.push_back(std::move(tb));
  inputs.push_back(std::move(tc));

  auto outputs_result = executor.run({inputs.data(), inputs.size()});
  ASSERT_TRUE(outputs_result.ok());
  auto& outputs = *outputs_result;

  // Add0 = A + B = {11, 22, 33, 44}
  ASSERT_EQ(outputs.size(), 2);
  ASSERT_EQ(outputs[0].sizes, (std::vector<uint64_t>{1, 4}));
  auto* out0 = outputs[0].data_const<float>();
  EXPECT_FLOAT_EQ(out0[0], 11.0f);
  EXPECT_FLOAT_EQ(out0[1], 22.0f);
  EXPECT_FLOAT_EQ(out0[2], 33.0f);
  EXPECT_FLOAT_EQ(out0[3], 44.0f);

  // Add1 = Add0 + C = {111, 222, 333, 444}
  // Add2 = Add0 + Add1 = {122, 244, 366, 488}
  ASSERT_EQ(outputs[1].sizes, (std::vector<uint64_t>{1, 4}));
  auto* out1 = outputs[1].data_const<float>();
  EXPECT_FLOAT_EQ(out1[0], 122.0f);
  EXPECT_FLOAT_EQ(out1[1], 244.0f);
  EXPECT_FLOAT_EQ(out1[2], 366.0f);
  EXPECT_FLOAT_EQ(out1[3], 488.0f);
}

TEST(TestE2E, linear) {
  auto builder = GraphBuilder();

  auto input_spec = TensorSpec{
      .dtype = DType::Float32,
      .sizes = {DimSizeSpec::constant(1), DimSizeSpec::constant(3)}};
  auto output_spec = TensorSpec{
      .dtype = DType::Float32,
      .sizes = {DimSizeSpec::constant(1), DimSizeSpec::constant(2)}};

  auto input = builder.createInput(input_spec);

  auto filter_tensor = std::make_shared<Tensor>();
  filter_tensor->dtype = DType::Float32;
  filter_tensor->sizes = {2, 3};
  filter_tensor->storage = make_owned(6 * sizeof(float));
  float* fw = filter_tensor->data_mut<float>();
  fw[0] = 1;
  fw[1] = 0;
  fw[2] = 0;
  fw[3] = 0;
  fw[4] = 1;
  fw[5] = 0;
  auto filter = builder.createConstant(filter_tensor);

  auto bias_tensor = std::make_shared<Tensor>();
  bias_tensor->dtype = DType::Float32;
  bias_tensor->sizes = {2};
  bias_tensor->storage = make_owned(2 * sizeof(float));
  float* bw = bias_tensor->data_mut<float>();
  bw[0] = 10;
  bw[1] = 20;
  auto bias = builder.createConstant(bias_tensor);

  auto out = builder.createOperator(
      Operator::Linear, output_spec, {input, filter, bias});
  builder.createOutput(out);

  auto graph = builder.build();
  auto executor_result = Executor::build(graph);
  ASSERT_TRUE(executor_result.ok());
  auto& executor = *executor_result;

  Tensor ti;
  ti.dtype = DType::Float32;
  ti.sizes = {1, 3};
  ti.storage = make_owned(3 * sizeof(float));
  float* di = ti.data_mut<float>();
  di[0] = 1;
  di[1] = 2;
  di[2] = 3;

  std::vector<Tensor> inputs;
  inputs.push_back(std::move(ti));

  auto outputs_result = executor.run({inputs.data(), inputs.size()});
  ASSERT_TRUE(outputs_result.ok());
  auto& outputs = *outputs_result;

  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(outputs[0].sizes, (std::vector<uint64_t>{1, 2}));
  auto* d = outputs[0].data_const<float>();
  EXPECT_FLOAT_EQ(d[0], 11.0f); // 1*1+2*0+3*0+10
  EXPECT_FLOAT_EQ(d[1], 22.0f); // 1*0+2*1+3*0+20
}

TEST(TestE2E, linear_no_bias) {
  auto builder = GraphBuilder();

  auto input_spec = TensorSpec{
      .dtype = DType::Float32,
      .sizes = {DimSizeSpec::constant(1), DimSizeSpec::constant(3)}};
  auto output_spec = TensorSpec{
      .dtype = DType::Float32,
      .sizes = {DimSizeSpec::constant(1), DimSizeSpec::constant(2)}};

  auto input = builder.createInput(input_spec);

  auto filter_tensor = std::make_shared<Tensor>();
  filter_tensor->dtype = DType::Float32;
  filter_tensor->sizes = {2, 3};
  filter_tensor->storage = make_owned(6 * sizeof(float));
  float* fw = filter_tensor->data_mut<float>();
  fw[0] = 1;
  fw[1] = 0;
  fw[2] = 0;
  fw[3] = 0;
  fw[4] = 1;
  fw[5] = 0;
  auto filter = builder.createConstant(filter_tensor);

  auto out = builder.createOperator(
      Operator::Linear, output_spec, {input, filter, ValueHandle::null()});
  builder.createOutput(out);

  auto graph = builder.build();
  auto executor_result = Executor::build(graph);
  ASSERT_TRUE(executor_result.ok());
  auto& executor = *executor_result;

  Tensor ti;
  ti.dtype = DType::Float32;
  ti.sizes = {1, 3};
  ti.storage = make_owned(3 * sizeof(float));
  float* di = ti.data_mut<float>();
  di[0] = 1;
  di[1] = 2;
  di[2] = 3;

  std::vector<Tensor> inputs;
  inputs.push_back(std::move(ti));

  auto outputs_result = executor.run({inputs.data(), inputs.size()});
  ASSERT_TRUE(outputs_result.ok());
  auto& outputs = *outputs_result;

  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(outputs[0].sizes, (std::vector<uint64_t>{1, 2}));
  auto* d = outputs[0].data_const<float>();
  EXPECT_FLOAT_EQ(d[0], 1.0f); // 1*1+2*0+3*0
  EXPECT_FLOAT_EQ(d[1], 2.0f); // 1*0+2*1+3*0
}

TEST(TestE2E, linear_no_bias_larger) {
  // Larger dimensions to exercise tiled packing in the in-tree SME kernel path.
  // On non-SME hardware this goes through XNNPACK and validates the same math.
  constexpr size_t M = 4;
  constexpr size_t K = 8;
  constexpr size_t N = 6;

  auto builder = GraphBuilder();

  auto input_spec = TensorSpec{
      .dtype = DType::Float32,
      .sizes = {DimSizeSpec::constant(M), DimSizeSpec::constant(K)}};
  auto output_spec = TensorSpec{
      .dtype = DType::Float32,
      .sizes = {DimSizeSpec::constant(M), DimSizeSpec::constant(N)}};

  auto input = builder.createInput(input_spec);

  // Weight tensor: N x K (transposed convention for Linear).
  auto filter_tensor = std::make_shared<Tensor>();
  filter_tensor->dtype = DType::Float32;
  filter_tensor->sizes = {N, K};
  filter_tensor->storage = make_owned(N * K * sizeof(float));
  float* fw = filter_tensor->data_mut<float>();
  for (size_t i = 0; i < N * K; i++) {
    fw[i] = static_cast<float>(i + 1) * 0.1f;
  }
  auto filter = builder.createConstant(filter_tensor);

  auto out = builder.createOperator(
      Operator::Linear, output_spec, {input, filter, ValueHandle::null()});
  builder.createOutput(out);

  auto graph = builder.build();
  auto executor_result = Executor::build(graph);
  ASSERT_TRUE(executor_result.ok());
  auto& executor = *executor_result;

  // Input: M x K
  Tensor ti;
  ti.dtype = DType::Float32;
  ti.sizes = {M, K};
  ti.storage = make_owned(M * K * sizeof(float));
  float* di = ti.data_mut<float>();
  for (size_t i = 0; i < M * K; i++) {
    di[i] = static_cast<float>(i + 1);
  }

  std::vector<Tensor> inputs;
  inputs.push_back(std::move(ti));

  auto outputs_result = executor.run({inputs.data(), inputs.size()});
  ASSERT_TRUE(outputs_result.ok());
  auto& outputs = *outputs_result;

  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(outputs[0].sizes, (std::vector<uint64_t>{M, N}));

  // Reference: out[m][n] = sum_k(input[m][k] * weight[n][k])
  auto* d = outputs[0].data_const<float>();
  for (size_t m = 0; m < M; m++) {
    for (size_t n = 0; n < N; n++) {
      float expected = 0;
      for (size_t k = 0; k < K; k++) {
        expected += static_cast<float>(m * K + k + 1) * fw[n * K + k];
      }
      EXPECT_NEAR(d[m * N + n], expected, std::abs(expected) * 1e-5f)
          << "at (" << m << ", " << n << ")";
    }
  }
}

TEST(TestE2E, linear_qint8_static_dequantized) {
  auto builder = GraphBuilder();

  auto input_spec = TensorSpec{
      .dtype = DType::QInt8,
      .sizes = {DimSizeSpec::constant(1), DimSizeSpec::constant(3)},
      .quant_params = qint8_per_tensor_sym(1.0f),
  };
  auto quant_output_spec = TensorSpec{
      .dtype = DType::QInt8,
      .sizes = {DimSizeSpec::constant(1), DimSizeSpec::constant(2)},
      .quant_params = qint8_per_tensor_sym(1.0f),
  };
  auto float_output_spec = TensorSpec{
      .dtype = DType::Float32,
      .sizes = {DimSizeSpec::constant(1), DimSizeSpec::constant(2)},
  };

  auto input = builder.createInput(input_spec);

  auto filter_tensor = std::make_shared<Tensor>();
  filter_tensor->dtype = DType::QInt8;
  filter_tensor->sizes = {2, 3};
  filter_tensor->storage = make_owned(6);
  auto* fw = filter_tensor->data_mut<int8_t>();
  fw[0] = 1;
  fw[1] = 0;
  fw[2] = 0;
  fw[3] = 0;
  fw[4] = 1;
  fw[5] = 0;
  filter_tensor->aux_storage.push_back(make_owned(2 * sizeof(float)));
  auto* scales = static_cast<float*>(filter_tensor->aux_storage[0].data);
  scales[0] = 1.0f;
  scales[1] = 1.0f;
  auto filter = builder.createConstant(filter_tensor, qint8_per_channel_sym(0));

  auto linear_out = builder.createOperator(
      Operator::Linear,
      quant_output_spec,
      {input, filter, ValueHandle::null()});
  auto dequant_out = builder.createOperator(
      Operator::Dequantize, float_output_spec, {linear_out});
  builder.createOutput(dequant_out);

  auto graph = builder.build();
  auto executor_result = Executor::build(graph);
  ASSERT_TRUE(executor_result.ok());
  auto& executor = *executor_result;

  Tensor ti;
  ti.dtype = DType::QInt8;
  ti.sizes = {1, 3};
  ti.storage = make_owned(3);
  auto* di = ti.data_mut<int8_t>();
  di[0] = 1;
  di[1] = 2;
  di[2] = 3;

  std::vector<Tensor> inputs;
  inputs.push_back(std::move(ti));

  auto outputs_result = executor.run({inputs.data(), inputs.size()});
  ASSERT_TRUE(outputs_result.ok());
  auto& outputs = *outputs_result;

  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(outputs[0].sizes, (std::vector<uint64_t>{1, 2}));
  auto* d = outputs[0].data_const<float>();
  EXPECT_FLOAT_EQ(d[0], 1.0f);
  EXPECT_FLOAT_EQ(d[1], 2.0f);
}

TEST(TestE2E, linear_qcint4_static_dequantized) {
  auto builder = GraphBuilder();

  auto input_spec = TensorSpec{
      .dtype = DType::QInt8,
      .sizes = {DimSizeSpec::constant(1), DimSizeSpec::constant(4)},
      .quant_params = qint8_per_tensor_sym(1.0f),
  };
  auto quant_output_spec = TensorSpec{
      .dtype = DType::QInt8,
      .sizes = {DimSizeSpec::constant(1), DimSizeSpec::constant(2)},
      .quant_params = qint8_per_tensor_sym(1.0f),
  };
  auto float_output_spec = TensorSpec{
      .dtype = DType::Float32,
      .sizes = {DimSizeSpec::constant(1), DimSizeSpec::constant(2)},
  };

  auto input = builder.createInput(input_spec);

  // Weight [2, 4], qcint4 per-channel (axis 0), scales = [1, 1].
  // Logical values: row0 = [1,0,0,0], row1 = [0,1,0,0].
  // qcint4 zero_point is 8, so stored nibble = value + 8.
  //   row0 nibbles = [9,8,8,8], row1 nibbles = [8,9,8,8].
  // Packed 2 nibbles/byte, element i in byte i/2 (low nibble for even i).
  auto filter_tensor = std::make_shared<Tensor>();
  filter_tensor->dtype = DType::QInt4;
  filter_tensor->sizes = {2, 4};
  filter_tensor->storage = make_owned(4); // 8 nibbles -> 4 bytes
  auto* fw = filter_tensor->data_mut<uint8_t>();
  fw[0] = (8 << 4) | 9; // row0 nibbles 0,1 = 9,8
  fw[1] = (8 << 4) | 8; // row0 nibbles 2,3 = 8,8
  fw[2] = (9 << 4) | 8; // row1 nibbles 0,1 = 8,9
  fw[3] = (8 << 4) | 8; // row1 nibbles 2,3 = 8,8
  filter_tensor->aux_storage.push_back(make_owned(2 * sizeof(float)));
  auto* scales = static_cast<float*>(filter_tensor->aux_storage[0].data);
  scales[0] = 1.0f;
  scales[1] = 1.0f;
  auto filter = builder.createConstant(filter_tensor, qint8_per_channel_sym(0));

  auto linear_out = builder.createOperator(
      Operator::Linear,
      quant_output_spec,
      {input, filter, ValueHandle::null()});
  auto dequant_out = builder.createOperator(
      Operator::Dequantize, float_output_spec, {linear_out});
  builder.createOutput(dequant_out);

  auto graph = builder.build();
  auto executor_result = Executor::build(graph);
  ASSERT_TRUE(executor_result.ok());
  auto& executor = *executor_result;

  Tensor ti;
  ti.dtype = DType::QInt8;
  ti.sizes = {1, 4};
  ti.storage = make_owned(4);
  auto* di = ti.data_mut<int8_t>();
  di[0] = 1;
  di[1] = 2;
  di[2] = 3;
  di[3] = 4;

  std::vector<Tensor> inputs;
  inputs.push_back(std::move(ti));

  auto outputs_result = executor.run({inputs.data(), inputs.size()});
  ASSERT_TRUE(outputs_result.ok());
  auto& outputs = *outputs_result;

  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(outputs[0].sizes, (std::vector<uint64_t>{1, 2}));
  auto* d = outputs[0].data_const<float>();
  // out[0] = row0 . input = 1*1 = 1 ; out[1] = row1 . input = 1*2 = 2
  EXPECT_FLOAT_EQ(d[0], 1.0f);
  EXPECT_FLOAT_EQ(d[1], 2.0f);
}

TEST(TestE2E, linear_qd8_qcint4_dynamic) {
  auto builder = GraphBuilder();

  // Float input [2, 4]; dynamically quantized to qdint8 before the matmul.
  auto float_input_spec = TensorSpec{
      .dtype = DType::Float32,
      .sizes = {DimSizeSpec::constant(2), DimSizeSpec::constant(4)},
  };
  auto dyn_quant_spec = TensorSpec{
      .dtype = DType::QInt8,
      .sizes = {DimSizeSpec::constant(2), DimSizeSpec::constant(4)},
      .quant_params = PerRowQuantParams{.axis = -1, .is_dynamic = true},
  };
  auto float_output_spec = TensorSpec{
      .dtype = DType::Float32,
      .sizes = {DimSizeSpec::constant(2), DimSizeSpec::constant(2)},
  };

  auto input = builder.createInput(float_input_spec);
  auto qinput =
      builder.createOperator(Operator::Quantize, dyn_quant_spec, {input});

  // Weight [2, 4], qcint4 per-channel (axis 0), non-trivial signed nibbles
  // and per-channel scales to exercise sign, high/low nibble order, and scale.
  //   row0 logical = [ 3, -2,  1, 0], scale0 = 2.0
  //   row1 logical = [-4,  7, -1, 2], scale1 = 0.5
  //   nibble = logical + 8:
  //   row0 = [11, 6, 9, 8], row1 = [4, 15, 7, 10]
  auto filter_tensor = std::make_shared<Tensor>();
  filter_tensor->dtype = DType::QInt4;
  filter_tensor->sizes = {2, 4};
  filter_tensor->storage = make_owned(4);
  auto* fw = filter_tensor->data_mut<uint8_t>();
  fw[0] = (6 << 4) | 11; // row0 nibbles 0,1
  fw[1] = (8 << 4) | 9; // row0 nibbles 2,3
  fw[2] = (15 << 4) | 4; // row1 nibbles 0,1
  fw[3] = (10 << 4) | 7; // row1 nibbles 2,3
  filter_tensor->aux_storage.push_back(make_owned(2 * sizeof(float)));
  auto* scales = static_cast<float*>(filter_tensor->aux_storage[0].data);
  scales[0] = 2.0f;
  scales[1] = 0.5f;
  auto filter = builder.createConstant(filter_tensor, qint8_per_channel_sym(0));

  auto linear_out = builder.createOperator(
      Operator::Linear,
      float_output_spec,
      {qinput, filter, ValueHandle::null()});
  builder.createOutput(linear_out);

  auto graph = builder.build();
  auto executor_result = Executor::build(graph);
  ASSERT_TRUE(executor_result.ok());
  auto& executor = *executor_result;

  Tensor ti;
  ti.dtype = DType::Float32;
  ti.sizes = {2, 4};
  ti.storage = make_owned(8 * sizeof(float));
  auto* di = ti.data_mut<float>();
  // row0 = [1,2,3,4], row1 = [2,1,0,-1]
  di[0] = 1;
  di[1] = 2;
  di[2] = 3;
  di[3] = 4;
  di[4] = 2;
  di[5] = 1;
  di[6] = 0;
  di[7] = -1;

  std::vector<Tensor> inputs;
  inputs.push_back(std::move(ti));

  auto outputs_result = executor.run({inputs.data(), inputs.size()});
  ASSERT_TRUE(outputs_result.ok());
  auto& outputs = *outputs_result;

  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(outputs[0].sizes, (std::vector<uint64_t>{2, 2}));
  auto* d = outputs[0].data_const<float>();
  // input row0 = [1,2,3,4]:
  //   out[0][0] = 2.0*(3*1 - 2*2 + 1*3 + 0*4) = 2.0*2  = 4.0
  //   out[0][1] = 0.5*(-4*1 + 7*2 - 1*3 + 2*4) = 0.5*15 = 7.5
  // input row1 = [2,1,0,-1]:
  //   out[1][0] = 2.0*(3*2 - 2*1 + 1*0 + 0*-1) = 2.0*4  = 8.0
  //   out[1][1] = 0.5*(-4*2 + 7*1 - 1*0 + 2*-1) = 0.5*-3 = -1.5
  EXPECT_NEAR(d[0], 4.0f, 1e-1f);
  EXPECT_NEAR(d[1], 7.5f, 1e-1f);
  EXPECT_NEAR(d[2], 8.0f, 1e-1f);
  EXPECT_NEAR(d[3], -1.5f, 1e-1f);
}

TEST(TestE2E, linear_qint8_static_requantized) {
  auto builder = GraphBuilder();

  auto input_spec = TensorSpec{
      .dtype = DType::QInt8,
      .sizes = {DimSizeSpec::constant(1), DimSizeSpec::constant(3)},
      .quant_params = qint8_per_tensor_sym(1.0f),
  };
  auto output_spec = TensorSpec{
      .dtype = DType::QInt8,
      .sizes = {DimSizeSpec::constant(1), DimSizeSpec::constant(2)},
      .quant_params = qint8_per_tensor_sym(1.0f),
  };

  auto input = builder.createInput(input_spec);

  auto filter_tensor = std::make_shared<Tensor>();
  filter_tensor->dtype = DType::QInt8;
  filter_tensor->sizes = {2, 3};
  filter_tensor->storage = make_owned(6);
  auto* fw = filter_tensor->data_mut<int8_t>();
  fw[0] = 1;
  fw[1] = 0;
  fw[2] = 0;
  fw[3] = 0;
  fw[4] = 1;
  fw[5] = 0;
  filter_tensor->aux_storage.push_back(make_owned(2 * sizeof(float)));
  auto* scales = static_cast<float*>(filter_tensor->aux_storage[0].data);
  scales[0] = 1.0f;
  scales[1] = 1.0f;
  auto filter = builder.createConstant(filter_tensor, qint8_per_channel_sym(0));

  auto out = builder.createOperator(
      Operator::Linear, output_spec, {input, filter, ValueHandle::null()});
  builder.createOutput(out);

  auto graph = builder.build();
  auto executor_result = Executor::build(graph);
  ASSERT_TRUE(executor_result.ok());
  auto& executor = *executor_result;

  Tensor ti;
  ti.dtype = DType::QInt8;
  ti.sizes = {1, 3};
  ti.storage = make_owned(3);
  auto* di = ti.data_mut<int8_t>();
  di[0] = 1;
  di[1] = 2;
  di[2] = 3;

  std::vector<Tensor> inputs;
  inputs.push_back(std::move(ti));

  auto outputs_result = executor.run({inputs.data(), inputs.size()});
  ASSERT_TRUE(outputs_result.ok());
  auto& outputs = *outputs_result;

  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(outputs[0].sizes, (std::vector<uint64_t>{1, 2}));
  auto* d = outputs[0].data_const<int8_t>();
  EXPECT_EQ(d[0], 1);
  EXPECT_EQ(d[1], 2);
}

TEST(TestE2E, dequantize_quint8) {
  auto builder = GraphBuilder();

  auto input_spec = TensorSpec{
      .dtype = DType::QUInt8,
      .sizes = {DimSizeSpec::constant(1), DimSizeSpec::constant(4)},
      .quant_params = quint8_per_tensor_asym(0.5f, 1),
  };
  auto output_spec = TensorSpec{
      .dtype = DType::Float32,
      .sizes = {DimSizeSpec::constant(1), DimSizeSpec::constant(4)},
  };

  auto input = builder.createInput(input_spec);
  auto out = builder.createOperator(Operator::Dequantize, output_spec, {input});
  builder.createOutput(out);

  auto graph = builder.build();
  auto executor_result = Executor::build(graph);
  ASSERT_TRUE(executor_result.ok());
  auto& executor = *executor_result;

  Tensor ti;
  ti.dtype = DType::QUInt8;
  ti.sizes = {1, 4};
  ti.storage = make_owned(4);
  auto* di = ti.data_mut<uint8_t>();
  di[0] = 0;
  di[1] = 1;
  di[2] = 2;
  di[3] = 3;

  std::vector<Tensor> inputs;
  inputs.push_back(std::move(ti));

  auto outputs_result = executor.run({inputs.data(), inputs.size()});
  ASSERT_TRUE(outputs_result.ok());
  auto& outputs = *outputs_result;

  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(outputs[0].sizes, (std::vector<uint64_t>{1, 4}));
  auto* d = outputs[0].data_const<float>();
  EXPECT_FLOAT_EQ(d[0], -0.5f);
  EXPECT_FLOAT_EQ(d[1], 0.0f);
  EXPECT_FLOAT_EQ(d[2], 0.5f);
  EXPECT_FLOAT_EQ(d[3], 1.0f);
}

TEST(TestE2E, quantize_quint8) {
  auto builder = GraphBuilder();

  auto input_spec = TensorSpec{
      .dtype = DType::Float32,
      .sizes = {DimSizeSpec::constant(1), DimSizeSpec::constant(4)},
  };
  auto output_spec = TensorSpec{
      .dtype = DType::QUInt8,
      .sizes = {DimSizeSpec::constant(1), DimSizeSpec::constant(4)},
      .quant_params = quint8_per_tensor_asym(0.5f, 1),
  };

  auto input = builder.createInput(input_spec);
  auto out = builder.createOperator(Operator::Quantize, output_spec, {input});
  builder.createOutput(out);

  auto graph = builder.build();
  auto executor_result = Executor::build(graph);
  ASSERT_TRUE(executor_result.ok());
  auto& executor = *executor_result;

  Tensor ti;
  ti.dtype = DType::Float32;
  ti.sizes = {1, 4};
  ti.storage = make_owned(4 * sizeof(float));
  auto* di = ti.data_mut<float>();
  di[0] = -0.5f;
  di[1] = 0.0f;
  di[2] = 0.5f;
  di[3] = 1.0f;

  std::vector<Tensor> inputs;
  inputs.push_back(std::move(ti));

  auto outputs_result = executor.run({inputs.data(), inputs.size()});
  ASSERT_TRUE(outputs_result.ok());
  auto& outputs = *outputs_result;

  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(outputs[0].sizes, (std::vector<uint64_t>{1, 4}));
  auto* d = outputs[0].data_const<uint8_t>();
  EXPECT_EQ(d[0], 0);
  EXPECT_EQ(d[1], 1);
  EXPECT_EQ(d[2], 2);
  EXPECT_EQ(d[3], 3);
}
