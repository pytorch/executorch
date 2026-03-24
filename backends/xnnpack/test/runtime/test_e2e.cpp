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

TEST(TestE2E, add) {
    // Build graph: output = input_a + input_b
    // Shape: [1, 4] float32, static sizes.
    auto builder = GraphBuilder();

    auto spec = TensorSpec {
        .dtype = DType::Float32,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(4) }
    };

    auto a = builder.createInput(spec);
    auto b = builder.createInput(spec);
    auto add = builder.createOperator(Operator::Add, spec, a, b);
    builder.createOutput(add);

    auto graph = builder.build();
    auto executor = Executor::build(graph);

    // Create input tensors.
    Tensor ta;
    ta.dtype = DType::Float32;
    ta.sizes = {1, 4};
    ta.storage = Storage::create_owned(4 * sizeof(float));
    float* da = ta.data_mut<float>();
    da[0] = 1.0f; da[1] = 2.0f; da[2] = 3.0f; da[3] = 4.0f;

    Tensor tb;
    tb.dtype = DType::Float32;
    tb.sizes = {1, 4};
    tb.storage = Storage::create_owned(4 * sizeof(float));
    float* db = tb.data_mut<float>();
    db[0] = 10.0f; db[1] = 20.0f; db[2] = 30.0f; db[3] = 40.0f;

    std::vector<Tensor> inputs;
    inputs.push_back(std::move(ta));
    inputs.push_back(std::move(tb));

    auto outputs = executor.run(inputs);

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

BinaryOpResult run_binary_op(Operator op, const float* a, const float* b, size_t n) {
    auto builder = GraphBuilder();
    auto spec = TensorSpec {
        .dtype = DType::Float32,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(static_cast<int64_t>(n)) }
    };

    auto ia = builder.createInput(spec);
    auto ib = builder.createInput(spec);
    auto out = builder.createOperator(op, spec, ia, ib);
    builder.createOutput(out);

    auto graph = builder.build();
    auto executor = Executor::build(graph);

    Tensor ta;
    ta.dtype = DType::Float32;
    ta.sizes = {1, static_cast<uint64_t>(n)};
    ta.storage = Storage::create_owned(n * sizeof(float));
    std::memcpy(ta.data_mut<float>(), a, n * sizeof(float));

    Tensor tb;
    tb.dtype = DType::Float32;
    tb.sizes = {1, static_cast<uint64_t>(n)};
    tb.storage = Storage::create_owned(n * sizeof(float));
    std::memcpy(tb.data_mut<float>(), b, n * sizeof(float));

    std::vector<Tensor> inputs;
    inputs.push_back(std::move(ta));
    inputs.push_back(std::move(tb));

    auto outputs = executor.run(inputs);
    return { std::move(executor), std::move(outputs) };
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
    auto [executor, outputs] = run_binary_op(Operator::SquaredDifference, a, b, 4);
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

    auto spec = TensorSpec {
        .dtype = DType::Float32,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::sym(s0.node) }
    };

    auto a = builder.createInput(spec);
    auto b = builder.createInput(spec);
    auto add = builder.createOperator(Operator::Add, spec, a, b);
    builder.createOutput(add);

    auto graph = builder.build();
    auto executor = Executor::build(graph);

    // First run: n=4
    {
        float da[] = {1, 2, 3, 4};
        float db[] = {10, 20, 30, 40};

        Tensor ta;
        ta.dtype = DType::Float32;
        ta.sizes = {1, 4};
        ta.storage = Storage::create_owned(4 * sizeof(float));
        std::memcpy(ta.data_mut<float>(), da, sizeof(da));

        Tensor tb;
        tb.dtype = DType::Float32;
        tb.sizes = {1, 4};
        tb.storage = Storage::create_owned(4 * sizeof(float));
        std::memcpy(tb.data_mut<float>(), db, sizeof(db));

        std::vector<Tensor> inputs;
        inputs.push_back(std::move(ta));
        inputs.push_back(std::move(tb));

        auto outputs = executor.run(inputs);
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
        ta.storage = Storage::create_owned(2 * sizeof(float));
        std::memcpy(ta.data_mut<float>(), da, sizeof(da));

        Tensor tb;
        tb.dtype = DType::Float32;
        tb.sizes = {1, 2};
        tb.storage = Storage::create_owned(2 * sizeof(float));
        std::memcpy(tb.data_mut<float>(), db, sizeof(db));

        std::vector<Tensor> inputs;
        inputs.push_back(std::move(ta));
        inputs.push_back(std::move(tb));

        auto outputs = executor.run(inputs);
        ASSERT_EQ(outputs.size(), 1);
        ASSERT_EQ(outputs[0].sizes, (std::vector<uint64_t>{1, 2}));
        auto* out = outputs[0].data_const<float>();
        EXPECT_FLOAT_EQ(out[0], 105.0f);
        EXPECT_FLOAT_EQ(out[1], 206.0f);
    }
}

TEST(TestE2E, three_adds_two_outputs) {
    auto builder = GraphBuilder();

    auto spec = TensorSpec {
        .dtype = DType::Float32,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(4) }
    };

    auto a = builder.createInput(spec);
    auto b = builder.createInput(spec);
    auto c = builder.createInput(spec);
    auto add0 = builder.createOperator(Operator::Add, spec, a, b);
    auto add1 = builder.createOperator(Operator::Add, spec, add0, c);
    auto add2 = builder.createOperator(Operator::Add, spec, add0, add1);
    builder.createOutput(add0);
    builder.createOutput(add2);

    auto graph = builder.build();
    auto executor = Executor::build(graph);

    Tensor ta;
    ta.dtype = DType::Float32;
    ta.sizes = {1, 4};
    ta.storage = Storage::create_owned(4 * sizeof(float));
    float* da = ta.data_mut<float>();
    da[0] = 1.0f; da[1] = 2.0f; da[2] = 3.0f; da[3] = 4.0f;

    Tensor tb;
    tb.dtype = DType::Float32;
    tb.sizes = {1, 4};
    tb.storage = Storage::create_owned(4 * sizeof(float));
    float* db = tb.data_mut<float>();
    db[0] = 10.0f; db[1] = 20.0f; db[2] = 30.0f; db[3] = 40.0f;

    Tensor tc;
    tc.dtype = DType::Float32;
    tc.sizes = {1, 4};
    tc.storage = Storage::create_owned(4 * sizeof(float));
    float* dc = tc.data_mut<float>();
    dc[0] = 100.0f; dc[1] = 200.0f; dc[2] = 300.0f; dc[3] = 400.0f;

    std::vector<Tensor> inputs;
    inputs.push_back(std::move(ta));
    inputs.push_back(std::move(tb));
    inputs.push_back(std::move(tc));

    auto outputs = executor.run(inputs);

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

    auto input_spec = TensorSpec {
        .dtype = DType::Float32,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(3) }
    };
    auto output_spec = TensorSpec {
        .dtype = DType::Float32,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(2) }
    };

    auto input = builder.createInput(input_spec);

    auto filter_tensor = std::make_shared<Tensor>();
    filter_tensor->dtype = DType::Float32;
    filter_tensor->sizes = {2, 3};
    filter_tensor->storage = Storage::create_owned(6 * sizeof(float));
    float* fw = filter_tensor->data_mut<float>();
    fw[0] = 1; fw[1] = 0; fw[2] = 0;
    fw[3] = 0; fw[4] = 1; fw[5] = 0;
    auto filter = builder.createConstant(filter_tensor);

    auto bias_tensor = std::make_shared<Tensor>();
    bias_tensor->dtype = DType::Float32;
    bias_tensor->sizes = {2};
    bias_tensor->storage = Storage::create_owned(2 * sizeof(float));
    float* bw = bias_tensor->data_mut<float>();
    bw[0] = 10; bw[1] = 20;
    auto bias = builder.createConstant(bias_tensor);

    auto out = builder.createOperator(Operator::Linear, output_spec, {input, filter, bias});
    builder.createOutput(out);

    auto graph = builder.build();
    auto executor = Executor::build(graph);

    Tensor ti;
    ti.dtype = DType::Float32;
    ti.sizes = {1, 3};
    ti.storage = Storage::create_owned(3 * sizeof(float));
    float* di = ti.data_mut<float>();
    di[0] = 1; di[1] = 2; di[2] = 3;

    std::vector<Tensor> inputs;
    inputs.push_back(std::move(ti));

    auto outputs = executor.run(inputs);

    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0].sizes, (std::vector<uint64_t>{1, 2}));
    auto* d = outputs[0].data_const<float>();
    EXPECT_FLOAT_EQ(d[0], 11.0f);  // 1*1+2*0+3*0+10
    EXPECT_FLOAT_EQ(d[1], 22.0f);  // 1*0+2*1+3*0+20
}

TEST(TestE2E, linear_no_bias) {
    auto builder = GraphBuilder();

    auto input_spec = TensorSpec {
        .dtype = DType::Float32,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(3) }
    };
    auto output_spec = TensorSpec {
        .dtype = DType::Float32,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(2) }
    };

    auto input = builder.createInput(input_spec);

    auto filter_tensor = std::make_shared<Tensor>();
    filter_tensor->dtype = DType::Float32;
    filter_tensor->sizes = {2, 3};
    filter_tensor->storage = Storage::create_owned(6 * sizeof(float));
    float* fw = filter_tensor->data_mut<float>();
    fw[0] = 1; fw[1] = 0; fw[2] = 0;
    fw[3] = 0; fw[4] = 1; fw[5] = 0;
    auto filter = builder.createConstant(filter_tensor);

    auto out = builder.createOperator(Operator::Linear, output_spec,
        {input, filter, ValueHandle::null()});
    builder.createOutput(out);

    auto graph = builder.build();
    auto executor = Executor::build(graph);

    Tensor ti;
    ti.dtype = DType::Float32;
    ti.sizes = {1, 3};
    ti.storage = Storage::create_owned(3 * sizeof(float));
    float* di = ti.data_mut<float>();
    di[0] = 1; di[1] = 2; di[2] = 3;

    std::vector<Tensor> inputs;
    inputs.push_back(std::move(ti));

    auto outputs = executor.run(inputs);

    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0].sizes, (std::vector<uint64_t>{1, 2}));
    auto* d = outputs[0].data_const<float>();
    EXPECT_FLOAT_EQ(d[0], 1.0f);   // 1*1+2*0+3*0
    EXPECT_FLOAT_EQ(d[1], 2.0f);   // 1*0+2*1+3*0
}

TEST(TestE2E, linear_qint8_static_dequantized) {
    auto builder = GraphBuilder();

    auto input_spec = TensorSpec {
        .dtype = DType::QInt8Sym,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(3) },
        .quant_params = qint8_per_tensor_sym(1.0f),
    };
    auto quant_output_spec = TensorSpec {
        .dtype = DType::QInt8Sym,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(2) },
        .quant_params = qint8_per_tensor_sym(1.0f),
    };
    auto float_output_spec = TensorSpec {
        .dtype = DType::Float32,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(2) },
    };

    auto input = builder.createInput(input_spec);

    auto filter_tensor = std::make_shared<Tensor>();
    filter_tensor->dtype = DType::QInt8Sym;
    filter_tensor->sizes = {2, 3};
    filter_tensor->storage = Storage::create_owned(6);
    auto* fw = filter_tensor->data_mut<int8_t>();
    fw[0] = 1; fw[1] = 0; fw[2] = 0;
    fw[3] = 0; fw[4] = 1; fw[5] = 0;
    filter_tensor->aux_storage.push_back(Storage::create_owned(2 * sizeof(float)));
    auto* scales = static_cast<float*>(filter_tensor->aux_storage[0].data);
    scales[0] = 1.0f; scales[1] = 1.0f;
    auto filter = builder.createConstant(filter_tensor, qint8_per_channel_sym(0));

    auto linear_out = builder.createOperator(Operator::Linear, quant_output_spec,
        {input, filter, ValueHandle::null()});
    auto dequant_out = builder.createOperator(Operator::Dequantize, float_output_spec,
        {linear_out});
    builder.createOutput(dequant_out);

    auto graph = builder.build();
    auto executor = Executor::build(graph);

    Tensor ti;
    ti.dtype = DType::QInt8Sym;
    ti.sizes = {1, 3};
    ti.storage = Storage::create_owned(3);
    auto* di = ti.data_mut<int8_t>();
    di[0] = 1; di[1] = 2; di[2] = 3;

    std::vector<Tensor> inputs;
    inputs.push_back(std::move(ti));

    auto outputs = executor.run(inputs);

    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0].sizes, (std::vector<uint64_t>{1, 2}));
    auto* d = outputs[0].data_const<float>();
    EXPECT_FLOAT_EQ(d[0], 1.0f);
    EXPECT_FLOAT_EQ(d[1], 2.0f);
}

TEST(TestE2E, linear_qint8_static_requantized) {
    auto builder = GraphBuilder();

    auto input_spec = TensorSpec {
        .dtype = DType::QInt8Sym,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(3) },
        .quant_params = qint8_per_tensor_sym(1.0f),
    };
    auto output_spec = TensorSpec {
        .dtype = DType::QInt8Sym,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(2) },
        .quant_params = qint8_per_tensor_sym(1.0f),
    };

    auto input = builder.createInput(input_spec);

    auto filter_tensor = std::make_shared<Tensor>();
    filter_tensor->dtype = DType::QInt8Sym;
    filter_tensor->sizes = {2, 3};
    filter_tensor->storage = Storage::create_owned(6);
    auto* fw = filter_tensor->data_mut<int8_t>();
    fw[0] = 1; fw[1] = 0; fw[2] = 0;
    fw[3] = 0; fw[4] = 1; fw[5] = 0;
    filter_tensor->aux_storage.push_back(Storage::create_owned(2 * sizeof(float)));
    auto* scales = static_cast<float*>(filter_tensor->aux_storage[0].data);
    scales[0] = 1.0f; scales[1] = 1.0f;
    auto filter = builder.createConstant(filter_tensor, qint8_per_channel_sym(0));

    auto out = builder.createOperator(Operator::Linear, output_spec,
        {input, filter, ValueHandle::null()});
    builder.createOutput(out);

    auto graph = builder.build();
    auto executor = Executor::build(graph);

    Tensor ti;
    ti.dtype = DType::QInt8Sym;
    ti.sizes = {1, 3};
    ti.storage = Storage::create_owned(3);
    auto* di = ti.data_mut<int8_t>();
    di[0] = 1; di[1] = 2; di[2] = 3;

    std::vector<Tensor> inputs;
    inputs.push_back(std::move(ti));

    auto outputs = executor.run(inputs);

    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0].sizes, (std::vector<uint64_t>{1, 2}));
    auto* d = outputs[0].data_const<int8_t>();
    EXPECT_EQ(d[0], 1);
    EXPECT_EQ(d[1], 2);
}

TEST(TestE2E, dequantize_quint8) {
    auto builder = GraphBuilder();

    auto input_spec = TensorSpec {
        .dtype = DType::QUInt8Asym,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(4) },
        .quant_params = quint8_per_tensor_asym(0.5f, 1),
    };
    auto output_spec = TensorSpec {
        .dtype = DType::Float32,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(4) },
    };

    auto input = builder.createInput(input_spec);
    auto out = builder.createOperator(Operator::Dequantize, output_spec, {input});
    builder.createOutput(out);

    auto graph = builder.build();
    auto executor = Executor::build(graph);

    Tensor ti;
    ti.dtype = DType::QUInt8Asym;
    ti.sizes = {1, 4};
    ti.storage = Storage::create_owned(4);
    auto* di = ti.data_mut<uint8_t>();
    di[0] = 0; di[1] = 1; di[2] = 2; di[3] = 3;

    std::vector<Tensor> inputs;
    inputs.push_back(std::move(ti));

    auto outputs = executor.run(inputs);

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

    auto input_spec = TensorSpec {
        .dtype = DType::Float32,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(4) },
    };
    auto output_spec = TensorSpec {
        .dtype = DType::QUInt8Asym,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(4) },
        .quant_params = quint8_per_tensor_asym(0.5f, 1),
    };

    auto input = builder.createInput(input_spec);
    auto out = builder.createOperator(Operator::Quantize, output_spec, {input});
    builder.createOutput(out);

    auto graph = builder.build();
    auto executor = Executor::build(graph);

    Tensor ti;
    ti.dtype = DType::Float32;
    ti.sizes = {1, 4};
    ti.storage = Storage::create_owned(4 * sizeof(float));
    auto* di = ti.data_mut<float>();
    di[0] = -0.5f; di[1] = 0.0f; di[2] = 0.5f; di[3] = 1.0f;

    std::vector<Tensor> inputs;
    inputs.push_back(std::move(ti));

    auto outputs = executor.run(inputs);

    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0].sizes, (std::vector<uint64_t>{1, 4}));
    auto* d = outputs[0].data_const<uint8_t>();
    EXPECT_EQ(d[0], 0);
    EXPECT_EQ(d[1], 1);
    EXPECT_EQ(d[2], 2);
    EXPECT_EQ(d[3], 3);
}

// --- Layer norm ---

TEST(TestE2E, layer_norm_no_weight_bias) {
    auto builder = GraphBuilder();

    auto spec = TensorSpec {
        .dtype = DType::Float32,
        .sizes = { DimSizeSpec::constant(2), DimSizeSpec::constant(4) }
    };

    auto input = builder.createInput(spec);
    auto out = builder.createOperator(
        Operator::LayerNorm, spec,
        {input, ValueHandle::null(), ValueHandle::null()},
        {int64_t{1}, 1e-5});
    builder.createOutput(out);

    auto graph = builder.build();
    auto executor = Executor::build(graph);

    Tensor ti;
    ti.dtype = DType::Float32;
    ti.sizes = {2, 4};
    ti.storage = Storage::create_owned(8 * sizeof(float));
    float* di = ti.data_mut<float>();
    di[0] = 1; di[1] = 2; di[2] = 3; di[3] = 4;
    di[4] = 2; di[5] = 4; di[6] = 6; di[7] = 8;

    std::vector<Tensor> inputs;
    inputs.push_back(std::move(ti));

    auto outputs = executor.run(inputs);

    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0].sizes, (std::vector<uint64_t>{2, 4}));
    auto* d = outputs[0].data_const<float>();

    // Row 0: {1,2,3,4}, mean=2.5, var=1.25
    float inv_std0 = 1.0f / std::sqrt(1.25f + 1e-5f);
    EXPECT_NEAR(d[0], (1 - 2.5f) * inv_std0, 1e-5f);
    EXPECT_NEAR(d[1], (2 - 2.5f) * inv_std0, 1e-5f);
    EXPECT_NEAR(d[2], (3 - 2.5f) * inv_std0, 1e-5f);
    EXPECT_NEAR(d[3], (4 - 2.5f) * inv_std0, 1e-5f);

    // Row 1: {2,4,6,8}, mean=5, var=5
    float inv_std1 = 1.0f / std::sqrt(5.0f + 1e-5f);
    EXPECT_NEAR(d[4], (2 - 5.0f) * inv_std1, 1e-5f);
    EXPECT_NEAR(d[5], (4 - 5.0f) * inv_std1, 1e-5f);
    EXPECT_NEAR(d[6], (6 - 5.0f) * inv_std1, 1e-5f);
    EXPECT_NEAR(d[7], (8 - 5.0f) * inv_std1, 1e-5f);
}

TEST(TestE2E, layer_norm_with_affine) {
    auto builder = GraphBuilder();

    auto spec = TensorSpec {
        .dtype = DType::Float32,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(4) }
    };
    auto weight_spec = TensorSpec {
        .dtype = DType::Float32,
        .sizes = { DimSizeSpec::constant(4) }
    };

    auto input = builder.createInput(spec);

    auto weight_tensor = std::make_shared<Tensor>();
    weight_tensor->dtype = DType::Float32;
    weight_tensor->sizes = {4};
    weight_tensor->storage = Storage::create_owned(4 * sizeof(float));
    float* ww = weight_tensor->data_mut<float>();
    ww[0] = 2.0f; ww[1] = 2.0f; ww[2] = 2.0f; ww[3] = 2.0f;
    auto weight = builder.createConstant(weight_tensor);

    auto bias_tensor = std::make_shared<Tensor>();
    bias_tensor->dtype = DType::Float32;
    bias_tensor->sizes = {4};
    bias_tensor->storage = Storage::create_owned(4 * sizeof(float));
    float* bw = bias_tensor->data_mut<float>();
    bw[0] = 1.0f; bw[1] = 1.0f; bw[2] = 1.0f; bw[3] = 1.0f;
    auto bias = builder.createConstant(bias_tensor);

    auto out = builder.createOperator(
        Operator::LayerNorm, spec,
        {input, weight, bias},
        {int64_t{1}, 1e-5});
    builder.createOutput(out);

    auto graph = builder.build();
    auto executor = Executor::build(graph);

    Tensor ti;
    ti.dtype = DType::Float32;
    ti.sizes = {1, 4};
    ti.storage = Storage::create_owned(4 * sizeof(float));
    float* di = ti.data_mut<float>();
    di[0] = 1; di[1] = 2; di[2] = 3; di[3] = 4;

    std::vector<Tensor> inputs;
    inputs.push_back(std::move(ti));

    auto outputs = executor.run(inputs);

    ASSERT_EQ(outputs.size(), 1);
    auto* d = outputs[0].data_const<float>();

    // mean=2.5, var=1.25, inv_std = 1/sqrt(1.25+eps)
    // result = normalized * weight + bias = normalized * 2 + 1
    float inv_std = 1.0f / std::sqrt(1.25f + 1e-5f);
    EXPECT_NEAR(d[0], (1 - 2.5f) * inv_std * 2.0f + 1.0f, 1e-5f);
    EXPECT_NEAR(d[1], (2 - 2.5f) * inv_std * 2.0f + 1.0f, 1e-5f);
    EXPECT_NEAR(d[2], (3 - 2.5f) * inv_std * 2.0f + 1.0f, 1e-5f);
    EXPECT_NEAR(d[3], (4 - 2.5f) * inv_std * 2.0f + 1.0f, 1e-5f);
}
