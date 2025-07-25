/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <emscripten.h>
#include <emscripten/bind.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#define THROW_JS_ERROR(errorType, message, ...)                 \
  ({                                                            \
    char msg_buf[256];                                          \
    snprintf(msg_buf, sizeof(msg_buf), message, ##__VA_ARGS__); \
    EM_ASM(throw new errorType(UTF8ToString($0)), msg_buf);     \
    __builtin_unreachable();                                    \
  })

/// Throws a JavaScript Error with the provided message if `error` is not `Ok`.
#define THROW_IF_ERROR(error, message, ...)          \
  ({                                                 \
    if ET_UNLIKELY ((error) != Error::Ok) {          \
      THROW_JS_ERROR(Error, message, ##__VA_ARGS__); \
    }                                                \
  })

/// Throws a JavaScript Error with the provided message if `cond` is not `true`.
#define THROW_IF_FALSE(cond, message, ...)           \
  ({                                                 \
    if ET_UNLIKELY (!(cond)) {                       \
      THROW_JS_ERROR(Error, message, ##__VA_ARGS__); \
    }                                                \
  })

using namespace emscripten;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using ::executorch::extension::BufferDataLoader;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::Result;
using ::executorch::runtime::Tag;
using ::executorch::runtime::TensorInfo;

namespace executorch {
namespace extension {
namespace wasm {

namespace {

// val represents all JS values. Using val_array to specify that we specifically
// want an array.
template <typename T>
using val_array = val;

template <typename T>
inline void js_array_push(val_array<T>& array, const T& value) {
  array.call<void>("push", value);
}

#define JS_FORALL_SUPPORTED_TENSOR_TYPES(_) \
  _(float, Float)                           \
  _(int64_t, Long)

inline ssize_t compute_expected_numel(
    const std::vector<torch::executor::Tensor::SizesType>& sizes) {
  return executorch::aten::compute_numel(sizes.data(), sizes.size());
}

template <typename T>
inline void assert_valid_numel(
    const std::vector<T>& data,
    const std::vector<torch::executor::Tensor::SizesType>& sizes) {
  auto computed_numel = compute_expected_numel(sizes);
  THROW_IF_FALSE(
      data.size() >= computed_numel,
      "Required %ld elements, given %ld",
      computed_numel,
      data.size());
}

template <typename T>
std::vector<T> convertJSGeneratorToNumberVector(val generator) {
  std::vector<T> data;
  while (true) {
    val next = generator.call<val>("next");
    if (next["done"].as<bool>()) {
      break;
    }
    data.push_back(next["value"].as<T>());
  }
  return data;
}

/**
 * EXPERIMENTAL: JavaScript wrapper for ExecuTorch Tensor.
 */
class ET_EXPERIMENTAL JsTensor {
 public:
  JsTensor() = delete;
  JsTensor(const JsTensor&) = delete;
  JsTensor& operator=(const JsTensor&) = delete;
  JsTensor(JsTensor&&) = default;
  JsTensor& operator=(JsTensor&&) = default;

  explicit JsTensor(TensorPtr tensor) : tensor_(std::move(tensor)) {}
  explicit JsTensor(Tensor&& tensor)
      : tensor_(std::make_shared<Tensor>(tensor)) {}

  const Tensor& get_tensor() const {
    THROW_IF_FALSE(tensor_, "Tensor is null");
    return *tensor_;
  }

  ScalarType get_scalar_type() const {
    THROW_IF_FALSE(tensor_, "Tensor is null");
    return tensor_->scalar_type();
  }
  val get_data() const {
    switch (get_scalar_type()) {
#define JS_CASE_TENSOR_TO_VAL_TYPE(T, NAME)                        \
  case ScalarType::NAME:                                           \
    THROW_IF_FALSE(tensor_->data_ptr<T>(), "Tensor data is null"); \
    return val(typed_memory_view(tensor_->numel(), tensor_->data_ptr<T>()));
      JS_FORALL_SUPPORTED_TENSOR_TYPES(JS_CASE_TENSOR_TO_VAL_TYPE)
      default:
        THROW_JS_ERROR(
            TypeError, "Unsupported Tensor type: %d", get_scalar_type());
    }
  }
  val_array<int> get_sizes() const {
    return val::array(get_tensor().sizes().begin(), get_tensor().sizes().end());
  }

  static std::unique_ptr<JsTensor>
  full(val_array<int> sizes, val fill_value, val type = val::undefined()) {
    auto sizes_vec =
        convertJSArrayToNumberVector<executorch::aten::SizesType>(sizes);
    ScalarType scalar_type =
        type.isUndefined() ? ScalarType::Float : type.as<ScalarType>();
    switch (scalar_type) {
#define JS_CASE_FULL_VECTOR_TYPE(T, NAME)                                 \
  case ScalarType::NAME: {                                                \
    TensorPtr tensor =                                                    \
        extension::full(sizes_vec, fill_value.as<T>(), ScalarType::NAME); \
    return std::make_unique<JsTensor>(std::move(tensor));                 \
  }
      JS_FORALL_SUPPORTED_TENSOR_TYPES(JS_CASE_FULL_VECTOR_TYPE)
      default:
        THROW_JS_ERROR(TypeError, "Unsupported Tensor type: %d", scalar_type);
    }
  }

  static std::unique_ptr<JsTensor> zeros(
      val_array<int> sizes,
      val type = val::undefined()) {
    auto sizes_vec =
        convertJSArrayToNumberVector<executorch::aten::SizesType>(sizes);
    ScalarType scalar_type =
        type.isUndefined() ? ScalarType::Float : type.as<ScalarType>();
    TensorPtr tensor = extension::zeros(sizes_vec, scalar_type);
    return std::make_unique<JsTensor>(std::move(tensor));
  }

  static std::unique_ptr<JsTensor> ones(
      val_array<int> sizes,
      val type = val::undefined()) {
    auto sizes_vec =
        convertJSArrayToNumberVector<executorch::aten::SizesType>(sizes);
    ScalarType scalar_type =
        type.isUndefined() ? ScalarType::Float : type.as<ScalarType>();
    TensorPtr tensor = extension::ones(sizes_vec, scalar_type);
    return std::make_unique<JsTensor>(std::move(tensor));
  }

  static std::unique_ptr<JsTensor> from_array(
      val_array<int> sizes,
      val_array<val> data,
      val type = val::undefined(),
      val_array<int> dim_order = val::undefined(),
      val_array<int> strides = val::undefined()) {
    auto sizes_vec =
        convertJSArrayToNumberVector<executorch::aten::SizesType>(sizes);

    auto dim_order_vec = dim_order.isUndefined()
        ? std::vector<executorch::aten::DimOrderType>()
        : convertJSArrayToNumberVector<executorch::aten::DimOrderType>(
              dim_order);
    auto strides_vec = strides.isUndefined()
        ? std::vector<executorch::aten::StridesType>()
        : convertJSArrayToNumberVector<executorch::aten::StridesType>(strides);

    // If type is undefined, infer the type from the data.
    // Assume it is a Bigint if not Number.
    ScalarType scalar_type = type.isUndefined()
        ? (data["length"].as<size_t>() == 0 || data[0].isNumber()
               ? ScalarType::Float
               : ScalarType::Long)
        : type.as<ScalarType>();
    switch (scalar_type) {
#define JS_CASE_FROM_ARRAY_VECTOR_TYPE(T, NAME)            \
  case ScalarType::NAME: {                                 \
    auto data_vec = convertJSArrayToNumberVector<T>(data); \
    assert_valid_numel(data_vec, sizes_vec);               \
    TensorPtr tensor = make_tensor_ptr(                    \
        std::move(sizes_vec),                              \
        std::move(data_vec),                               \
        std::move(dim_order_vec),                          \
        std::move(strides_vec),                            \
        ScalarType::NAME);                                 \
    return std::make_unique<JsTensor>(std::move(tensor));  \
  }
      JS_FORALL_SUPPORTED_TENSOR_TYPES(JS_CASE_FROM_ARRAY_VECTOR_TYPE)
      default:
        THROW_JS_ERROR(TypeError, "Unsupported Tensor type: %d", scalar_type);
    }
  }

  static std::unique_ptr<JsTensor> from_iter(
      val_array<int> sizes,
      val_array<val> data,
      val type = val::undefined(),
      val_array<int> dim_order = val::undefined(),
      val_array<int> strides = val::undefined()) {
    auto sizes_vec =
        convertJSArrayToNumberVector<executorch::aten::SizesType>(sizes);

    auto dim_order_vec = dim_order.isUndefined()
        ? std::vector<executorch::aten::DimOrderType>()
        : convertJSArrayToNumberVector<executorch::aten::DimOrderType>(
              dim_order);
    auto strides_vec = strides.isUndefined()
        ? std::vector<executorch::aten::StridesType>()
        : convertJSArrayToNumberVector<executorch::aten::StridesType>(strides);

    // If type is undefined, infer the type from the data.
    // Assume it is a Bigint if not Number.
    ScalarType scalar_type = type.isUndefined()
        ? (data["length"].as<size_t>() == 0 || data[0].isNumber()
               ? ScalarType::Float
               : ScalarType::Long)
        : type.as<ScalarType>();
    switch (scalar_type) {
#define JS_CASE_FROM_ITER_VECTOR_TYPE(T, NAME)                 \
  case ScalarType::NAME: {                                     \
    auto data_vec = convertJSGeneratorToNumberVector<T>(data); \
    assert_valid_numel(data_vec, sizes_vec);                   \
    TensorPtr tensor = make_tensor_ptr(                        \
        std::move(sizes_vec),                                  \
        std::move(data_vec),                                   \
        std::move(dim_order_vec),                              \
        std::move(strides_vec),                                \
        ScalarType::NAME);                                     \
    return std::make_unique<JsTensor>(std::move(tensor));      \
  }
      JS_FORALL_SUPPORTED_TENSOR_TYPES(JS_CASE_FROM_ITER_VECTOR_TYPE)
      default:
        THROW_JS_ERROR(TypeError, "Unsupported Tensor type: %d", scalar_type);
    }
  }

 private:
  TensorPtr tensor_;
};

// Converts JS value to EValue.
EValue to_evalue(val v) {
  if (v.isUndefined()) {
    THROW_JS_ERROR(TypeError, "Value cannot be undefined");
  }
  if (v.isNull()) {
    return EValue();
  } else if (v.isNumber()) {
    return EValue(v.as<double>());
  } else if (v.isTrue()) {
    return EValue(true);
  } else if (v.isFalse()) {
    return EValue(false);
  } else {
    const std::string& type_str = v.typeOf().as<std::string>();
    if (type_str == "bigint") {
      return EValue(v.as<int64_t>());
    } else if (type_str == "object") {
      // If it is an object, assume it is a tensor.
      return EValue(v.as<JsTensor&>().get_tensor());
    }
    THROW_JS_ERROR(
        TypeError, "Unsupported JavaScript type: %s", type_str.c_str());
  }
}

// Converts EValue to JS value.
val to_val(EValue v) {
  if (v.isNone()) {
    return val::null();
  } else if (v.isInt()) {
    return val(v.toInt());
  } else if (v.isDouble()) {
    return val(v.toDouble());
  } else if (v.isBool()) {
    return val(v.toBool());
  } else if (v.isTensor()) {
    Tensor tensor = v.toTensor();
    std::unique_ptr<JsTensor> wrapper =
        std::make_unique<JsTensor>(std::move(tensor));
    return val(std::move(wrapper));
  } else {
    char tag_buf[32];
    runtime::tag_to_string(v.tag, tag_buf, sizeof(tag_buf));
    THROW_JS_ERROR(TypeError, "Unsupported EValue type: %s", tag_buf);
  }
}

/**
 * EXPERIMENTAL: JavaScript object containing tensor metadata.
 */
struct ET_EXPERIMENTAL JsTensorInfo {
  val_array<int32_t> sizes;
  val_array<uint8_t> dim_order;
  ScalarType scalar_type;
  bool is_memory_planned;
  size_t nbytes;
  std::string name;

  static JsTensorInfo from_tensor_info(const TensorInfo& info) {
    return {
        val::array(info.sizes().begin(), info.sizes().end()),
        val::array(info.dim_order().begin(), info.dim_order().end()),
        info.scalar_type(),
        info.is_memory_planned(),
        info.nbytes(),
        std::string(info.name())};
  }
};

/**
 * EXPERIMENTAL: JavaScript object containing method metadata.
 */
struct ET_EXPERIMENTAL JsMethodMeta {
  std::string name;
  val_array<Tag> input_tags;
  val_array<JsTensorInfo> input_tensor_meta;
  val_array<Tag> output_tags;
  val_array<JsTensorInfo> output_tensor_meta;
  val_array<JsTensorInfo> attribute_tensor_meta;
  val_array<int64_t> memory_planned_buffer_sizes;
  val_array<std::string> backends;
  ET_DEPRECATED size_t num_instructions;

  static JsMethodMeta from_method_meta(const MethodMeta& meta) {
    JsMethodMeta new_meta{
        meta.name(),
        val::array(),
        val::array(),
        val::array(),
        val::array(),
        val::array(),
        val::array(),
        val::array(),
        meta.num_instructions()};
    for (int i = 0; i < meta.num_inputs(); i++) {
      js_array_push(new_meta.input_tags, meta.input_tag(i).get());
      js_array_push(
          new_meta.input_tensor_meta,
          JsTensorInfo::from_tensor_info(meta.input_tensor_meta(i).get()));
    }
    for (int i = 0; i < meta.num_outputs(); i++) {
      js_array_push(new_meta.output_tags, meta.output_tag(i).get());
      js_array_push(
          new_meta.output_tensor_meta,
          JsTensorInfo::from_tensor_info(meta.output_tensor_meta(i).get()));
    }
    for (int i = 0; i < meta.num_attributes(); i++) {
      js_array_push(
          new_meta.attribute_tensor_meta,
          JsTensorInfo::from_tensor_info(meta.attribute_tensor_meta(i).get()));
    }
    for (int i = 0; i < meta.num_memory_planned_buffers(); i++) {
      js_array_push(
          new_meta.memory_planned_buffer_sizes,
          meta.memory_planned_buffer_size(i).get());
    }
    for (int i = 0; i < meta.num_backends(); i++) {
      js_array_push(
          new_meta.backends, val::u8string(meta.get_backend_name(i).get()));
    }
    return new_meta;
  }
};

/**
 * EXPERIMENTAL: Wrapper around extension/Module for JavaScript.
 */
class ET_EXPERIMENTAL JsModule final {
 public:
  JsModule() = delete;
  JsModule(const JsModule&) = delete;
  JsModule& operator=(const JsModule&) = delete;
  JsModule(JsModule&&) = default;
  JsModule& operator=(JsModule&&) = default;

  explicit JsModule(std::unique_ptr<Module> module)
      : buffer_(0), module_(std::move(module)) {}

  explicit JsModule(std::vector<uint8_t> buffer, std::unique_ptr<Module> module)
      : buffer_(std::move(buffer)), module_(std::move(module)) {}

  static std::unique_ptr<JsModule> load_from_uint8_array(val data) {
    size_t length = data["length"].as<size_t>();
    std::vector<uint8_t> buffer(length);
    val memory_view = val(typed_memory_view(length, buffer.data()));
    memory_view.call<void>("set", data);
    auto loader = std::make_unique<BufferDataLoader>(buffer.data(), length);
    return std::make_unique<JsModule>(
        std::move(buffer), std::make_unique<Module>(std::move(loader)));
  }

  static std::unique_ptr<JsModule> load(val data) {
    if (data.isNull() || data.isUndefined()) {
      THROW_JS_ERROR(TypeError, "Data cannot be null or undefined");
    }
    if (data.isString()) {
      return std::make_unique<JsModule>(
          std::make_unique<Module>(data.as<std::string>()));
    } else if (data.instanceof (val::global("Uint8Array"))) {
      return load_from_uint8_array(data);
    } else if (data.instanceof (val::global("ArrayBuffer"))) {
      return load_from_uint8_array(val::global("Uint8Array").new_(data));
    } else {
      THROW_JS_ERROR(
          TypeError,
          "Unsupported data type: %s",
          data.typeOf().as<std::string>().c_str());
    }
  }

  val get_methods() {
    auto res = module_->method_names();
    THROW_IF_ERROR(
        res.error(),
        "Failed to get methods, error: 0x%" PRIx32,
        static_cast<uint32_t>(res.error()));
    return val::array(res.get().begin(), res.get().end());
  }

  void load_method(const std::string& method_name) {
    Error res = module_->load_method(method_name);
    THROW_IF_ERROR(
        res,
        "Failed to load method %s, error: 0x%" PRIx32,
        method_name.c_str(),
        static_cast<uint32_t>(res));
  }

  JsMethodMeta get_method_meta(const std::string& method_name) {
    auto res = module_->method_meta(method_name);
    THROW_IF_ERROR(
        res.error(),
        "Failed to get method meta for %s, error: 0x%" PRIx32,
        method_name.c_str(),
        static_cast<uint32_t>(res.error()));
    return JsMethodMeta::from_method_meta(res.get());
  }

  val_array<val> execute(const std::string& method, val js_inputs) {
    std::vector<EValue> inputs;
    if (js_inputs.isArray()) {
      inputs.reserve(js_inputs["length"].as<size_t>());
      for (val v : js_inputs) {
        inputs.push_back(to_evalue(v));
      }
    } else {
      inputs.push_back(to_evalue(js_inputs));
    }
    auto res = module_->execute(method, inputs);
    THROW_IF_ERROR(
        res.error(),
        "Failed to execute method %s, error: 0x%" PRIx32,
        method.c_str(),
        static_cast<uint32_t>(res.error()));
    std::vector<EValue> outputs = res.get();
    val js_outputs = val::array();
    for (auto& output : outputs) {
      js_array_push(js_outputs, to_val(std::move(output)));
    }
    return js_outputs;
  }

  val_array<val> forward(val inputs) {
    return execute("forward", inputs);
  }

 private:
  // If loaded from a buffer, keeps it alive for the lifetime of the module.
  std::vector<uint8_t> buffer_;
  std::unique_ptr<Module> module_;
};

} // namespace

EMSCRIPTEN_BINDINGS(WasmBindings) {
  enum_<ScalarType>("ScalarType")
#define JS_DECLARE_SCALAR_TYPE(T, NAME) .value(#NAME, ScalarType::NAME)
      JS_FORALL_SUPPORTED_TENSOR_TYPES(JS_DECLARE_SCALAR_TYPE);
  enum_<Tag>("Tag")
#define JS_DECLARE_TAG(NAME) .value(#NAME, Tag::NAME)
      EXECUTORCH_FORALL_TAGS(JS_DECLARE_TAG);

  class_<JsModule>("Module")
      .class_function("load", &JsModule::load)
      .function("getMethods", &JsModule::get_methods)
      .function("loadMethod", &JsModule::load_method)
      .function("getMethodMeta", &JsModule::get_method_meta)
      .function("execute", &JsModule::execute)
      .function("forward", &JsModule::forward);
  class_<JsTensor>("Tensor")
      .class_function("zeros", &JsTensor::zeros)
      .class_function("ones", &JsTensor::ones)
      .class_function("full", &JsTensor::full)
      .class_function("fromArray", &JsTensor::from_array)
      .class_function("fromIter", &JsTensor::from_iter)
      .property("scalarType", &JsTensor::get_scalar_type)
      .property("data", &JsTensor::get_data)
      .property("sizes", &JsTensor::get_sizes);
  value_object<JsTensorInfo>("TensorInfo")
      .field("sizes", &JsTensorInfo::sizes)
      .field("dimOrder", &JsTensorInfo::dim_order)
      .field("scalarType", &JsTensorInfo::scalar_type)
      .field("isMemoryPlanned", &JsTensorInfo::is_memory_planned)
      .field("nbytes", &JsTensorInfo::nbytes)
      .field("name", &JsTensorInfo::name);
  value_object<JsMethodMeta>("MethodMeta")
      .field("name", &JsMethodMeta::name)
      .field("inputTags", &JsMethodMeta::input_tags)
      .field("inputTensorMeta", &JsMethodMeta::input_tensor_meta)
      .field("outputTags", &JsMethodMeta::output_tags)
      .field("outputTensorMeta", &JsMethodMeta::output_tensor_meta)
      .field("attributeTensorMeta", &JsMethodMeta::attribute_tensor_meta)
      .field(
          "memoryPlannedBufferSizes",
          &JsMethodMeta::memory_planned_buffer_sizes)
      .field("backends", &JsMethodMeta::backends)
      .field("numInstructions", &JsMethodMeta::num_instructions);

// For some reason Embind doesn't make it easy to get the names of enums.
// Additionally, different enums of the same type are considered to be equal.
// Assigning the name field fixes both of these issues.
#define JS_ASSIGN_SCALAR_TYPE_NAME(T, NAME) \
  EM_ASM(Module.ScalarType.NAME.name = #NAME);
  JS_FORALL_SUPPORTED_TENSOR_TYPES(JS_ASSIGN_SCALAR_TYPE_NAME)
#define JS_ASSIGN_TAG_NAME(NAME) EM_ASM(Module.Tag.NAME.name = #NAME);
  EXECUTORCH_FORALL_TAGS(JS_ASSIGN_TAG_NAME)
}

} // namespace wasm
} // namespace extension
} // namespace executorch
