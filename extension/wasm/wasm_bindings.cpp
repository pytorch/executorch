/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <emscripten.h>
#include <emscripten/bind.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#define THROW_JS_ERROR(errorType, message, ...)                 \
  ({                                                            \
    char msg_buf[128];                                          \
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

// Base class for all JS Tensor types. Subclasses are not exposed to JS.
class JsBaseTensor {
 public:
  virtual ~JsBaseTensor() = default;

  virtual const Tensor& get_tensor() = 0;
  virtual ScalarType get_scalar_type() const = 0;
  val_array<val> get_data() {
    switch (get_scalar_type()) {
#define JS_CASE_TENSOR_TO_VAL_TYPE(T, NAME) \
  case ScalarType::NAME:                    \
    return val::array(                      \
        get_tensor().data_ptr<T>(),         \
        get_tensor().data_ptr<T>() + get_tensor().numel());
      JS_FORALL_SUPPORTED_TENSOR_TYPES(JS_CASE_TENSOR_TO_VAL_TYPE)
      default:
        THROW_JS_ERROR(
            TypeError, "Unsupported Tensor type: %d", get_scalar_type());
    }
  }
  val_array<int> get_sizes() {
    return val::array(get_tensor().sizes().begin(), get_tensor().sizes().end());
  }
};

// Tensor that owns its own data. JS only has access to the static methods.
template <typename T, ScalarType S>
class JsTensor final : public JsBaseTensor {
 public:
  JsTensor(std::vector<T> data, TensorPtr tensor)
      : data_(std::move(data)), tensor_(std::move(tensor)) {}

  static std::unique_ptr<JsBaseTensor> fill_internal(
      const std::vector<torch::executor::Tensor::SizesType>&& sizes,
      T fill_value) {
    std::vector<T> data_vec(compute_expected_numel(sizes), fill_value);
    TensorPtr tensor = from_blob(data_vec.data(), sizes, S);
    return std::make_unique<JsTensor>(std::move(data_vec), std::move(tensor));
  }

  static std::unique_ptr<JsBaseTensor> full(
      val_array<int> sizes,
      val fill_value) {
    auto sizes_vec =
        convertJSArrayToNumberVector<torch::executor::Tensor::SizesType>(sizes);
    return fill_internal(std::move(sizes_vec), fill_value.as<T>());
  }

  static std::unique_ptr<JsBaseTensor> zeros(val_array<int> sizes) {
    auto sizes_vec =
        convertJSArrayToNumberVector<torch::executor::Tensor::SizesType>(sizes);
    return fill_internal(std::move(sizes_vec), 0);
  }

  static std::unique_ptr<JsBaseTensor> ones(val_array<int> sizes) {
    auto sizes_vec =
        convertJSArrayToNumberVector<torch::executor::Tensor::SizesType>(sizes);
    return fill_internal(std::move(sizes_vec), 1);
  }

  static std::unique_ptr<JsBaseTensor> from_array(
      val_array<val> data,
      val_array<int> sizes) {
    return from_array(data, sizes, val::null());
  }

  static std::unique_ptr<JsBaseTensor> from_array(
      val_array<val> data,
      val_array<int> sizes,
      val_array<int> strides) {
    auto data_vec = convertJSArrayToNumberVector<T>(data);
    auto sizes_vec =
        convertJSArrayToNumberVector<torch::executor::Tensor::SizesType>(sizes);
    assert_valid_numel(data_vec, sizes_vec);

    if (strides.isNull()) {
      TensorPtr tensor = from_blob(data_vec.data(), std::move(sizes_vec), S);
      return std::make_unique<JsTensor>(std::move(data_vec), std::move(tensor));
    }
    auto strides_vec =
        convertJSArrayToNumberVector<torch::executor::Tensor::StridesType>(
            strides);
    TensorPtr tensor = from_blob(
        data_vec.data(), std::move(sizes_vec), std::move(strides_vec), S);
    return std::make_unique<JsTensor>(std::move(data_vec), std::move(tensor));
  }
  const Tensor& get_tensor() override {
    return *tensor_;
  }
  ScalarType get_scalar_type() const override {
    return S;
  }

 private:
  std::vector<T> data_;
  TensorPtr tensor_;
};

#define JS_DECLARE_TENSOR_TYPE(T, NAME) \
  using Js##NAME##Tensor = JsTensor<T, ScalarType::NAME>;
JS_FORALL_SUPPORTED_TENSOR_TYPES(JS_DECLARE_TENSOR_TYPE)

// Tensor that does not own its own data. It is a wrapper around a C++ Tensor.
// This class is not exposed to JS.
class JsOutputTensor final : public JsBaseTensor {
 public:
  JsOutputTensor() = delete;
  JsOutputTensor(const JsOutputTensor&) = delete;
  JsOutputTensor& operator=(const JsOutputTensor&) = delete;
  JsOutputTensor(JsOutputTensor&&) = default;
  JsOutputTensor& operator=(JsOutputTensor&&) = default;

  explicit JsOutputTensor(Tensor tensor) : tensor_(tensor) {}

  const Tensor& get_tensor() override {
    return tensor_;
  }

  ScalarType get_scalar_type() const override {
    return tensor_.scalar_type();
  }

 private:
  Tensor tensor_;
};

// Converts JS value to EValue.
EValue to_evalue(val v) {
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
      return EValue(v.as<JsBaseTensor&>().get_tensor());
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
    std::unique_ptr<JsBaseTensor> wrapper =
        std::make_unique<JsOutputTensor>(std::move(tensor));
    return val(std::move(wrapper));
  } else {
    char tag_buf[32];
    runtime::tag_to_string(v.tag, tag_buf, 32);
    THROW_JS_ERROR(TypeError, "Unsupported EValue type: %s", tag_buf);
  }
}

// JS object containing tensor metadata.
struct JsTensorInfo {
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

// JS object containing method metadata.
struct JsMethodMeta {
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

// Wrapper around extension/Module.
class JsModule final {
 public:
  JsModule() = delete;
  JsModule(const JsModule&) = delete;
  JsModule& operator=(const JsModule&) = delete;
  JsModule(JsModule&&) = default;
  JsModule& operator=(JsModule&&) = default;

  explicit JsModule(std::unique_ptr<Module> module)
      : module_(std::move(module)) {}

  static std::unique_ptr<JsModule> load(const std::string& path) {
    return std::make_unique<JsModule>(std::make_unique<Module>(path));
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
  class_<JsBaseTensor>("Tensor")
      .property("scalarType", &JsBaseTensor::get_scalar_type)
      .function("getData", &JsBaseTensor::get_data)
      .function("getSizes", &JsBaseTensor::get_sizes);
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

#define JS_DECLARE_TENSOR_BINDINGS(T, NAME)                              \
  class_<Js##NAME##Tensor>(#NAME "Tensor")                               \
      .class_function("zeros", &Js##NAME##Tensor::zeros)                 \
      .class_function("ones", &Js##NAME##Tensor::ones)                   \
      .class_function("full", &Js##NAME##Tensor::full)                   \
      .class_function(                                                   \
          "fromArray",                                                   \
          select_overload<std::unique_ptr<JsBaseTensor>(val, val, val)>( \
              &Js##NAME##Tensor::from_array))                            \
      .class_function(                                                   \
          "fromArray",                                                   \
          select_overload<std::unique_ptr<JsBaseTensor>(val, val)>(      \
              &Js##NAME##Tensor::from_array));
  JS_FORALL_SUPPORTED_TENSOR_TYPES(JS_DECLARE_TENSOR_BINDINGS)
}

} // namespace wasm
} // namespace extension
} // namespace executorch
