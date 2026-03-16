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
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <numeric>

#ifdef EXECUTORCH_ENABLE_EVENT_TRACER
#include <executorch/devtools/etdump/etdump_flatcc.h>
#endif

#define THROW_JS_ERROR(errorType, message, ...)                           \
  ({                                                                      \
    char msg_buf[256];                                                    \
    int len = snprintf(msg_buf, sizeof(msg_buf), message, ##__VA_ARGS__); \
    if (len < sizeof(msg_buf)) {                                          \
      EM_ASM(throw new errorType(UTF8ToString($0)), msg_buf);             \
    } else {                                                              \
      std::string msg;                                                    \
      msg.resize(len);                                                    \
      snprintf(&msg[0], len + 1, message, ##__VA_ARGS__);                 \
      EM_ASM(throw new errorType(UTF8ToString($0)), msg.c_str());         \
    }                                                                     \
    __builtin_unreachable();                                              \
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
using ::executorch::runtime::EventTracer;
using ::executorch::runtime::Result;
using ::executorch::runtime::Tag;
using ::executorch::runtime::TensorInfo;

#ifdef EXECUTORCH_ENABLE_EVENT_TRACER
using executorch::etdump::ETDumpGen;
#endif

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

constexpr size_t MAX_ELEMENTS = 8 * 1024 * 1024;

template <typename T>
std::vector<T> convertJSGeneratorToNumberVector(val generator) {
  std::vector<T> data;
  while (true) {
    val next = generator.call<val>("next");
    if (next["done"].as<bool>()) {
      break;
    }
    data.push_back(next["value"].as<T>());
    if (data.size() >= MAX_ELEMENTS) {
      THROW_JS_ERROR(
          RangeError,
          "Generator exceeded maximum element count of %zu",
          MAX_ELEMENTS);
    }
  }
  return data;
}

// make_tensor_ptr() assertions will abort the program if they fail.
// These checks will throw a JS error instead.
void assert_dim_order_and_strides_valid(
    const std::vector<executorch::aten::SizesType>& sizes,
    std::vector<executorch::aten::DimOrderType>& dim_order,
    std::vector<executorch::aten::StridesType>& strides) {
  THROW_IF_FALSE(
      dim_order.size() == 0 || dim_order.size() == sizes.size(),
      "dim_order size must match sizes or be empty.");
  THROW_IF_FALSE(
      strides.size() == 0 || strides.size() == sizes.size(),
      "strides size must match sizes or be empty.");

  if (dim_order.empty()) {
    dim_order.resize(sizes.size());
    std::iota(dim_order.begin(), dim_order.end(), 0);
    if (!strides.empty()) {
      std::sort(dim_order.begin(), dim_order.end(), [&](size_t a, size_t b) {
        return strides[a] > strides[b];
      });
    }
  }
  std::vector<executorch::aten::StridesType> computed_strides(sizes.size());

  auto error = runtime::dim_order_to_stride(
      sizes.data(), dim_order.data(), sizes.size(), computed_strides.data());
  THROW_IF_ERROR(error, "Failed to compute strides.");

  if (!strides.empty()) {
    for (size_t i = 0; i < sizes.size(); i++) {
      THROW_IF_FALSE(
          strides[i] == computed_strides[i] || sizes[i] == 1,
          "invalid strides for dim %zu: %" ET_PRI_SIZES_AND_STRIDES
          "!= %" ET_PRI_SIZES_AND_STRIDES
          " while its size is %" ET_PRI_SIZES_AND_STRIDES " != 1",
          i,
          strides[i],
          computed_strides[i],
          sizes[i]);
    }
  }

  strides = std::move(computed_strides);
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

  static std::unique_ptr<JsTensor> full(val_array<int> sizes, val fill_value) {
    // If type is unspecified, infer the type from the fill value.
    // Assume it is a Bigint if not Number.
    return full(
        sizes,
        fill_value,
        fill_value.isNumber() ? ScalarType::Float : ScalarType::Long);
  }

  static std::unique_ptr<JsTensor>
  full(val_array<int> sizes, val fill_value, ScalarType type) {
    auto sizes_vec =
        convertJSArrayToNumberVector<executorch::aten::SizesType>(sizes);
    switch (type) {
#define JS_CASE_FULL_VECTOR_TYPE(T, NAME)                                 \
  case ScalarType::NAME: {                                                \
    TensorPtr tensor =                                                    \
        extension::full(sizes_vec, fill_value.as<T>(), ScalarType::NAME); \
    return std::make_unique<JsTensor>(std::move(tensor));                 \
  }
      JS_FORALL_SUPPORTED_TENSOR_TYPES(JS_CASE_FULL_VECTOR_TYPE)
      default:
        THROW_JS_ERROR(TypeError, "Unsupported Tensor type: %d", type);
    }
  }

  static std::unique_ptr<JsTensor> zeros(val_array<int> sizes) {
    return zeros(sizes, ScalarType::Float);
  }

  static std::unique_ptr<JsTensor> zeros(
      val_array<int> sizes,
      ScalarType type) {
    auto sizes_vec =
        convertJSArrayToNumberVector<executorch::aten::SizesType>(sizes);
    TensorPtr tensor = extension::zeros(sizes_vec, type);
    return std::make_unique<JsTensor>(std::move(tensor));
  }

  static std::unique_ptr<JsTensor> ones(val_array<int> sizes) {
    return ones(sizes, ScalarType::Float);
  }

  static std::unique_ptr<JsTensor> ones(val_array<int> sizes, ScalarType type) {
    auto sizes_vec =
        convertJSArrayToNumberVector<executorch::aten::SizesType>(sizes);
    TensorPtr tensor = extension::ones(sizes_vec, type);
    return std::make_unique<JsTensor>(std::move(tensor));
  }

  static std::unique_ptr<JsTensor> from_array(
      val_array<int> sizes,
      val_array<val> data) {
    // If type is unspecified, infer the type from the data.
    // Assume it is a Bigint if not Number.
    return from_array(
        sizes,
        data,
        data["length"].as<size_t>() == 0 || data[0].isNumber()
            ? ScalarType::Float
            : ScalarType::Long);
  }

  static std::unique_ptr<JsTensor>
  from_array(val_array<int> sizes, val_array<val> data, ScalarType type) {
    return from_array(sizes, data, type, val::array());
  }

  static std::unique_ptr<JsTensor> from_array(
      val_array<int> sizes,
      val_array<val> data,
      ScalarType type,
      val_array<int> dim_order) {
    return from_array(sizes, data, type, dim_order, val::array());
  }

  static std::unique_ptr<JsTensor> from_array(
      val_array<int> sizes,
      val_array<val> data,
      ScalarType type,
      val_array<int> dim_order,
      val_array<int> strides) {
    auto sizes_vec =
        convertJSArrayToNumberVector<executorch::aten::SizesType>(sizes);

    auto dim_order_vec =
        convertJSArrayToNumberVector<executorch::aten::DimOrderType>(dim_order);
    auto strides_vec =
        convertJSArrayToNumberVector<executorch::aten::StridesType>(strides);

    assert_dim_order_and_strides_valid(sizes_vec, dim_order_vec, strides_vec);
    switch (type) {
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
        THROW_JS_ERROR(TypeError, "Unsupported Tensor type: %d", type);
    }
  }

  static std::unique_ptr<JsTensor> from_iter(
      val_array<int> sizes,
      val_array<val> data) {
    return from_iter(sizes, data, ScalarType::Float);
  }

  static std::unique_ptr<JsTensor>
  from_iter(val_array<int> sizes, val_array<val> data, ScalarType type) {
    return from_iter(sizes, data, type, val::array());
  }

  static std::unique_ptr<JsTensor> from_iter(
      val_array<int> sizes,
      val_array<val> data,
      ScalarType type,
      val_array<int> dim_order) {
    return from_iter(sizes, data, type, dim_order, val::array());
  }

  static std::unique_ptr<JsTensor> from_iter(
      val_array<int> sizes,
      val_array<val> data,
      ScalarType type,
      val_array<int> dim_order,
      val_array<int> strides) {
    auto sizes_vec =
        convertJSArrayToNumberVector<executorch::aten::SizesType>(sizes);
    auto dim_order_vec =
        convertJSArrayToNumberVector<executorch::aten::DimOrderType>(dim_order);
    auto strides_vec =
        convertJSArrayToNumberVector<executorch::aten::StridesType>(strides);

    assert_dim_order_and_strides_valid(sizes_vec, dim_order_vec, strides_vec);

    switch (type) {
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
        THROW_JS_ERROR(TypeError, "Unsupported Tensor type: %d", type);
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
      THROW_IF_FALSE(
          v.instanceof
          (val::module_property("Tensor")),
          "Received non-tensor object: %s",
          val::global("JSON").call<std::string>("stringify", v).c_str());
      return EValue(v.as<JsTensor&>().get_tensor());
    }
    THROW_JS_ERROR(
        TypeError, "Unsupported JavaScript type: %s", type_str.c_str());
  }
}

// Converts EValue to JS value.
val to_val(EValue&& v) {
  if (v.isNone()) {
    return val::null();
  } else if (v.isInt()) {
    return val(v.toInt());
  } else if (v.isDouble()) {
    return val(v.toDouble());
  } else if (v.isBool()) {
    return val(v.toBool());
  } else if (v.isTensor()) {
    Tensor tensor = std::move(v).toTensor();
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
      Tag tag = meta.input_tag(i).get();
      js_array_push(new_meta.input_tags, tag);
      if (tag == Tag::Tensor) {
        js_array_push(
            new_meta.input_tensor_meta,
            JsTensorInfo::from_tensor_info(meta.input_tensor_meta(i).get()));
      } else {
        js_array_push(new_meta.input_tensor_meta, val::undefined());
      }
    }
    for (int i = 0; i < meta.num_outputs(); i++) {
      Tag tag = meta.output_tag(i).get();
      js_array_push(new_meta.output_tags, tag);
      if (tag == Tag::Tensor) {
        js_array_push(
            new_meta.output_tensor_meta,
            JsTensorInfo::from_tensor_info(meta.output_tensor_meta(i).get()));
      } else {
        js_array_push(new_meta.output_tensor_meta, val::undefined());
      }
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
 * EXPERIMENTAL: Wrapper around ETDumpResult for JavaScript.
 */
#ifdef EXECUTORCH_ENABLE_EVENT_TRACER
class ET_EXPERIMENTAL JsETDumpResult final {
 public:
  JsETDumpResult() = delete;
  JsETDumpResult(const JsETDumpResult&) = delete;
  JsETDumpResult& operator=(const JsETDumpResult&) = delete;
  JsETDumpResult(JsETDumpResult&&) = default;
  JsETDumpResult& operator=(JsETDumpResult&&) = default;

  explicit JsETDumpResult(uint8_t* buffer, size_t size)
      : buffer_(buffer), size_(size) {}

  ~JsETDumpResult() {
    free(buffer_);
  }

  val get_buffer() const {
    return val(typed_memory_view(size_, buffer_));
  }

 private:
  uint8_t* buffer_;
  size_t size_;
};
#endif

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

#ifdef EXECUTORCH_ENABLE_EVENT_TRACER
    std::unique_ptr<EventTracer> etdump_gen = std::make_unique<ETDumpGen>();
#else
    std::unique_ptr<EventTracer> etdump_gen = nullptr;
#endif
    return std::make_unique<JsModule>(
        std::move(buffer),
        std::make_unique<Module>(
            std::move(loader), nullptr, nullptr, std::move(etdump_gen)));
  }

  static std::unique_ptr<JsModule> load(val data) {
    if (data.isNull() || data.isUndefined()) {
      THROW_JS_ERROR(TypeError, "Data cannot be null or undefined");
    }
    if (data.isString()) {
#ifdef EXECUTORCH_ENABLE_EVENT_TRACER
      std::unique_ptr<EventTracer> etdump_gen = std::make_unique<ETDumpGen>();
#else
      std::unique_ptr<EventTracer> etdump_gen = nullptr;
#endif
      return std::make_unique<JsModule>(std::make_unique<Module>(
          data.as<std::string>(),
          Module::LoadMode::File,
          std::move(etdump_gen)));
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

#ifdef EXECUTORCH_ENABLE_EVENT_TRACER
  std::unique_ptr<JsETDumpResult> etdump() {
    ETDumpGen* etdump_gen = dynamic_cast<ETDumpGen*>(module_->event_tracer());
    if (etdump_gen == nullptr) {
      return nullptr;
    }
    auto etdump_data = etdump_gen->get_etdump_data();
    return std::make_unique<JsETDumpResult>(
        static_cast<uint8_t*>(etdump_data.buf), etdump_data.size);
  }
#endif

  val_array<val> execute(const std::string& method, val js_inputs) {
    std::vector<EValue> inputs;
    if (js_inputs.isArray()) {
      size_t len = js_inputs["length"].as<size_t>();
      inputs.reserve(len);
      for (int i = 0; i < len; i++) {
        inputs.push_back(to_evalue(js_inputs[i]));
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

#ifdef EXECUTORCH_ENABLE_EVENT_TRACER
  class_<JsETDumpResult>("ETDumpResult")
      .property("buffer", &JsETDumpResult::get_buffer);
#endif

  class_<JsModule>("Module")
      .class_function("load", &JsModule::load)
      .function("getMethods", &JsModule::get_methods)
      .function("loadMethod", &JsModule::load_method)
      .function("getMethodMeta", &JsModule::get_method_meta)
#ifdef EXECUTORCH_ENABLE_EVENT_TRACER
      .function("etdump", &JsModule::etdump)
#endif
      .function("execute", &JsModule::execute)
      .function("forward", &JsModule::forward);
  class_<JsTensor>("Tensor")
      .class_function(
          "zeros",
          select_overload<std::unique_ptr<JsTensor>(val)>(&JsTensor::zeros))
      .class_function(
          "zeros",
          select_overload<std::unique_ptr<JsTensor>(val, ScalarType)>(
              &JsTensor::zeros))
      .class_function(
          "ones",
          select_overload<std::unique_ptr<JsTensor>(val)>(&JsTensor::ones))
      .class_function(
          "ones",
          select_overload<std::unique_ptr<JsTensor>(val, ScalarType)>(
              &JsTensor::ones))
      .class_function(
          "full",
          select_overload<std::unique_ptr<JsTensor>(val, val)>(&JsTensor::full))
      .class_function(
          "full",
          select_overload<std::unique_ptr<JsTensor>(val, val, ScalarType)>(
              &JsTensor::full))
      .class_function(
          "fromArray",
          select_overload<std::unique_ptr<JsTensor>(val, val)>(
              &JsTensor::from_array))
      .class_function(
          "fromArray",
          select_overload<std::unique_ptr<JsTensor>(val, val, ScalarType)>(
              &JsTensor::from_array))
      .class_function(
          "fromArray",
          select_overload<std::unique_ptr<JsTensor>(val, val, ScalarType, val)>(
              &JsTensor::from_array))
      .class_function(
          "fromArray",
          select_overload<std::unique_ptr<JsTensor>(
              val, val, ScalarType, val, val)>(&JsTensor::from_array))
      .class_function(
          "fromIter",
          select_overload<std::unique_ptr<JsTensor>(val, val)>(
              &JsTensor::from_iter))
      .class_function(
          "fromIter",
          select_overload<std::unique_ptr<JsTensor>(val, val, ScalarType)>(
              &JsTensor::from_iter))
      .class_function(
          "fromIter",
          select_overload<std::unique_ptr<JsTensor>(val, val, ScalarType, val)>(
              &JsTensor::from_iter))
      .class_function(
          "fromIter",
          select_overload<std::unique_ptr<JsTensor>(
              val, val, ScalarType, val, val)>(&JsTensor::from_iter))
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
  val::module_property("ScalarType")[#NAME].set("name", #NAME);
  JS_FORALL_SUPPORTED_TENSOR_TYPES(JS_ASSIGN_SCALAR_TYPE_NAME)
#define JS_ASSIGN_TAG_NAME(NAME) \
  val::module_property("Tag")[#NAME].set("name", #NAME);
  EXECUTORCH_FORALL_TAGS(JS_ASSIGN_TAG_NAME)
}

} // namespace wasm
} // namespace extension
} // namespace executorch
