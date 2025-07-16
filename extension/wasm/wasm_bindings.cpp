
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

/// Throws a JavaScript Error with the provided message if `error` is not `Ok`.
#define THROW_IF_FALSE(cond, message, ...)           \
  ({                                                 \
    if ET_UNLIKELY (!(cond)) {                       \
      THROW_JS_ERROR(Error, message, ##__VA_ARGS__); \
    }                                                \
  })

using namespace emscripten;
using executorch::aten::Tensor;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::Result;
using ::executorch::runtime::TensorInfo;

namespace executorch {
namespace extension {
namespace wasm {

namespace {

#define JS_FORALL_SUPPORTED_TENSOR_TYPES(_) \
  _(float, Float)                           \
  _(int, Int)

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

class JsBaseTensor {
 public:
  virtual ~JsBaseTensor() = default;

  virtual Tensor get_tensor() = 0;
  virtual int get_scalar_type() const = 0;
  val get_data() {
    switch (get_scalar_type()) {
#define JS_CASE_TENSOR_TO_VAL_TYPE(T, NAME)      \
  case static_cast<int>(aten::ScalarType::NAME): \
    return val::array(                           \
        get_tensor().data_ptr<T>(),              \
        get_tensor().data_ptr<T>() + get_tensor().numel());
      JS_FORALL_SUPPORTED_TENSOR_TYPES(JS_CASE_TENSOR_TO_VAL_TYPE)
      default:
        THROW_JS_ERROR(
            TypeError, "Unsupported Tensor type: %d", get_scalar_type());
    }
  }
  val get_sizes() {
    return val::array(get_tensor().sizes().begin(), get_tensor().sizes().end());
  }
};

template <typename T, aten::ScalarType S>
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

  static std::unique_ptr<JsBaseTensor> full(val sizes, val fill_value) {
    auto sizes_vec =
        convertJSArrayToNumberVector<torch::executor::Tensor::SizesType>(sizes);
    return fill_internal(std::move(sizes_vec), fill_value.as<T>());
  }

  static std::unique_ptr<JsBaseTensor> zeros(val sizes) {
    auto sizes_vec =
        convertJSArrayToNumberVector<torch::executor::Tensor::SizesType>(sizes);
    return fill_internal(std::move(sizes_vec), 0);
  }

  static std::unique_ptr<JsBaseTensor> ones(val sizes) {
    auto sizes_vec =
        convertJSArrayToNumberVector<torch::executor::Tensor::SizesType>(sizes);
    return fill_internal(std::move(sizes_vec), 1);
  }

  static std::unique_ptr<JsBaseTensor> from_array(val data, val sizes) {
    return from_array(data, sizes, val::null());
  }

  static std::unique_ptr<JsBaseTensor>
  from_array(val data, val sizes, val strides) {
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
  Tensor get_tensor() override {
    return *tensor_;
  }
  int get_scalar_type() const override {
    return static_cast<int>(S);
  }

 private:
  std::vector<T> data_;
  TensorPtr tensor_;
};

#define JS_DECLARE_TENSOR_TYPE(T, NAME) \
  using Js##NAME##Tensor = JsTensor<T, aten::ScalarType::NAME>;

JS_FORALL_SUPPORTED_TENSOR_TYPES(JS_DECLARE_TENSOR_TYPE)

class JsOutputTensor final : public JsBaseTensor {
 public:
  JsOutputTensor() = delete;
  JsOutputTensor(const JsOutputTensor&) = delete;
  JsOutputTensor& operator=(const JsOutputTensor&) = delete;
  JsOutputTensor(JsOutputTensor&&) = default;
  JsOutputTensor& operator=(JsOutputTensor&&) = default;

  explicit JsOutputTensor(std::unique_ptr<Tensor> tensor)
      : tensor_(std::move(tensor)) {}

  Tensor get_tensor() override {
    return *tensor_;
  }

  int get_scalar_type() const override {
    return static_cast<int>(tensor_->scalar_type());
  }

 private:
  std::unique_ptr<Tensor> tensor_;
};

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
    if (type_str == "object") {
      // If it is an object, assume it is a tensor.
      return EValue(v.as<JsBaseTensor&>().get_tensor());
    }
    THROW_JS_ERROR(
        TypeError, "Unsupported JavaScript type: %s", type_str.c_str());
  }
}

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
    std::unique_ptr<JsBaseTensor> wrapper = std::make_unique<JsOutputTensor>(
        std::make_unique<Tensor>(std::move(tensor)));
    return val(std::move(wrapper));
  } else {
    char tag_buf[32];
    runtime::tag_to_string(v.tag, tag_buf, 32);
    THROW_JS_ERROR(TypeError, "Unsupported EValue type: %s", tag_buf);
  }
}

class JsTensorInfo final {
 public:
  JsTensorInfo() = delete;
  JsTensorInfo(const JsTensorInfo&) = delete;
  JsTensorInfo& operator=(const JsTensorInfo&) = delete;
  JsTensorInfo(JsTensorInfo&&) = default;
  JsTensorInfo& operator=(JsTensorInfo&&) = default;

  explicit JsTensorInfo(std::unique_ptr<TensorInfo> tensor_info)
      : tensor_info_(std::move(tensor_info)) {}

  val sizes() const {
    return val::array(
        tensor_info_->sizes().begin(), tensor_info_->sizes().end());
  }

 private:
  std::unique_ptr<TensorInfo> tensor_info_;
};

class JsMethodMeta final {
 public:
  JsMethodMeta() = delete;
  JsMethodMeta(const JsMethodMeta&) = delete;
  JsMethodMeta& operator=(const JsMethodMeta&) = delete;
  JsMethodMeta(JsMethodMeta&&) = default;
  JsMethodMeta& operator=(JsMethodMeta&&) = default;

  explicit JsMethodMeta(std::unique_ptr<MethodMeta> meta)
      : meta_(std::move(meta)) {}

  val name() const {
    return val::u8string(meta_->name());
  }

  size_t num_inputs() const {
    return meta_->num_inputs();
  }

  std::unique_ptr<JsTensorInfo> input_tensor_meta(size_t index) {
    auto res = meta_->input_tensor_meta(index);
    THROW_IF_ERROR(
        res.error(),
        "Failed to get input tensor info for index %zu, error: 0x%" PRIx32,
        index,
        static_cast<uint32_t>(res.error()));
    return std::make_unique<JsTensorInfo>(
        std::make_unique<TensorInfo>(std::move(res.get())));
  }

 private:
  std::unique_ptr<MethodMeta> meta_;
};

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

  std::unique_ptr<JsMethodMeta> get_method_meta(
      const std::string& method_name) {
    auto res = module_->method_meta(method_name);
    THROW_IF_ERROR(
        res.error(),
        "Failed to get method meta for %s, error: 0x%" PRIx32,
        method_name.c_str(),
        static_cast<uint32_t>(res.error()));
    return std::make_unique<JsMethodMeta>(
        std::make_unique<MethodMeta>(std::move(res.get())));
  }

  val execute(const std::string& method, val js_inputs) {
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
      js_outputs.call<void>("push", to_val(std::move(output)));
    }
    return js_outputs;
  }

  val forward(val inputs) {
    return execute("forward", inputs);
  }

 private:
  std::unique_ptr<Module> module_;
};

} // namespace

EMSCRIPTEN_BINDINGS(WasmBindings) {
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
  class_<JsMethodMeta>("MethodMeta")
      .property("name", &JsMethodMeta::name)
      .property("numInputs", &JsMethodMeta::num_inputs)
      .function("inputTensorMeta", &JsMethodMeta::input_tensor_meta);
  class_<JsTensorInfo>("TensorInfo").property("sizes", &JsTensorInfo::sizes);
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
