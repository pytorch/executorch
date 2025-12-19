#pragma once

// Copied from c10/core/DeviceType.h with some modifications:
// * enum values are kept the same as c10 and guarded by device_type_test
// * Make the implementaion header-only
// * Simplify some implementation
// * Disable PrivateUse1 name registration

#include <algorithm>
#include <cstdint>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <executorch/backends/aoti/slim/c10/util/Exception.h>

namespace standalone::c10 {
enum class DeviceType : int8_t {
  CPU = 0,
  CUDA = 1, // CUDA.
  MKLDNN = 2, // Reserved for explicit MKLDNN
  OPENGL = 3, // OpenGL
  OPENCL = 4, // OpenCL
  IDEEP = 5, // IDEEP.
  HIP = 6, // AMD HIP
  FPGA = 7, // FPGA
  MAIA = 8, // ONNX Runtime / Microsoft
  XLA = 9, // XLA / TPU
  Vulkan = 10, // Vulkan
  Metal = 11, // Metal
  XPU = 12, // XPU
  MPS = 13, // MPS
  Meta = 14, // Meta (tensors with no data)
  HPU = 15, // HPU / HABANA
  VE = 16, // SX-Aurora / NEC
  Lazy = 17, // Lazy Tensors
  IPU = 18, // Graphcore IPU
  MTIA = 19, // Meta training and inference devices
  PrivateUse1 = 20, // PrivateUse1 device
  // NB: If you add more devices:
  //  - Change the implementations of DeviceTypeName and isValidDeviceType
  //  - Change the number below
  COMPILE_TIME_MAX_DEVICE_TYPES = 21,
};

constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;
constexpr DeviceType kMKLDNN = DeviceType::MKLDNN;
constexpr DeviceType kOPENGL = DeviceType::OPENGL;
constexpr DeviceType kOPENCL = DeviceType::OPENCL;
constexpr DeviceType kIDEEP = DeviceType::IDEEP;
constexpr DeviceType kHIP = DeviceType::HIP;
constexpr DeviceType kFPGA = DeviceType::FPGA;
constexpr DeviceType kMAIA = DeviceType::MAIA;
constexpr DeviceType kXLA = DeviceType::XLA;
constexpr DeviceType kVulkan = DeviceType::Vulkan;
constexpr DeviceType kMetal = DeviceType::Metal;
constexpr DeviceType kXPU = DeviceType::XPU;
constexpr DeviceType kMPS = DeviceType::MPS;
constexpr DeviceType kMeta = DeviceType::Meta;
constexpr DeviceType kHPU = DeviceType::HPU;
constexpr DeviceType kVE = DeviceType::VE;
constexpr DeviceType kLazy = DeviceType::Lazy;
constexpr DeviceType kIPU = DeviceType::IPU;
constexpr DeviceType kMTIA = DeviceType::MTIA;
constexpr DeviceType kPrivateUse1 = DeviceType::PrivateUse1;

// define explicit int constant
constexpr int COMPILE_TIME_MAX_DEVICE_TYPES =
    static_cast<int>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);

static_assert(
    COMPILE_TIME_MAX_DEVICE_TYPES <= 21,
    "Hey!  You seem to be adding a lot of new DeviceTypes.  The intent was "
    "for this constant to reflect the actual number of DeviceTypes we support "
    "in PyTorch; it's important that this number is not too large as we "
    "use this to allocate stack arrays in some places in our code.  If you "
    "are indeed just adding the 20th device type, feel free to change "
    "the check to 32; but if you are adding some sort of extensible device "
    "types registration, please be aware that you are affecting code that "
    "this number is small.  Try auditing uses of this constant.");

// Doesn't support PrivateUse1 name registration in standalone
inline std::string get_privateuse1_backend(bool lower_case = true) {
  return lower_case ? "privateuse1" : "PrivateUse1";
}

inline std::string DeviceTypeName(DeviceType d, bool lower_case = false) {
  static const std::string device_names[] = {
      "CPU",  "CUDA", "MKLDNN", "OPENGL", "OPENCL", "IDEEP", "HIP",
      "FPGA", "MAIA", "XLA",    "VULKAN", "METAL",  "XPU",   "MPS",
      "META", "HPU",  "VE",     "LAZY",   "IPU",    "MTIA"};

  int idx = static_cast<int>(d);
  if (idx < 0 || idx >= COMPILE_TIME_MAX_DEVICE_TYPES) {
    STANDALONE_CHECK(false, "Unknown device: ", static_cast<int16_t>(d));
  }
  if (d == DeviceType::PrivateUse1) {
    return get_privateuse1_backend(lower_case);
  }
  std::string name = device_names[idx];
  if (lower_case) {
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);
  }
  return name;
}

// NB: Per the C++ standard (e.g.,
// https://stackoverflow.com/questions/18195312/what-happens-if-you-static-cast-invalid-value-to-enum-class)
// as long as you cast from the same underlying type, it is always valid to cast
// into an enum class (even if the value would be invalid by the enum.)  Thus,
// the caller is allowed to cast a possibly invalid int16_t to DeviceType and
// then pass it to this function.  (I considered making this function take an
// int16_t directly, but that just seemed weird.)
inline bool isValidDeviceType(DeviceType d) {
  int idx = static_cast<int>(d);
  return idx >= 0 && idx < COMPILE_TIME_MAX_DEVICE_TYPES;
}

inline std::ostream& operator<<(std::ostream& stream, DeviceType type) {
  stream << DeviceTypeName(type, /* lower case */ true);
  return stream;
}
} // namespace standalone::c10

namespace std {
template <>
struct hash<standalone::c10::DeviceType> {
  std::size_t operator()(standalone::c10::DeviceType k) const {
    return std::hash<int>()(static_cast<int>(k));
  }
};
} // namespace std
