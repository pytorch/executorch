#include <vector>

#include <standalone/slim/core/SlimTensor.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

inline std::vector<AtenTensorHandle> unsafe_alloc_new_handles_from_tensors(
    const std::vector<standalone::slim::SlimTensor> &tensors) {
  std::vector<AtenTensorHandle> result;
  result.reserve(tensors.size());
  for (auto tensor : tensors) {
    auto allocated = new standalone::slim::SlimTensor(std::move(tensor));
    result.push_back(allocated);
  }
  return result;
}

inline std::vector<standalone::slim::SlimTensor>
alloc_tensors_by_stealing_from_handles(AtenTensorHandle *handles,
                                       size_t length) {
  // Find duplicates by recording the last known index for each handle.
  std::unordered_map<AtenTensorHandle, size_t> lastKnownIdx;
  for (size_t i = 0; i < length; i++) {
    lastKnownIdx[handles[i]] = i;
  }

  std::vector<standalone::slim::SlimTensor> result;
  result.reserve(length);
  for (size_t i = 0; i < length; i++) {
    if (handles[i] == nullptr) {
      // result.emplace_back();
      continue;
    }

    standalone::slim::SlimTensor tensor = *handles[i];
    result.emplace_back(std::move(tensor));
    if (lastKnownIdx[handles[i]] == i) {
      aoti_torch_delete_tensor_object(handles[i]);
    }
    handles[i] = nullptr;
  }

  return result;
}
