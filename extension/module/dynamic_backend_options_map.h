
#pragma once

#include <vector>
#include <string>
#include <initializer_list>
#include <executorch/runtime/backend/backend_options.h>

namespace executorch {
namespace runtime {

class DynamicBackendOptionsMap {
 public:
  using OptionList = std::initializer_list<BackendOption>;

  DynamicBackendOptionsMap(
      std::initializer_list<std::pair<const char*, OptionList>> list) {
    entries_.reserve(list.size());
    for (const auto& item : list) {
      // Store backend name
      backend_names_.push_back(item.first);
      // Store options
      options_storage_.push_back(std::vector<BackendOption>(item.second));
      // Create Entry with stable references
      entries_.push_back({
          backend_names_.back().c_str(),
          ArrayRef<BackendOption>(options_storage_.back().data(), options_storage_.back().size())
      });
    }
  }

  ArrayRef<Entry> entries() const {
    return ArrayRef<Entry>(entries_.data(), entries_.size());
  }

 private:
  std::vector<std::string> backend_names_;
  std::vector<std::vector<BackendOption>> options_storage_;
  std::vector<Entry> entries_;
};

} // namespace runtime
} // namespace executorch
