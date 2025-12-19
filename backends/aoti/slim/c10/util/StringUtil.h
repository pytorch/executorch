#pragma once

#include <sstream>
#include <string>

namespace standalone::c10 {
template <class Container>
inline std::string Join(const std::string& delimiter, const Container& v) {
  std::stringstream s;
  int cnt = static_cast<int64_t>(v.size()) - 1;
  for (auto i = v.begin(); i != v.end(); ++i, --cnt) {
    s << (*i) << (cnt ? delimiter : "");
  }
  return std::move(s).str();
}
} // namespace standalone::c10
