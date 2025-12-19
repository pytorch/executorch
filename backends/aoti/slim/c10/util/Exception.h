#pragma once

#include <executorch/backends/aoti/slim/c10/macros/Macros.h>

#include <sstream>
#include <string>

// In the standalone version, STANDALONE_CHECK throws std::runtime_error
// instead of executorch::backends::aoti::slim::c10::Error.
namespace executorch::backends::aoti::slim::c10::detail {
template <typename... Args>
std::string torchCheckMsgImpl(const char* /*msg*/, const Args&... args) {
  // This is similar to the one in c10/util/Exception.h, but does
  // not depend on the more complex c10::str() function.
  // ostringstream may support less data types than c10::str(),
  // but should be sufficient in the standalone world.
  std::ostringstream oss;
  ((oss << args), ...);
  return oss.str();
}
inline const char* torchCheckMsgImpl(const char* msg) {
  return msg;
}
// If there is just 1 user-provided C-string argument, use it.
inline const char* torchCheckMsgImpl(const char* /*msg*/, const char* args) {
  return args;
}
} // namespace executorch::backends::aoti::slim::c10::detail

#define STANDALONE_CHECK_MSG(cond, type, ...)                          \
  (::executorch::backends::aoti::slim::c10::detail::torchCheckMsgImpl( \
      "Expected " #cond                                                \
      " to be true, but got false.  "                                  \
      "(Could this error message be improved?  If so, "                \
      "please report an enhancement request to PyTorch.)",             \
      ##__VA_ARGS__))
#define STANDALONE_CHECK(cond, ...)                \
  if (STANDALONE_UNLIKELY_OR_CONST(!(cond))) {     \
    throw std::runtime_error(STANDALONE_CHECK_MSG( \
        cond,                                      \
        "",                                        \
        __func__,                                  \
        ", ",                                      \
        __FILE__,                                  \
        ":",                                       \
        __LINE__,                                  \
        ", ",                                      \
        ##__VA_ARGS__));                           \
  }
#define STANDALONE_INTERNAL_ASSERT(cond, ...)      \
  if (STANDALONE_UNLIKELY_OR_CONST(!(cond))) {     \
    throw std::runtime_error(STANDALONE_CHECK_MSG( \
        cond,                                      \
        "",                                        \
        __func__,                                  \
        ", ",                                      \
        __FILE__,                                  \
        ":",                                       \
        __LINE__,                                  \
        ", ",                                      \
        #cond,                                     \
        " INTERNAL ASSERT FAILED: ",               \
        ##__VA_ARGS__));                           \
  }

#define WARNING_MESSAGE_STRING(...)                                   \
  ::executorch::backends::aoti::slim::c10::detail::torchCheckMsgImpl( \
      __VA_ARGS__)

#ifdef DISABLE_WARN
#define _STANDALONE_WARN_WITH(...) ((void)0);
#else
#define _STANDALONE_WARN_WITH(...)                                     \
  std::cerr << __func__ << ", " << __FILE__ << ":" << __LINE__ << ", " \
            << WARNING_MESSAGE_STRING(__VA_ARGS__) << std::endl;
#endif

#define STANDALONE_WARN(...) _STANDALONE_WARN_WITH(__VA_ARGS__);

#ifdef NDEBUG
// Optimized version - generates no code.
#define STANDALONE_INTERNAL_ASSERT_DEBUG_ONLY(...) \
  while (false)                                    \
  STANDALONE_EXPAND_MSVC_WORKAROUND(STANDALONE_INTERNAL_ASSERT(__VA_ARGS__))
#else
#define STANDALONE_INTERNAL_ASSERT_DEBUG_ONLY(...) \
  STANDALONE_EXPAND_MSVC_WORKAROUND(STANDALONE_INTERNAL_ASSERT(__VA_ARGS__))
#endif
