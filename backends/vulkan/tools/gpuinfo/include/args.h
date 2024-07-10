/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// NOTE: This is a modified excerpt of
//  https://github.com/PENGUINLIONG/graphi-t/blob/da31ec530df07c9899e056eeced08a64062dcfce/args.hpp;
// MIT-licensed by Rendong Liang.

// Argument parsing utilities.
// @PENGUINLIONG
#pragma once
#include <string>

namespace gpuinfo {

namespace args {

struct ArgumentParseConfig {
  // Expected number of arguments segments.
  uint32_t narg;
  // Returns true if the parsing is successful.
  bool (*parser)(const char*[], void*);
  // Returns the literal of default value.
  std::string (*lit)(const void*);
  // Destination to be written with parsed value.
  void* dst;
};

// Optionally initialize argument parser with application name and usage
// description.
extern void init_arg_parse(const char* app_name, const char* desc);
// Get the name of this app set by the user. Empty string is returned if this
// function is called before `init_arg_parse`.
extern const char* get_app_name();
// Print help message to the standard output.
extern void print_help();
// Erase the type of argument parser and bind the type-erased parser to the
// value destination. User code MUST ensure the `dst` buffer can contain the
// parsing result.
template <typename TTypedParser>
ArgumentParseConfig make_parse_cfg(void* dst) {
  ArgumentParseConfig parse_cfg;
  parse_cfg.narg = TTypedParser::narg;
  parse_cfg.dst = dst;
  parse_cfg.parser = &TTypedParser::parse;
  parse_cfg.lit = &TTypedParser::lit;
  return parse_cfg;
}
// Register customized argument parsing.
extern void reg_arg(
    const char* short_flag,
    const char* long_flag,
    const ArgumentParseConfig& parse_cfg,
    const char* help);
// Register a structural argument parsing.
template <typename TTypedParser>
inline void reg_arg(
    const char* short_flag,
    const char* long_flag,
    typename TTypedParser::arg_ty& dst,
    const char* help) {
  reg_arg(short_flag, long_flag, make_parse_cfg<TTypedParser>(&dst), help);
}
// Parse arguments. Arguments will be matched against argument parsers
// registered before.
extern void parse_args(int argc, const char** argv);

//
// Parsers.
//

template <typename T>
struct TypedArgumentParser {
  typedef struct {
  } arg_ty;
  // Number of argument entries needed for this argument.
  static const uint32_t narg = -1;
  // Parser function. Convert the literal in the first parameter into structured
  // native representation. Return `true` on success.
  static bool parse(const char* lit[], void* dst) {
    return false;
  }
  static std::string lit(const void* src) {
    return {};
  }
};
template <>
struct TypedArgumentParser<std::string> {
  typedef std::string arg_ty;
  static const uint32_t narg = 1;
  static bool parse(const char* lit[], void* dst) {
    *(std::string*)dst = lit[0];
    return true;
  }
  static std::string lit(const void* src) {
    return *(const std::string*)src;
  }
};
template <>
struct TypedArgumentParser<int32_t> {
  typedef int arg_ty;
  static const uint32_t narg = 1;
  static bool parse(const char* lit[], void* dst) {
    *(int32_t*)dst = std::atoi(lit[0]);
    return true;
  }
  static std::string lit(const void* src) {
    return std::to_string(*(const arg_ty*)src);
  }
};
template <>
struct TypedArgumentParser<uint32_t> {
  typedef uint32_t arg_ty;
  static const uint32_t narg = 1;
  static bool parse(const char* lit[], void* dst) {
    *(uint32_t*)dst = std::atoi(lit[0]);
    return true;
  }
  static std::string lit(const void* src) {
    return std::to_string(*(const arg_ty*)src);
  }
};
template <>
struct TypedArgumentParser<float> {
  typedef float arg_ty;
  static const uint32_t narg = 1;
  static bool parse(const char* lit[], void* dst) {
    *(float*)dst = std::atof(lit[0]);
    return true;
  }
  static std::string lit(const void* src) {
    return std::to_string(*(const arg_ty*)src);
  }
};
// NOTE: This is used for arguments like `-f true` and `-f false`. If you need a
// boolean argument that don't need to be set explicitly. Use
// `SwitchArgumentParser` instead.
template <>
struct TypedArgumentParser<bool> {
  typedef bool arg_ty;
  static const uint32_t narg = 1;
  static bool parse(const char* lit[], void* dst) {
    if (strcmp(lit[0], "true") == 0 || strcmp(lit[0], "True") == 0) {
      *(bool*)dst = true;
      return true;
    } else if (strcmp(lit[0], "false") == 0 || strcmp(lit[0], "False") == 0) {
      *(bool*)dst = false;
      return true;
    } else {
      return false;
    }
  }
  static std::string lit(const void* src) {
    if (*(const arg_ty*)src) {
      return "true";
    } else {
      return "false";
    }
  }
};
struct SwitchArgumentParser {
  typedef bool arg_ty;
  static const uint32_t narg = 0;
  static bool parse(const char* lit[], void* dst) {
    *(bool*)dst = true;
    return true;
  }
  static std::string lit(const void* src) {
    return {};
  }
};

using IntParser = TypedArgumentParser<int32_t>;
using UintParser = TypedArgumentParser<uint32_t>;
using FloatParser = TypedArgumentParser<float>;
using BoolParser = TypedArgumentParser<bool>;
using StringParser = TypedArgumentParser<std::string>;
using SwitchParser = SwitchArgumentParser;

} // namespace args

} // namespace gpuinfo
