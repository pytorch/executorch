/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// NOTE: This is a modified excerpt of
//  https://github.com/PENGUINLIONG/graphi-t/blob/da31ec530df07c9899e056eeced08a64062dcfce/src/args.cpp;
// MIT-licensed by Rendong Liang.
#include <iostream>
#include <map>
#include <vector>

#include "args.h"
#include "assert.h"

namespace gpuinfo {

namespace args {

struct ArgumentHelp {
  std::string short_flag;
  std::string long_flag;
  std::string help;
};
struct ArgumentConfig {
  std::string app_name = "[APPNAME]";
  std::string desc;
  // Short flag name -> ID.
  std::map<char, size_t> short_map;
  // Long flag name -> ID.
  std::map<std::string, size_t> long_map;
  // Argument parsing info.
  std::vector<ArgumentParseConfig> parse_cfgs;
  // Argument help info.
  std::vector<ArgumentHelp> helps;
} arg_cfg;

void init_arg_parse(const char* app_name, const char* desc) {
  arg_cfg.app_name = app_name;
  arg_cfg.desc = desc;
}
const char* get_app_name() {
  return arg_cfg.app_name.c_str();
}
void print_help() {
  std::cout << "usage: " << arg_cfg.app_name << " [OPTIONS]" << std::endl;
  if (!arg_cfg.desc.empty()) {
    std::cout << arg_cfg.desc << std::endl;
  }
  for (const auto& help : arg_cfg.helps) {
    std::cout << help.short_flag << "\t" << help.long_flag << "\t\t"
              << help.help << std::endl;
  }
  std::cout << "-h\t--help\t\tPrint this message." << std::endl;
  std::exit(0);
}
void report_unknown_arg(const char* arg) {
  std::cout << "unknown argument: " << arg << std::endl;
  print_help();
}

void reg_arg(
    const char* short_flag,
    const char* long_flag,
    const ArgumentParseConfig& parse_cfg,
    const char* help) {
  using std::strlen;
  size_t i = arg_cfg.parse_cfgs.size();
  if (strlen(short_flag) == 2 && short_flag[0] == '-') {
    arg_cfg.short_map[short_flag[1]] = i;
  }
  if (strlen(long_flag) > 3 && long_flag[1] == '-' && long_flag[0] == '-') {
    arg_cfg.long_map[long_flag + 2] = i;
  }
  arg_cfg.parse_cfgs.emplace_back(parse_cfg);
  std::string help_str = help;
  auto lit = parse_cfg.lit(parse_cfg.dst);
  if (!lit.empty()) {
    help_str += " (default=" + lit + ")";
  }
  ArgumentHelp arg_help{short_flag, long_flag, help_str};
  arg_cfg.helps.emplace_back(std::move(arg_help));
}

void parse_args(int argc, const char** argv) {
  auto i = 1;
  int iarg_entry = -1;
  while (i < argc || iarg_entry >= 0) {
    if (iarg_entry >= 0) {
      auto& parse_cfg = arg_cfg.parse_cfgs[iarg_entry];
      gpuinfo::assert(
          parse_cfg.parser(argv + i, parse_cfg.dst),
          "unable to parse argument");
      gpuinfo::assert(
          (argc - i >= parse_cfg.narg), "no enough argument segments");
      i += parse_cfg.narg;
      iarg_entry = -1;
    } else {
      const char* arg = argv[i];
      if (arg[0] != '-') {
        // Free argument.
        gpuinfo::panic("free argument is currently unsupported");
      } else if (arg[1] != '-') {
        if (arg[1] == 'h') {
          print_help();
        }
        // Short flag argument.
        auto it = arg_cfg.short_map.find(arg[1]);
        if (it != arg_cfg.short_map.end()) {
          iarg_entry = it->second;
        } else {
          report_unknown_arg(arg);
        }
        ++i;
      } else {
        if (std::strcmp(arg + 2, "help") == 0) {
          print_help();
        }
        // Long flag argument.
        auto it = (arg_cfg.long_map.find(arg + 2));
        if (it != arg_cfg.long_map.end()) {
          iarg_entry = it->second;
        } else {
          report_unknown_arg(arg);
        }
        ++i;
      }
    }
  }
}

} // namespace args

} // namespace gpuinfo
