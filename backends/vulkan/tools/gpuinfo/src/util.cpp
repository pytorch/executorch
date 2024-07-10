/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// NOTE: This is a modified excerpt of
//  https://github.com/PENGUINLIONG/graphi-t/blob/da31ec530df07c9899e056eeced08a64062dcfce/src/util.cpp;
// MIT-licensed by Rendong Liang.

#include <fstream>

#include "assert.h"
#include "util.h"

namespace gpuinfo {

namespace util {

std::vector<uint8_t> load_file(const char* path) {
  std::ifstream f(path, std::ios::ate | std::ios::binary | std::ios::in);
  assert(f.is_open(), "unable to open file: ", path);
  size_t size = f.tellg();
  f.seekg(std::ios::beg);
  std::vector<uint8_t> buf;
  buf.resize(size);
  f.read((char*)buf.data(), size);
  f.close();
  return buf;
}
std::string load_text(const char* path) {
  std::ifstream f(path, std::ios::ate | std::ios::binary | std::ios::in);
  assert(f.is_open(), "unable to open file: ", path);
  size_t size = f.tellg();
  f.seekg(std::ios::beg);
  std::string buf;
  buf.reserve(size + 1);
  buf.resize(size);
  f.read((char*)buf.data(), size);
  f.close();
  return buf;
}
void save_file(const char* path, const void* data, size_t size) {
  std::ofstream f(path, std::ios::trunc | std::ios::out | std::ios::binary);
  assert(f.is_open(), "unable to open file: ", path);
  f.write((const char*)data, size);
  f.close();
}
void save_text(const char* path, const std::string& txt) {
  std::ofstream f(path, std::ios::trunc | std::ios::out | std::ios::binary);
  assert(f.is_open(), "unable to open file: ", path);
  f << txt;
  f.close();
}

} // namespace util

} // namespace gpuinfo
