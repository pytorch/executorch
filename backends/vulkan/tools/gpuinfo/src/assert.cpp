// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// NOTE: This is a modified excerpt of
//  https://github.com/PENGUINLIONG/graphi-t/blob/d291c3d1ce3795fe4b305e5efd76b4f586d23e3b/src/assert.cpp;
// MIT-licensed by Rendong Liang.
#include "assert.h"

namespace gpuinfo {

AssertionFailedException::AssertionFailedException(const std::string& msg)
    : msg(msg) {}
const char* AssertionFailedException::what() const noexcept {
  return msg.c_str();
}

} // namespace gpuinfo
