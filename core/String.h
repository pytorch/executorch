/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Changes:
- Include guard -> pragma
- Naming changes
==============================================================================*/
#pragma once

#include <sys/types.h> // TODO(T126923429): Include size_t, ssize_t

#include <cstdarg>

// Implements simple string formatting for numeric types.  Returns the number of
// bytes written to output.
extern "C" {
// Functionally equivalent to vsnprintf.
// EtSnprintf() is implemented using EtVsnprintf().
ssize_t ETVsnprintf(char* output, size_t len, const char* format, va_list args);
// Functionally equavalent to snprintf.
// For example, EtSnprintf(buffer, 10, "int %d", 10) will put the string
// "int 10" in the buffer.
// Floating point values are logged in exponent notation (1.XXX*2^N).
ssize_t ETSnprintf(char* output, size_t len, const char* format, ...);
}
