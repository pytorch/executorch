/* ****************************************************************************
 *
 * Copyright 2025 Samsung Electronics. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *****************************************************************************/

#pragma once

// #define _DEBUG
#ifdef _DEBUG

#include <android/trace.h>

class ExynosScopedTrace {
 public:
  ExynosScopedTrace(const char* name) {
    ATrace_beginSection(name);
  }
  ~ExynosScopedTrace() {
    ATrace_endSection();
  }
};

#define EXYNOS_ATRACE_FUNCTION_LINE() \
  ExynosScopedTrace ___atrace_scope(  \
      (std::string(__FUNCTION__) + ":" + std::to_string(__LINE__)).c_str())

#define EXYNOS_ATRACE_NAME_LINE(name)                            \
  ExynosScopedTrace ___atrace_scope((std::string(name) + " - " + \
                                     __FUNCTION__ + ":" +        \
                                     std::to_string(__LINE__))   \
                                        .c_str())
#define EXYNOS_ATRACE_BEGIN(name) ATrace_beginSection(name)
#define EXYNOS_ATRACE_END() ATrace_endSection()

#else
#define EXYNOS_ATRACE_FUNCTION_LINE()
#define EXYNOS_ATRACE_NAME_LINE(name)
#define EXYNOS_ATRACE_BEGIN(name)
#define EXYNOS_ATRACE_END()
#endif
