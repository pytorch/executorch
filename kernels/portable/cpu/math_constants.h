/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * On most platforms these constants are brought in through cmath or math.h.
 * However bringing it in through those requires a preprocesor macro, so its
 * portability is questionable. This is provided as an option for kernel
 * devlopers to get the constants regardless of whether or not their version of
 * cmath or math.h defines these constants
 */
#include <cmath>

#ifndef M_E
#define M_E 2.7182818284590452354 /* e */
#endif

#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634074 /* log_2 e */
#endif

#ifndef M_LOG10E
#define M_LOG10E 0.43429448190325182765 /* log_10 e */
#endif

#ifndef M_LN2
#define M_LN2 0.69314718055994530942 /* log_e 2 */
#endif

#ifndef M_LN10
#define M_LN10 2.30258509299404568402 /* log_e 10 */
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846 /* pi */
#endif

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923 /* pi/2 */
#endif

#ifndef M_PI_4
#define M_PI_4 0.78539816339744830962 /* pi/4 */
#endif

#ifndef M_1_PI
#define M_1_PI 0.31830988618379067154 /* 1/pi */
#endif

#ifndef M_2_PI
#define M_2_PI 0.63661977236758134308 /* 2/pi */
#endif

#ifndef M_2_SQRTPI
#define M_2_SQRTPI 1.12837916709551257390 /* 2/sqrt(pi) */
#endif

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880 /* sqrt(2) */
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440 /* 1/sqrt(2) */
#endif
