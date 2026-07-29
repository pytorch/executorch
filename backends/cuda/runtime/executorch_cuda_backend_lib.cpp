/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This translation unit intentionally contains no symbols. The
// executorch_cuda_backend shared library exists solely to bundle the
// whole-archived aoti_cuda_backend static library so its "CudaBackend"
// registration constructor is retained and runs when the .so is loaded. CMake
// requires a SHARED target to have at least one source file.
