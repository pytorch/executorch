// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

// Referencing this symbol forces the PTN Module provider out of a static
// archive on linkers that otherwise discard registration-only objects.
extern "C" void executorch_native_module_ptn_link_anchor();
