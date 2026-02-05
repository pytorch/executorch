/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/pybind11.h>

#include <executorch/extension/pybindings/pybindings_data_loader.h>

namespace py = pybind11;

using ::executorch::extension::pybindings::PyDataLoader;

PYBIND11_MODULE(data_loader, m) {
  py::class_<PyDataLoader, std::shared_ptr<PyDataLoader>>(m, "PyDataLoader");
}
