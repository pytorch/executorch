# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

target_sources(
  PyEnnWrapperAdaptor PUBLIC PyEnnWrapperAdaptor.cpp PyEnnWrapperAdaptor.h
)
target_sources(
  PyGraphWrapperAdaptor
  PUBLIC PyGraphWrapperAdaptor.cpp PyGraphWrapperAdaptor.h
         wrappers/op_param_wrapper.h wrappers/op_wrapper.h
         wrappers/tensor_wrapper.h
)
