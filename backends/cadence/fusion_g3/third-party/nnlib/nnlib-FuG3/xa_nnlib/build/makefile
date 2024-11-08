#
# Copyright (c) 2024 Cadence Design Systems, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to use this Software with Cadence processor cores only and
# not with any other processors and platforms, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

.PHONY: all install clean

ifneq ($(BUILD_SCRATCH_SIZE_LIB), 1)
include detect_core.mk
endif

all: nnlib

install: install_nnlib

clean: clean_nnlib

nnlib:
	$(MAKE) -f makefile_nn_lib_fusion_g3 all

install_nnlib: nnlib
	$(MAKE) -f makefile_nn_lib_fusion_g3 install

clean_nnlib:
	$(MAKE) -f makefile_nn_lib_fusion_g3 clean

