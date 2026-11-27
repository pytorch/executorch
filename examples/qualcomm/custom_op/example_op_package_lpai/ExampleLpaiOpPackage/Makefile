# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ==============================================================================
# Makefile for ExampleLpaiOpPackage
# Builds libQnnExampleLpaiOpPackage.so for multiple targets:
#   - x86_64-linux-clang  (compiler-side, host)
#   - hexagon-v79         (inference-side, DSP)
#
# NOTE: the library is named libQnn<PackageName>.so to match the naming
# convention used by the other ExecuTorch op package examples.
# ==============================================================================

# define default
default: all

# define package name
export PACKAGE_NAME := ExampleLpaiOpPackage

# define source directories
export SRC_DIR     := src
export SRC_DIR_OPS := src/ops
export INCLUDE_DIR := include

make_dir := makefiles

# define library name
export LIBRARY_NAME := libQnn$(PACKAGE_NAME).so

# ==============================================================================
# Environment checks
# ==============================================================================

.PHONY: check_qnn_sdk
check_qnn_sdk:
ifndef QNN_SDK_ROOT
	$(error ERROR: QNN_SDK_ROOT is not set. Please set it to the QNN SDK root path.)
endif

.PHONY: check_hexagon_sdk
check_hexagon_sdk:
ifndef HEXAGON_SDK_ROOT
	$(error ERROR: HEXAGON_SDK_ROOT is not set. Please set it to the Hexagon SDK root path.)
endif
ifndef HEXAGON_TOOLS_VERSION
	$(error ERROR: HEXAGON_TOOLS_VERSION is not set. Please set it to the Hexagon Tools version.)
endif

# ==============================================================================
# Build targets
# ==============================================================================

.PHONY: all
all: lpai_x86 lpai_hexagon_v79

# x86_64 target (compiler-side, used by QNN compiler/tools on host)
.PHONY: lpai_x86
lpai_x86: check_qnn_sdk
	$(MAKE) -f $(make_dir)/Makefile.linux-x86_64

# hexagon-v79 target (inference-side, runs on DSP)
.PHONY: lpai_hexagon_v79
lpai_hexagon_v79: check_qnn_sdk check_hexagon_sdk
	$(MAKE) -f $(make_dir)/Makefile.hexagon-v79

# ==============================================================================
# Clean targets
# ==============================================================================

.PHONY: clean
clean: clean_x86 clean_hexagon

.PHONY: clean_x86
clean_x86:
	@rm -rf libs/x86_64-linux-clang obj/x86_64-linux-clang

.PHONY: clean_hexagon
clean_hexagon:
	@rm -rf libs/hexagon-v79 obj/hexagon-v79


# ==============================================================================
# Help
# ==============================================================================

.PHONY: help
help:
	@echo "Makefile for $(PACKAGE_NAME)"
	@echo ""
	@echo "Targets:"
	@echo "  all              - Build for x86_64 and hexagon-v79 (default)"
	@echo "  lpai_x86         - Build for x86_64-linux-clang (compiler-side, host)"
	@echo "  lpai_hexagon_v79 - Build for hexagon-v79 (inference-side, DSP)"
	@echo "  clean            - Remove all build artifacts"
	@echo ""
	@echo "Required environment variables:"
	@echo "  QNN_SDK_ROOT          - Path to QNN SDK"
	@echo "  HEXAGON_SDK_ROOT      - Path to Hexagon SDK (for the hexagon target)"
	@echo "  HEXAGON_TOOLS_VERSION - Hexagon tools version (for the hexagon target)"
	@echo ""
	@echo "Output:"
	@echo "  libs/x86_64-linux-clang/$(LIBRARY_NAME)"
	@echo "  libs/hexagon-v79/$(LIBRARY_NAME)"
	@echo "  libs/hexagon-v79/libLpaiOpPackageIsland.so"
