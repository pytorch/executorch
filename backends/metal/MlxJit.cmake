# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Shared CMake glue for the metal_v2 per-shape MLX JIT path
# (ops/mlx_jit/). Two CMakeLists.txt files build the metal_v2 sources:
#
#   - backends/apple/metal/CMakeLists.txt        (AOTI Metal backend, v2)
#   - backends/native/CMakeLists.txt             (native_backend, Apple metal provider)
#
# Both need to:
#   1. Generate per-snippet .cpp files at build time from the MLX 0.31.2
#      submodule headers (one .cpp per make_jit_source boundary).
#   2. Compile ops/mlx_jit/KernelLoader.mm (the loader implementation that
#      pulls those snippets together with TemplateGen + the generic
#      MetalKernelCompiler primitive).
#   3. Apply -fno-objc-arc to KernelLoader.mm (matches the existing
#      manual-retain memory model used by the rest of metal_v2).
#
# This file packages all three so each consumer is one include + one
# function call:
#
#     include(${CMAKE_CURRENT_SOURCE_DIR}/../backends/metal/MlxJit.cmake)
#     add_metal_v2_mlx_jit_sources(_my_source_list_var)
#
# The function APPENDS to the named source list (PARENT_SCOPE). It also
# applies the no-ARC source property to KernelLoader.mm.
#
# Self-locating: derives all paths from CMAKE_CURRENT_LIST_DIR at include
# time, so callers don't need to set EXECUTORCH_ROOT or similar.

# Resolve paths relative to THIS file (backends/backends/metal/).
set(_mlx_jit_module_dir
    ${CMAKE_CURRENT_LIST_DIR}/ops/mlx_jit
    CACHE INTERNAL "ops/mlx_jit/ source directory"
)
# Submodule lives at backends/mlx/third-party/mlx/. From this file
# (backends/metal/MlxJit.cmake) that's one level up then back down
# through mlx/third-party/mlx.
get_filename_component(
  _mlx_submodule_root_default
  "${CMAKE_CURRENT_LIST_DIR}/../mlx/third-party/mlx"
  ABSOLUTE
)
set(_mlx_jit_submodule_dir
    ${_mlx_submodule_root_default}
    CACHE INTERNAL "MLX 0.31.2 submodule root (mlx/backend/metal/kernels/...)"
)
set(_mlx_jit_snippet_script
    ${_mlx_jit_module_dir}/make_mlx_jit_snippet.sh
    CACHE INTERNAL "Script that generates per-snippet .cpp from MLX headers"
)

# The set of MLX header subpaths (under kernels/, no .h suffix) we vendor as
# JIT snippets. Kept in sync with ops/mlx_jit/Snippets.h declarations.
# basename(SUBPATH) becomes the function name in mlx_jit::Snippets:: e.g.
#   utils                                       -> utils()
#   steel/gemm/gemm                             -> gemm()
#   steel/gemm/kernels/steel_gemm_fused_nax     -> steel_gemm_fused_nax()
set(_mlx_jit_snippet_subpaths
    utils
    steel/gemm/gemm
    steel/gemm/gemm_nax
    steel/gemm/kernels/steel_gemm_fused
    steel/gemm/kernels/steel_gemm_splitk
    steel/gemm/kernels/steel_gemm_fused_nax
    steel/gemm/kernels/steel_gemm_splitk_nax
    CACHE INTERNAL "MLX header subpaths to vendor as JIT snippets"
)

# Local-source snippets: vendored from files in our own tree (e.g. thin
# wrappers that #include from MLX with macros redefined). Format:
# "subpath_for_function_name|absolute_path_to_source_file" pairs.
set(_mlx_jit_local_snippets
    "gemv|${_mlx_jit_module_dir}/local_snippets/gemv.h"
    "sdpa_vector|${_mlx_jit_module_dir}/local_snippets/sdpa_vector.h"
    "steel_attention|${_mlx_jit_module_dir}/local_snippets/steel_attention.h"
    "steel_attention_nax|${_mlx_jit_module_dir}/local_snippets/steel_attention_nax.h"
    "quantized|${_mlx_jit_module_dir}/local_snippets/quantized.h"
    "quantized_nax|${_mlx_jit_module_dir}/local_snippets/quantized_nax.h"
    CACHE INTERNAL "Local snippet sources (basename|abs_path)"
)

# Append the metal_v2 mlx_jit sources (generated snippets + KernelLoader.mm)
# to the caller-provided source list variable. Outputs are written to
# ${CMAKE_BINARY_DIR}/mlx_jit_snippets/ so multiple consumers in the same
# build tree share a single set of generated files.
function(add_metal_v2_mlx_jit_sources OUT_VAR)
  set(_out_dir ${CMAKE_BINARY_DIR}/mlx_jit_snippets)
  set(_added)

  # MLX-submodule snippets: source path derived from SUBPATH.
  foreach(_subpath IN LISTS _mlx_jit_snippet_subpaths)
    get_filename_component(_snippet_name ${_subpath} NAME)
    set(_snippet_cpp ${_out_dir}/${_snippet_name}.cpp)
    set(_snippet_input
        ${_mlx_jit_submodule_dir}/mlx/backend/metal/kernels/${_subpath}.h
    )
    get_property(_already_defined GLOBAL PROPERTY _MLX_JIT_${_snippet_name}_DEFINED)
    if(NOT _already_defined)
      add_custom_command(
        OUTPUT ${_snippet_cpp}
        COMMAND
          bash ${_mlx_jit_snippet_script} ${_out_dir}
          ${_mlx_jit_submodule_dir} ${_subpath}
        DEPENDS ${_mlx_jit_snippet_script} ${_snippet_input}
        COMMENT "Generating MLX JIT snippet ${_snippet_name} from ${_subpath}.h"
        VERBATIM
      )
      set_property(GLOBAL PROPERTY _MLX_JIT_${_snippet_name}_DEFINED TRUE)
    endif()
    list(APPEND _added ${_snippet_cpp})
  endforeach()

  # Local-source snippets: thin wrappers that #include MLX kernels with
  # AOT-instantiation macros redefined. Same script, source path passed
  # explicitly as the 4th arg.
  foreach(_entry IN LISTS _mlx_jit_local_snippets)
    string(REPLACE "|" ";" _pair ${_entry})
    list(GET _pair 0 _snippet_name)
    list(GET _pair 1 _snippet_input)
    set(_snippet_cpp ${_out_dir}/${_snippet_name}.cpp)
    get_property(_already_defined GLOBAL PROPERTY _MLX_JIT_${_snippet_name}_DEFINED)
    if(NOT _already_defined)
      add_custom_command(
        OUTPUT ${_snippet_cpp}
        COMMAND
          bash ${_mlx_jit_snippet_script} ${_out_dir}
          ${_mlx_jit_submodule_dir} ${_snippet_name} ${_snippet_input}
        DEPENDS ${_mlx_jit_snippet_script} ${_snippet_input}
        COMMENT "Generating MLX JIT snippet ${_snippet_name} from ${_snippet_input}"
        VERBATIM
      )
      set_property(GLOBAL PROPERTY _MLX_JIT_${_snippet_name}_DEFINED TRUE)
    endif()
    list(APPEND _added ${_snippet_cpp})
  endforeach()

  # KernelLoader.mm — implements typeToName / KernelLoader::getDenseGemmKernel
  # / getDenseNaxKernel / getSplitKKernel / getSplitKNaxKernel /
  # getSplitKAccumKernel and the mlx_jit::shared() singleton accessor.
  set(_loader_mm ${_mlx_jit_module_dir}/KernelLoader.mm)
  list(APPEND _added ${_loader_mm})

  # Match the rest of metal_v2's manual-retain build: KernelLoader.mm has
  # to be compiled without ARC. The .mm extension auto-resolves to OBJCXX
  # — we deliberately do NOT set LANGUAGE OBJCXX here because that would
  # force callers to pre-enable_language(OBJCXX); the existing
  # backends/apple/metal/CMakeLists.txt doesn't (and doesn't need to —
  # the file-extension dispatch handles it on Apple platforms).
  set_source_files_properties(
    ${_loader_mm}
    PROPERTIES COMPILE_FLAGS "-fno-objc-arc"
  )

  set(${OUT_VAR}
      ${${OUT_VAR}} ${_added}
      PARENT_SCOPE
  )
endfunction()
