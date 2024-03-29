load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_flatcc_targets():
    # Flatcc public headers
    PUBLIC_HEADERS = [
        "flatcc/config/config.h",
        "flatcc/include/flatcc/flatcc.h",
        "flatcc/include/flatcc/flatcc_assert.h",
        "flatcc/include/flatcc/flatcc_version.h",
        "flatcc/include/flatcc/flatcc_emitter.h",
        "flatcc/include/flatcc/flatcc_alloc.h",
        "flatcc/include/flatcc/flatcc_json_printer.h",
        "flatcc/include/flatcc/flatcc_verifier.h",
        "flatcc/include/flatcc/flatcc_refmap.h",
        "flatcc/include/flatcc/flatcc_unaligned.h",
        "flatcc/include/flatcc/portable/grisu3_print.h",
        "flatcc/include/flatcc/portable/pprintfp.h",
        "flatcc/include/flatcc/portable/pbase64.h",
        "flatcc/include/flatcc/portable/punaligned.h",
        "flatcc/include/flatcc/portable/pdiagnostic.h",
        "flatcc/include/flatcc/portable/pinttypes.h",
        "flatcc/include/flatcc/portable/pinline.h",
        "flatcc/include/flatcc/portable/pprintint.h",
        "flatcc/include/flatcc/portable/pdiagnostic_pop.h",
        "flatcc/include/flatcc/portable/include/std/stdalign.h",
        "flatcc/include/flatcc/portable/include/std/inttypes.h",
        "flatcc/include/flatcc/portable/include/std/stdbool.h",
        "flatcc/include/flatcc/portable/include/std/stdint.h",
        "flatcc/include/flatcc/portable/include/linux/endian.h",
        "flatcc/include/flatcc/portable/pversion.h",
        "flatcc/include/flatcc/portable/pstdalign.h",
        "flatcc/include/flatcc/portable/pdiagnostic_push.h",
        "flatcc/include/flatcc/portable/pendian_detect.h",
        "flatcc/include/flatcc/portable/paligned_alloc.h",
        "flatcc/include/flatcc/portable/pendian.h",
        "flatcc/include/flatcc/portable/pstatic_assert.h",
        "flatcc/include/flatcc/portable/pwarnings.h",
        "flatcc/include/flatcc/portable/pparsefp.h",
        "flatcc/include/flatcc/portable/portable_basic.h",
        "flatcc/include/flatcc/portable/portable.h",
        "flatcc/include/flatcc/portable/grisu3_math.h",
        "flatcc/include/flatcc/portable/pstdint.h",
        "flatcc/include/flatcc/portable/pstdbool.h",
        "flatcc/include/flatcc/portable/pstatic_assert_scope.h",
        "flatcc/include/flatcc/portable/grisu3_parse.h",
        "flatcc/include/flatcc/portable/pparseint.h",
        "flatcc/include/flatcc/flatcc_endian.h",
        "flatcc/include/flatcc/flatcc_iov.h",
        "flatcc/include/flatcc/flatcc_rtconfig.h",
        "flatcc/include/flatcc/flatcc_accessors.h",
        "flatcc/include/flatcc/flatcc_epilogue.h",
        "flatcc/include/flatcc/flatcc_identifier.h",
        "flatcc/include/flatcc/flatcc_prologue.h",
        "flatcc/include/flatcc/flatcc_builder.h",
        "flatcc/include/flatcc/support/readfile.h",
        "flatcc/include/flatcc/support/cdump.h",
        "flatcc/include/flatcc/support/elapsed.h",
        "flatcc/include/flatcc/support/hexdump.h",
        "flatcc/include/flatcc/flatcc_json_parser.h",
        "flatcc/include/flatcc/flatcc_flatbuffers.h",
        "flatcc/include/flatcc/flatcc_portable.h",
        "flatcc/include/flatcc/flatcc_types.h",
        "flatcc/include/flatcc/reflection/reflection_reader.h",
        "flatcc/include/flatcc/reflection/flatbuffers_common_reader.h",
        "flatcc/include/flatcc/reflection/reflection_builder.h",
        "flatcc/include/flatcc/reflection/reflection_verifier.h",
        "flatcc/include/flatcc/reflection/flatbuffers_common_builder.h",
    ]

    # FlatCC static libraries

    runtime.cxx_library(
        name = "flatccrt",
        srcs = [
            "flatcc/src/runtime/builder.c",
            "flatcc/src/runtime/emitter.c",
            "flatcc/src/runtime/refmap.c",
            "flatcc/src/runtime/verifier.c",
            "flatcc/src/runtime/json_parser.c",
            "flatcc/src/runtime/json_printer.c",
        ],
        public_include_directories = [
            "flatcc/include",
            "flatcc/config",
            "include",
        ],
        headers = PUBLIC_HEADERS,
        visibility = ["@EXECUTORCH_CLIENTS"],
    )

    runtime.cxx_library(
        name = "flatcc",
        srcs = [
            "flatcc/external/hash/cmetrohash64.c",
            "flatcc/external/hash/ptr_set.c",
            "flatcc/external/hash/str_set.c",
            "flatcc/src/compiler/codegen_c.c",
            "flatcc/src/compiler/codegen_c_builder.c",
            "flatcc/src/compiler/codegen_c_json_parser.c",
            "flatcc/src/compiler/codegen_c_json_printer.c",
            "flatcc/src/compiler/codegen_c_reader.c",
            "flatcc/src/compiler/codegen_c_sort.c",
            "flatcc/src/compiler/codegen_c_sorter.c",
            "flatcc/src/compiler/codegen_c_verifier.c",
            "flatcc/src/compiler/codegen_schema.c",
            "flatcc/src/compiler/coerce.c",
            "flatcc/src/compiler/fileio.c",
            "flatcc/src/compiler/flatcc.c",
            "flatcc/src/compiler/hash_tables/name_table.c",
            "flatcc/src/compiler/hash_tables/schema_table.c",
            "flatcc/src/compiler/hash_tables/scope_table.c",
            "flatcc/src/compiler/hash_tables/symbol_table.c",
            "flatcc/src/compiler/hash_tables/value_set.c",
            "flatcc/src/compiler/parser.c",
            "flatcc/src/compiler/semantics.c",
        ],
        compiler_flags = [
            "-D FLATCC_REFLECTION=1",
            "-D FLATCC_ALLOW_RPC_SERVICE_ATTRIBUTES=1",
            "-D FLATCC_ALLOW_RPC_METHOD_ATTRIBUTES=1",
            "-D FLATCC_JSON_PARSE_FORCE_DEFAULTS=0",
        ],
        include_directories = [
            "flatcc/external",
        ],
        public_include_directories = [
            "flatcc/include",
            "flatcc/config",
        ],
        headers = PUBLIC_HEADERS + [
            "flatcc/external/lex/tokens.h",
            "flatcc/external/lex/luthor.h",
            "flatcc/external/lex/luthor.c",
            "flatcc/external/hash/hash_table_impl_rh.h",
            "flatcc/external/hash/ht64rh.h",
            "flatcc/external/hash/unaligned.h",
            "flatcc/external/hash/ht64.h",
            "flatcc/external/hash/PMurHash.h",
            "flatcc/external/hash/ht_portable.h",
            "flatcc/external/hash/hash_table_def.h",
            "flatcc/external/hash/int_set.h",
            "flatcc/external/hash/hash_table.h",
            "flatcc/external/hash/cmetrohash.h",
            "flatcc/external/hash/ht_hash_function.h",
            "flatcc/external/hash/ht32rh.h",
            "flatcc/external/hash/ptr_set.h",
            "flatcc/external/hash/hash_table_impl.h",
            "flatcc/external/hash/ht32.h",
            "flatcc/external/hash/ht_trace.h",
            "flatcc/external/hash/pstdint.h",
            "flatcc/external/hash/str_set.h",
            "flatcc/external/hash/token_map.h",
            "flatcc/external/hash/hash.h",
            "flatcc/external/grisu3/grisu3_print.h",
            "flatcc/external/grisu3/grisu3_math.h",
            "flatcc/external/grisu3/grisu3_parse.h",
            "flatcc/src/compiler/symbols.h",
            "flatcc/src/compiler/parser.h",
            "flatcc/src/compiler/codegen_c.h",
            "flatcc/src/compiler/semantics.h",
            "flatcc/src/compiler/catalog.h",
            "flatcc/src/compiler/codegen.h",
            "flatcc/src/compiler/coerce.h",
            "flatcc/src/compiler/pstrutil.h",
            "flatcc/src/compiler/fileio.h",
            "flatcc/src/compiler/keywords.h",
            "flatcc/src/compiler/codegen_c_sort.h",
        ],
        visibility = ["@EXECUTORCH_CLIENTS"],
        deps = [":flatccrt"],
    )

    runtime.cxx_library(
        name = "flatcc-host",
        srcs = [
            "flatcc/external/hash/cmetrohash64.c",
            "flatcc/external/hash/ptr_set.c",
            "flatcc/external/hash/str_set.c",
            "flatcc/src/compiler/codegen_c.c",
            "flatcc/src/compiler/codegen_c_builder.c",
            "flatcc/src/compiler/codegen_c_json_parser.c",
            "flatcc/src/compiler/codegen_c_json_printer.c",
            "flatcc/src/compiler/codegen_c_reader.c",
            "flatcc/src/compiler/codegen_c_sort.c",
            "flatcc/src/compiler/codegen_c_sorter.c",
            "flatcc/src/compiler/codegen_c_verifier.c",
            "flatcc/src/compiler/codegen_schema.c",
            "flatcc/src/compiler/coerce.c",
            "flatcc/src/compiler/fileio.c",
            "flatcc/src/compiler/flatcc.c",
            "flatcc/src/compiler/hash_tables/name_table.c",
            "flatcc/src/compiler/hash_tables/schema_table.c",
            "flatcc/src/compiler/hash_tables/scope_table.c",
            "flatcc/src/compiler/hash_tables/symbol_table.c",
            "flatcc/src/compiler/hash_tables/value_set.c",
            "flatcc/src/compiler/parser.c",
            "flatcc/src/compiler/semantics.c",
            "flatcc/src/runtime/builder.c",
            "flatcc/src/runtime/emitter.c",
            "flatcc/src/runtime/refmap.c",
        ],
        compiler_flags = [
            "-D FLATCC_REFLECTION=1",
            "-D FLATCC_JSON_PARSE_FORCE_DEFAULTS=0",
        ],
        include_directories = [
            "flatcc/external",
        ],
        public_include_directories = [
            "flatcc/include",
            "flatcc/config",
        ],
        headers = PUBLIC_HEADERS.append([
            "flatcc/external/lex/tokens.h",
            "flatcc/external/lex/luthor.h",
            "flatcc/external/lex/luthor.c",
            "flatcc/external/hash/hash_table_impl_rh.h",
            "flatcc/external/hash/ht64rh.h",
            "flatcc/external/hash/unaligned.h",
            "flatcc/external/hash/ht64.h",
            "flatcc/external/hash/PMurHash.h",
            "flatcc/external/hash/ht_portable.h",
            "flatcc/external/hash/hash_table_def.h",
            "flatcc/external/hash/int_set.h",
            "flatcc/external/hash/hash_table.h",
            "flatcc/external/hash/cmetrohash.h",
            "flatcc/external/hash/ht_hash_function.h",
            "flatcc/external/hash/ht32rh.h",
            "flatcc/external/hash/ptr_set.h",
            "flatcc/external/hash/hash_table_impl.h",
            "flatcc/external/hash/ht32.h",
            "flatcc/external/hash/ht_trace.h",
            "flatcc/external/hash/pstdint.h",
            "flatcc/external/hash/str_set.h",
            "flatcc/external/hash/token_map.h",
            "flatcc/external/hash/hash.h",
            "flatcc/external/grisu3/grisu3_print.h",
            "flatcc/external/grisu3/grisu3_math.h",
            "flatcc/external/grisu3/grisu3_parse.h",
            "flatcc/src/compiler/symbols.h",
            "flatcc/src/compiler/parser.h",
            "flatcc/src/compiler/codegen_c.h",
            "flatcc/src/compiler/semantics.h",
            "flatcc/src/compiler/catalog.h",
            "flatcc/src/compiler/codegen.h",
            "flatcc/src/compiler/coerce.h",
            "flatcc/src/compiler/pstrutil.h",
            "flatcc/src/compiler/fileio.h",
            "flatcc/src/compiler/keywords.h",
            "flatcc/src/compiler/codegen_c_sort.h",
        ]),
        visibility = ["@EXECUTORCH_CLIENTS"],
    )

    # FlatCC CLI
    runtime.cxx_binary(
        name = "flatcc-cli",
        srcs = [
            "flatcc/src/cli/flatcc_cli.c",
        ],
        compiler_flags = [
            "-D FLATCC_REFLECTION=1",
        ],
        include_directories = [
            "flatcc/include",
            "flatcc/config",
        ],
        deps = [":flatcc-host"],
        visibility = ["@EXECUTORCH_CLIENTS"],
    )
