#!/usr/bin/env python


import confu
parser = confu.standard_parser("FP16 configuration script")
parser.add_argument("--compare", dest="compare", action="store_true",
    help="Enable performance comparison with other half-precision implementations")

def main(args):
    options = parser.parse_args(args)
    build = confu.Build.from_options(options)

    build.export_cpath("include", ["fp16.h"])

    with build.options(source_dir="test", extra_include_dirs="test", deps=[build.deps.googletest, build.deps.psimd]):
        fp16_tables = build.cxx("tables.cc")
        build.unittest("ieee-to-fp32-bits",
            [build.cxx("ieee-to-fp32-bits.cc"), fp16_tables])
        build.unittest("ieee-to-fp32-value",
            [build.cxx("ieee-to-fp32-value.cc"), fp16_tables])
        build.unittest("ieee-from-fp32-value",
            [build.cxx("ieee-from-fp32-value.cc"), fp16_tables])

        build.unittest("alt-to-fp32-bits",
            [build.cxx("alt-to-fp32-bits.cc"), fp16_tables])
        build.unittest("alt-to-fp32-value",
            [build.cxx("alt-to-fp32-value.cc"), fp16_tables])
        build.unittest("alt-from-fp32-value",
            [build.cxx("alt-from-fp32-value.cc"), fp16_tables])

        if build.target.is_x86_64:
            stubs = build.peachpy("peachpy/stubs.py")
            build.unittest("alt-xmm-to-fp32-ymm-avx", [build.cxx("peachpy/alt-xmm-to-fp32-xmm-avx.cc"), stubs])
            build.unittest("alt-xmm-to-fp32-ymm-avx2", [build.cxx("peachpy/alt-xmm-to-fp32-ymm-avx2.cc"), stubs])

        if not build.target.is_emscripten:
            build.unittest("ieee-to-fp32-psimd", build.cxx("ieee-to-fp32-psimd.cc"))
            build.unittest("alt-to-fp32-psimd", build.cxx("alt-to-fp32-psimd.cc"))

            build.unittest("ieee-to-fp32x2-psimd", build.cxx("ieee-to-fp32x2-psimd.cc"))
            build.unittest("alt-to-fp32x2-psimd", build.cxx("alt-to-fp32x2-psimd.cc"))

        build.unittest("bitcasts", build.cxx("bitcasts.cc"))

    macros = ["BENCHMARK_HAS_NO_INLINE_ASSEMBLY"]
    if options.compare:
        macros.append("FP16_COMPARATIVE_BENCHMARKS")
    with build.options(source_dir="bench", extra_include_dirs=".", macros=macros,
            deps=[build.deps.googlebenchmark, build.deps.psimd]):

        build.benchmark("ieee-element-bench", build.cxx("ieee-element.cc"))
        build.benchmark("alt-element-bench", build.cxx("alt-element.cc"))

        build.benchmark("from-ieee-array-bench", build.cxx("from-ieee-array.cc"))
        build.benchmark("from-alt-array-bench", build.cxx("from-alt-array.cc"))

        build.benchmark("to-ieee-array-bench", build.cxx("to-ieee-array.cc"))
        build.benchmark("to-alt-array-bench", build.cxx("to-alt-array.cc"))

    return build


if __name__ == "__main__":
    import sys
    main(sys.argv[1:]).generate()
