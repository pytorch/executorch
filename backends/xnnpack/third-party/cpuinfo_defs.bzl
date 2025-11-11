def define_cpuinfo_and_clog():
    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "cpuinfo",
        srcs = [
            "cpuinfo-wrappers/api.c",
            "cpuinfo-wrappers/arm/android/properties.c",
            "cpuinfo-wrappers/arm/cache.c",
            "cpuinfo-wrappers/arm/linux/aarch32-isa.c",
            "cpuinfo-wrappers/arm/linux/aarch64-isa.c",
            "cpuinfo-wrappers/arm/linux/chipset.c",
            "cpuinfo-wrappers/arm/linux/clusters.c",
            "cpuinfo-wrappers/arm/linux/cpuinfo.c",
            "cpuinfo-wrappers/arm/linux/hwcap.c",
            "cpuinfo-wrappers/arm/linux/init.c",
            "cpuinfo-wrappers/arm/linux/midr.c",
            "cpuinfo-wrappers/arm/mach/init.c",
            "cpuinfo-wrappers/arm/uarch.c",
            "cpuinfo-wrappers/cache.c",
            "cpuinfo-wrappers/init.c",
            "cpuinfo-wrappers/linux/cpulist.c",
            "cpuinfo-wrappers/linux/multiline.c",
            "cpuinfo-wrappers/linux/processors.c",
            "cpuinfo-wrappers/linux/smallfile.c",
            "cpuinfo-wrappers/log.c",
            "cpuinfo-wrappers/mach/topology.c",
            "cpuinfo-wrappers/x86/cache/descriptor.c",
            "cpuinfo-wrappers/x86/cache/deterministic.c",
            "cpuinfo-wrappers/x86/cache/init.c",
            "cpuinfo-wrappers/x86/info.c",
            "cpuinfo-wrappers/x86/init.c",
            "cpuinfo-wrappers/x86/isa.c",
            "cpuinfo-wrappers/x86/linux/cpuinfo.c",
            "cpuinfo-wrappers/x86/linux/init.c",
            "cpuinfo-wrappers/x86/mach/init.c",
            "cpuinfo-wrappers/x86/name.c",
            "cpuinfo-wrappers/x86/topology.c",
            "cpuinfo-wrappers/x86/uarch.c",
            "cpuinfo-wrappers/x86/vendor.c",
            "cpuinfo-wrappers/x86/windows/init.c",
        ],
        include_directories = ["cpuinfo/src"],
        public_include_directories = ["cpuinfo/include"],
        raw_headers = native.glob([
            "cpuinfo/src/**/*.h",
            "cpuinfo/src/**/*.c",
        ]),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DCPUINFO_LOG_LEVEL=2",
            "-D_GNU_SOURCE=1",
        ],
        visibility = ["PUBLIC"],
        deps = [
            ":clog",
        ],
    )

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "clog",
        srcs = [
            "cpuinfo/deps/clog/src/clog.c",
        ],
        raw_headers = native.glob([
            "cpuinfo/deps/clog/include/*.h",
        ]),
        public_include_directories = [
            "cpuinfo/deps/clog/include/",
        ],
        force_static = True,
        visibility = ["PUBLIC"],
    )
