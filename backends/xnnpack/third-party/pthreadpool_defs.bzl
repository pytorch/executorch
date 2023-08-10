def define_pthreadpool():
    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "pthreadpool",
        srcs = ["pthreadpool/src/legacy-api.c", "pthreadpool/src/memory.c", "pthreadpool/src/portable-api.c", "pthreadpool/src/pthreads.c"],
        deps = [
            ":FXdiv",
        ],
        exported_deps = [
            ":pthreadpool_header",
        ],
        compiler_flags = [
            "-w",
            "-Os",
            "-fstack-protector-strong",
            "-fno-delete-null-pointer-checks",
        ],
        headers = {
            "threadpool-atomics.h": "pthreadpool/src/threadpool-atomics.h",
            "threadpool-common.h": "pthreadpool/src/threadpool-common.h",
            "threadpool-object.h": "pthreadpool/src/threadpool-object.h",
            "threadpool-utils.h": "pthreadpool/src/threadpool-utils.h",
        },
        header_namespace = "",
        preferred_linkage = "static",
        platform_preprocessor_flags = [["windows", ["-D_WINDOWS", "-D_WIN32", "-DWIN32", "-DNOMINMAX", "-D_CRT_SECURE_NO_WARNINGS", "-D_USE_MATH_DEFINES"]], ["windows.*64$", ["-D_WIN64"]]],
        preprocessor_flags = ["-DPTHREADPOOL_USE_FUTEX=0", "-DPTHREADPOOL_USE_GCD=0"],
        visibility = ["PUBLIC"],
    )

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "pthreadpool_header",
        header_namespace = "",
        exported_headers = {
            "pthreadpool.h": "pthreadpool/include/pthreadpool.h",
        },
        visibility = ["PUBLIC"],
    )
