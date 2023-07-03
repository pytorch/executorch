"""
Common macros used by the profiler go into this file.
"""

def profiling_enabled():
    return native.read_config("executorch", "prof_enabled", "false") == "true"

def get_profiling_flags():
    profiling_flags = []
    if profiling_enabled():
        profiling_flags += ["-DPROFILING_ENABLED"]
    prof_buf_size = native.read_config("executorch", "prof_buf_size", None)
    if prof_buf_size != None:
        if not profiling_enabled():
            fail("Cannot set profiling buffer size without enabling profiling first.")
        profiling_flags += ["-DMAX_PROFILE_EVENTS={}".format(prof_buf_size), "-DMAX_MEM_PROFILE_EVENTS={}".format(prof_buf_size)]
    num_prof_blocks = native.read_config("executorch", "num_prof_blocks", None)
    if num_prof_blocks != None:
        if not profiling_enabled():
            fail("Cannot configure number of profiling blocks without enabling profiling first.")
        profiling_flags += ["-DMAX_PROFILE_BLOCKS={}".format(num_prof_blocks)]
    return profiling_flags
