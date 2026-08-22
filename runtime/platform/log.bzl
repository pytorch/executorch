load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def et_logging_enabled():
    return native.read_config("executorch", "enable_et_log", "true") == "true"

def et_log_level():
    raw_level = native.read_config("executorch", "log_level", "Info").lower()
    if raw_level == "debug":
        return "Debug"
    elif raw_level == "info":
        return "Info"
    elif raw_level == "error":
        return "Error"
    elif raw_level == "fatal":
        return "Fatal"
    else:
        fail("Unknown log level '{}'. Expected one of 'Debug', 'Info', 'Error', or 'Fatal'.".format(raw_level))

def get_et_logging_flags():
    if et_logging_enabled():
        if runtime.is_oss:
            return ["-DET_MIN_LOG_LEVEL=" + et_log_level()]

        # On by default; allow opt-out via constraint (the executorch.enable_et_log
        # buckconfig above remains an independent way to disable logging).
        return select({
            "DEFAULT": ["-DET_MIN_LOG_LEVEL=" + et_log_level()],
            "fbsource//xplat/executorch/tools/buck/constraints:executorch-et-log-disabled": ["-DET_LOG_ENABLED=0"],
        })
    else:
        return ["-DET_LOG_ENABLED=0"]
