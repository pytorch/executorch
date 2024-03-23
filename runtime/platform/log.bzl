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
        # On by default.
        return ["-DET_MIN_LOG_LEVEL=" + et_log_level()]
    else:
        return ["-DET_LOG_ENABLED=0"]
