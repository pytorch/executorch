def build_sdk():
    return native.read_config("executorch", "build_sdk", "false") == "true"

def get_sdk_flags():
    sdk_flags = []
    if build_sdk():
        sdk_flags += ["-DEXECUTORCH_BUILD_DEVTOOLS"]
    return sdk_flags
