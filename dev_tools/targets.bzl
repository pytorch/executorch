def build_dev_tools():
    return native.read_config("executorch", "build_dev_tools", "false") == "true"

def get_dev_tools_flags():
    dev_tools_flags = []
    if build_dev_tools():
        dev_tools_flags += ["-DEXECUTORCH_BUILD_DEV_TOOLS"]
    return dev_tools_flags
