def get_qnn_library_version():
    return read_config("executorch", "qnn_version", "2.51")
