load("@prelude//rules.bzl", "prebuilt_python_library", "remote_file")

def define_prebuilt_python_library(name, url, sha1, out, additional_deps = []):
    remote_file(
        name = "{}-download".format(name),
        url = url,
        sha1 = sha1,
        out = out,
    )

    prebuilt_python_library(
        name = name,
        binary_src = ":{}-download".format(name),
        visibility = ["PUBLIC"],
        deps = [":{}-download".format(name)] + additional_deps,
    )

def add_prebuilt_python_library_targets(targets):
    for name, config in targets.items():
        define_prebuilt_python_library(name, **config)
