load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def cpu_provider(dependency, header, factory):
    """Describes a provider's public C++ factory and its runtime dependency."""
    return struct(dependency = dependency, header = header, factory = factory)

def cpu_backend(
        name,
        providers,
        kernel_libraries = None,
        preferences = (),
        force = False,
        visibility = None):
    """Links one CpuBackend with an immutable, application-composed provider list."""
    if not providers:
        fail("CpuBackend requires at least one provider")
    if force and not preferences:
        fail("Force requires at least one preference")
    preference_values = []
    for provider, implementation in preferences:
        if not provider:
            fail("Preferences require a provider name")
        preference_values.append("{" + json.encode(provider) + "," + json.encode(implementation) + "}")
    source = "#include <executorch/backends/cpu/runtime/KernelProvider.h>\n"
    for provider in providers:
        source += "#include <{}>\n".format(provider.header)
    source += "namespace executorch::backends::cpu {\nnamespace {\n"
    source += "RuntimeConfiguration runtime_configuration() {\n"
    source += "static const ProviderFactory factories[] = {" + ",".join([provider.factory for provider in providers]) + "};\n"
    source += "return {{factories, sizeof(factories) / sizeof(factories[0])}, {" + ",".join(preference_values) + "}, " + ("true" if force else "false") + "};\n}\n"
    source += "const bool registered = (register_runtime_configuration(runtime_configuration), true);\n}\n}\n"
    runtime.genrule(
        name = name + "_providers",
        outs = {"provider_registry.cpp": ["provider_registry.cpp"]},
        default_outs = ["."],
        cmd = "cat > \"$OUT/provider_registry.cpp\" <<'CPU_PROVIDERS'\n" + source + "CPU_PROVIDERS\n",
    )
    runtime.cxx_library(
        name = name,
        srcs = [":" + name + "_providers[provider_registry.cpp]"],
        exported_deps = [
            "//executorch/backends/cpu:cpu_engine",
        ] + [provider.dependency for provider in providers] + (kernel_libraries or []),
        visibility = visibility or ["PUBLIC"],
        # @lint-ignore BUCKLINT: Retain the generated provider composition.
        link_whole = True,
    )
