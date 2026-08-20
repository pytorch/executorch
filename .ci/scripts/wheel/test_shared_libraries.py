# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checks that the wheel ships its runtime as separate shared libraries.

The Python bindings extension used to contain the runtime, the registries, the
CPU kernels, the XNNPACK delegate and the profiler, all statically linked into
one file. It now links them as shared libraries the wheel ships alongside it.
These checks run against the installed wheel only; they never look at the source
tree's build directory, because a checkout on the module search path makes every
check below pass while inspecting the wrong thing.

The properties verified here are the ones the split exists to create:

1. Each component has exactly one definer, and it is the library that is meant
   to own it. Counting definers alone is not enough: the monolithic layout also
   had exactly one of each, inside the Python extension.
2. The extension contains none of those components and depends on every shipped
   library instead.
3. Every shipped library loads with no absolute runtime search path, including
   after being moved, so the wheel is relocatable rather than only working on
   the machine that built it.
"""

import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

# Registry entry points. A second definer of any of these means a second
# process-wide registry.
_REGISTRY_SYMBOLS = (
    "executorch::runtime::register_backend",
    "executorch::runtime::get_num_registered_backends",
    "executorch::runtime::get_backend_class",
)

# The thread pool accessor. A second definer means a second pool, which
# oversubscribes the CPU because each pool sizes itself to all cores.
_THREADPOOL_SYMBOLS = ("executorch::extension::threadpool::get_threadpool",)

# What the Python extension calls itself, so these have to appear in its dynamic
# symbol table as undefined. pybindings.cpp reaches each one with no build option
# in front of it, which makes absence from both tables a hidden private copy
# rather than a symbol the extension simply does not use. register_backend is not
# among them: the extension registers nothing, the delegate libraries do.
_EXTENSION_IMPORTS = (
    "executorch::runtime::get_num_registered_backends",
    "executorch::runtime::get_backend_class",
    *_THREADPOOL_SYMBOLS,
)

# A representative operator from the merged CPU kernels. A second definer means
# the operators are registered twice, which aborts at startup.
_KERNEL_SYMBOLS = ("torch::executor::native::abs_out",)

# The quantized kernels, whose own library the wheel ships when they are built.
# A separate group because they have a separate owner, and because a wheel built
# without them ships neither the library nor these symbols.
# The CUDA delegate and its stream helper, for a wheel built from a CUDA index. The
# stream helper matters most: two copies means two notions of the caller's stream, so
# work queued through one is invisible to the other.
# A strongly defined symbol, chosen by reading the built library rather than guessed. The
# CudaBackend methods are emitted weak, and a weak definition can be replaced at load time by a
# strong one elsewhere, so naming one of those would count a definition that may not be the one
# the process uses.
_CUDA_BACKEND_SYMBOLS = ("executorch::backends::cuda::load_library",)
_CUDA_STREAM_SYMBOLS = ("executorch::extension::cuda::getCallerStream",)

# The AOTI shim layer and the stream-guard state that lives with it. This is the state that was
# genuinely duplicated: extracting the shims with a PUBLIC whole-archive replayed the extraction at
# every consumer's link, so the guard's thread_local and these shims landed in three shipped binaries
# at once, and a stream selected through one copy was invisible to the other two. The row above cannot
# catch that, because its symbol only ever existed in one unconditionally shared library.
_AOTI_SHIM_SYMBOLS = (
    "aoti_torch_empty_strided",
    "aoti_torch_delete_tensor_object",
    "executorch::backends::cuda::CUDAStreamGuard::create",
    # From the CUDA sources rather than the C++ ones. The build drops every .cu file when no working
    # compiler is found, and the library is still produced from its .cpp sources, so a check that
    # names only C++ symbols passes on a library missing every kernel it was gated for. Every
    # kernel shim is listed rather than a sample, because a partially built library is exactly the
    # failure this row exists to catch.
    "aoti_torch_cuda__weight_int4pack_mm",
    "aoti_torch_cuda_int4_plain_mm",
    "aoti_torch_cuda_int5_plain_mm",
    "aoti_torch_cuda_int6_plain_mm",
    "aoti_torch_cuda_int8_plain_mm",
    "aoti_torch_cuda_rand",
    "aoti_torch_cuda_randint_low_out",
    "aoti_torch_cuda_sort_stable",
    "aoti_torch_cuda_guard_set_index",
)

_QUANTIZED_KERNEL_SYMBOLS = (
    "torch::executor::native::quantize_per_tensor_out",
    "torch::executor::native::dequantize_per_tensor_out",
)

# The registry entry points, kept separate from the kernel implementations above.
# A library that carries its own copy of these has its own registration code, which
# is what this split is meant to prevent: one owner of the operator table. Checking
# only a kernel implementation would miss that entirely.
_KERNEL_REGISTRY_SYMBOLS = (
    "executorch::runtime::register_kernels",
    "executorch::runtime::get_registered_kernels",
)

# A representative symbol from the XNNPACK delegate. A second definer means the
# process carries two copies of the delegate.
_XNNPACK_SYMBOLS = (
    "executorch::backends::xnnpack::XnnpackBackendOptions::workspace_manager",
)

# Third-party code the shipped libraries bundle rather than depend on. These are C
# symbols with default visibility, so a second copy in the same process is not a
# duplicate of ExecuTorch's own code but it is still two thread pools or two
# XNNPACK runtimes, and which one a caller reaches depends on load order.
#
# Checked separately from the wrapper symbols above because the wrappers can each
# have exactly one owner while the bundled code underneath them does not. That is
# the same failure the split exists to prevent, reached by a different route.
# pthreadpool is compiled with hidden visibility on Apple, deliberately, so that the
# copy inside libtorch_cpu cannot take precedence over the bundled one. Its symbols are
# present but not exported there, so only cpuinfo can serve as the sentinel on that
# platform. Both are checked elsewhere.
if sys.platform == "darwin":
    _BUNDLED_THREADPOOL_SYMBOLS = ("cpuinfo_initialize",)
else:
    _BUNDLED_THREADPOOL_SYMBOLS = ("pthreadpool_create", "cpuinfo_initialize")
# The delegate's own entry points. A second definer means the delegate is compiled
# into the Python extension as well, which would register it twice in one process.
_OPENVINO_BACKEND_SYMBOLS = ("executorch::backends::openvino::OpenvinoBackend",)

_BUNDLED_XNNPACK_SYMBOLS = ("xnn_create_runtime_v4",)

# A representative symbol from the profiler. A second definer means two event
# tracers, so a trace records only part of what ran.
_ETDUMP_SYMBOLS = ("executorch::etdump::ETDumpGen::ETDumpGen",)

# `nm -DC` prints "<hexaddr> <kind> <name>" for a definition and
# "                 U <name>" for an undefined reference.
_DEFINED = re.compile(r"^[0-9a-fA-F]+\s+(?P<kind>[A-Za-z])\s+(?P<name>.+)$")

# Thin and fat Mach-O headers, both byte orders, used to tell a file the Mach-O
# tools should have been able to read from one that merely carries their suffix.
_MACH_O_MAGIC = frozenset(
    {
        b"\xcf\xfa\xed\xfe",
        b"\xce\xfa\xed\xfe",
        b"\xfe\xed\xfa\xcf",
        b"\xfe\xed\xfa\xce",
        b"\xca\xfe\xba\xbe",
        b"\xbe\xba\xfe\xca",
    }
)

# Symbol kinds that mean the object owns the code or storage.
_OWNING_KINDS = frozenset("TtBbDdGgSsRrWV")


def _declared_requirements() -> set:
    """Import names the installed wheel declares a requirement for.

    Used to tell a package this environment simply does not have from one the wheel
    said it needed, because only the second is a packaging defect.

    Three sources, because none alone is sufficient. importlib's reverse map is
    authoritative where a distribution is INSTALLED, which is what makes
    PyYAML -> yaml, ruamel.yaml -> ruamel and hydra-core -> hydra resolve correctly.
    But it enumerates installed distributions only, so a declared dependency that is
    MISSING can never appear in it, which is precisely the case this function exists
    to catch. The transformed distribution name covers most of the rest.

    Some distributions import under a name no transformation produces, so those are
    listed. Measured against the wheel's own declared list: py-cpuinfo -> cpuinfo,
    PyYAML -> yaml and hydra-core -> hydra are all missed by the transformation, and
    each would turn a dependency the wheel failed to install into a quiet skip.
    """
    # Distribution name to import name, where the two are unrelated. Keyed on the
    # normalised distribution name so a change in case or separator still matches.
    unrelated_import_names = {
        "py_cpuinfo": ("cpuinfo",),
        "pyyaml": ("yaml", "_yaml"),
        "hydra_core": ("hydra",),
        "scikit_learn": ("sklearn",),
        "typing_extensions": ("typing_extensions",),
        "pillow": ("PIL",),
        "protobuf": ("google",),
        "opencv_python": ("cv2",),
    }
    try:
        from importlib.metadata import packages_distributions, requires
    except ImportError:  # pragma: no cover
        return set()
    try:
        declared = requires("executorch") or []
    except Exception:
        return set()

    wanted, names = set(), set()
    for requirement in declared:
        # Only the distribution name, dropping any version specifier, extra or
        # environment marker.
        name = re.split(r"[\s;\[<>=!~(]", requirement.strip(), maxsplit=1)[0]
        if not name:
            continue
        normalised = name.lower().replace("-", "_").replace(".", "_")
        wanted.add(normalised)
        # The likely import name, so a MISSING declared dependency is still
        # recognised as declared rather than silently skipped.
        names.add(name)
        names.add(name.replace("-", "_"))
        names.add(name.split(".")[0])
        names.update(unrelated_import_names.get(normalised, ()))

    for import_name, distributions in packages_distributions().items():
        for distribution in distributions:
            if distribution.lower().replace("-", "_").replace(".", "_") in wanted:
                names.add(import_name)
    return names


def _installed_package_dir() -> Path:
    """The installed executorch package, never the source checkout.

    Enforced rather than assumed. Python puts the working directory on the module
    search path, so running from a checkout resolves `executorch` to the source
    tree, where there are no shipped libraries and every check below passes while
    testing nothing. That is worse than a failure, because it looks like a pass.
    """
    import executorch

    paths = [Path(entry).resolve() for entry in executorch.__path__]
    # Every entry, not just the first. This is a namespace package, so a checkout on
    # the module search path adds a second entry, and a module can then resolve from
    # the checkout while the first entry still looks like a clean install.
    outside = [
        path
        for path in paths
        if "site-packages" not in path.parts and "dist-packages" not in path.parts
    ]
    assert not outside, (
        f"executorch also resolves through {outside}, which is not an installed "
        "package. Run this from a directory that contains no executorch checkout, "
        "or the checks silently inspect the source tree instead of the wheel."
    )
    assert len(paths) == 1, (
        f"executorch resolves through {len(paths)} paths ({paths}). Even when all of "
        "them are installs, a module could come from either, so which artifact is "
        "under test is ambiguous."
    )
    return paths[0]


def _tool(name: str):
    """Locate a build tool, including one pip installed beside this interpreter.

    `shutil.which` searches PATH only, and a virtual environment's `bin` is on PATH
    only when the environment is activated. These tests are normally run by invoking
    the interpreter directly, so a tool pip installed into that environment is present
    on disk and invisible to a PATH search.
    """
    found = shutil.which(name)
    if found:
        return found
    beside = Path(sys.executable).parent / name
    return str(beside) if beside.is_file() else None


def _loader_clean_environment() -> dict:
    """The environment with every loader override removed.

    A search path or an injected library inherited from the build environment
    supplies what a shipped library failed to record, so a child process started
    without stripping these can load a package that would not load anywhere else.
    Both spellings go on both platforms: the one that does not apply is absent
    rather than harmful, and naming only the ELF variable is what left the macOS
    checks honouring DYLD_LIBRARY_PATH.
    """
    overrides = (
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
    )
    return {key: value for key, value in os.environ.items() if key not in overrides}


def _dynamic_section(library) -> str | None:
    """What a library records about its dependencies and search paths.

    Returns None when no tool can read it, so a caller can tell "nothing recorded"
    apart from "could not look", which are different verdicts.

    readelf prints the ELF dynamic section. otool -l prints the Mach-O load commands,
    which carry the same facts under different names: LC_LOAD_DYLIB for a dependency
    where ELF has NEEDED, and LC_RPATH where ELF has RUNPATH.
    """
    if sys.platform == "darwin":
        tool, args = _tool("otool"), ["-l"]
    else:
        tool, args = _tool("readelf"), ["-d"]
    if tool is None:
        return None
    return subprocess.run(
        [tool, *args, str(library)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout


def _linked_libraries(library) -> str | None:
    """The libraries this one resolves at load time, as text to search.

    ldd resolves an ELF library's dependencies transitively. otool -L lists a Mach-O
    library's direct dependencies without resolving them, which is weaker, so a macOS
    result says which names are recorded rather than whether each one was found.
    """
    if sys.platform == "darwin":
        tool, args = _tool("otool"), ["-L"]
    else:
        tool, args = _tool("ldd"), ["-r"]
    if tool is None:
        return None
    result = subprocess.run(
        [tool, *args, str(library)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout + result.stderr


def _recorded_dependencies(library) -> set:
    """The library names this one records a dependency on, without resolving them.

    ELF names them in NEEDED entries of the dynamic section, already as bare file
    names. Mach-O names them in LC_LOAD_DYLIB commands, which otool -L prints as the
    install name, a path, so only its last component is comparable with the ELF answer.

    The tool is required rather than optional, because a caller cannot tell an empty
    result apart from one this could not read, and the second is a passing check that
    examined nothing.
    """
    if sys.platform == "darwin":
        tool = _tool("otool")
        assert tool is not None, "otool is required to inspect the wheel"
        # The first line names the file itself, and each later line is one dependency
        # followed by the version information otool appends in parentheses.
        return {
            Path(line.strip().split(" (", 1)[0]).name
            for line in subprocess.run(
                [tool, "-L", str(library)], capture_output=True, text=True, check=True
            ).stdout.splitlines()[1:]
            if line.strip()
        }
    tool = _tool("readelf")
    assert tool is not None, "readelf is required to inspect the wheel"
    return {
        line.split("[", 1)[1].rstrip("]").strip()
        for line in subprocess.run(
            [tool, "-d", str(library)], capture_output=True, text=True, check=True
        ).stdout.splitlines()
        if "NEEDED" in line
    }


def _raw_recorded_identity(library) -> str | None:
    """The identity exactly as recorded, without reducing it to a basename.

    Separate from _recorded_identity because that function's basename reduction is correct for
    comparing names but hides whether the recorded string is absolute, which is a different fault.
    """
    section = _dynamic_section(library)
    if section is None:
        return None
    # otool -l prints a name field for LC_LOAD_DYLIB as well as LC_ID_DYLIB, so the current load
    # command has to be tracked: taking the first name line returned a dependency such as
    # /usr/lib/libc++.1.dylib for a library that has no install name at all, which would report an
    # absolute install name where the truth is that there is none.
    in_id_command = False
    for line in section.splitlines():
        stripped = line.strip()
        if sys.platform == "darwin":
            if stripped.startswith("cmd "):
                # Containment, matching how every other otool -l parser in this file reads a load
                # command. Exact equality would return None for every library if any otool spelled
                # the line differently, and the absolute check would then inspect nothing and pass.
                in_id_command = "LC_ID_DYLIB" in stripped
            elif in_id_command and stripped.startswith("name "):
                return stripped.split(" (offset", 1)[0][len("name ") :]
        elif "SONAME" in stripped and "[" in stripped:
            return stripped.split("[", 1)[1].rstrip("]")
    return None


def _recorded_identity(library) -> str | None:
    """The name this library tells a consumer to record when linking against it.

    ELF calls it the soname. Mach-O calls it the install name and spells it as a path,
    usually relative to the consumer's runtime search path, so the two compare only by
    the last component.
    """
    if sys.platform == "darwin":
        tool = _tool("otool")
        assert tool is not None, "otool is required to inspect the wheel"
        # otool -D prints the file name it was given, then the install name if the
        # library has one. A library without one prints only the first line.
        lines = [
            line.strip()
            for line in subprocess.run(
                [tool, "-D", str(library)], capture_output=True, text=True, check=True
            ).stdout.splitlines()[1:]
            if line.strip()
        ]
        return Path(lines[0]).name if lines else None
    tool = _tool("readelf")
    assert tool is not None, "readelf is required to inspect the wheel"
    return next(
        (
            line.split("[", 1)[1].rstrip("]").strip()
            for line in subprocess.run(
                [tool, "-d", str(library)], capture_output=True, text=True, check=False
            ).stdout.splitlines()
            if "SONAME" in line
        ),
        None,
    )


def _runtime_search_paths(library) -> list | None:
    """The runtime search path entries recorded in a shipped library.

    Returns None when the file cannot be read, so a caller can tell "records nothing"
    apart from "could not look", which are different verdicts.

    patchelf prints an ELF RPATH as one colon separated string. Mach-O keeps each entry
    in its own LC_RPATH load command, so otool is parsed for those instead. patchelf
    cannot read Mach-O at all, which is why it is not simply reused here.
    """
    if sys.platform == "darwin":
        tool = _tool("otool")
        if tool is None:
            return None
        result = subprocess.run(
            [tool, "-l", str(library)], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            return None
        entries = []
        lines = result.stdout.splitlines()
        for index, line in enumerate(lines):
            if "LC_RPATH" not in line:
                continue
            # The path sits a couple of lines below its command, followed by the
            # offset otool appends, which is not part of the value.
            for following in lines[index + 1 : index + 4]:
                stripped = following.strip()
                if stripped.startswith("path "):
                    entries.append(stripped.split(" (offset", 1)[0][len("path ") :])
                    break
        return entries
    tool = _tool("patchelf")
    if tool is None:
        return None
    result = subprocess.run(
        [tool, "--print-rpath", str(library)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    # A library with no search path prints nothing, which is not the same as a search path holding
    # an empty entry. The fields are kept rather than filtered, because the loader reads an empty
    # entry as the process working directory and the caller rejects exactly that.
    recorded = result.stdout.strip()
    return recorded.split(":") if recorded else []


def _assert_mach_o_architecture_matches(wheel: Path) -> None:
    """Fail if a macOS wheel's declared architecture is not what its binaries contain.

    auditwheel answers this on Linux by classifying against manylinux, which has no macOS
    equivalent, so lipo is asked directly instead. It reports the architectures present in
    a Mach-O file, and every shipped binary has to be one the tag promises.

    A universal binary lists several architectures, so containing the declared one is the
    test rather than equalling it.
    """
    lipo = _tool("lipo")
    assert lipo is not None, (
        "lipo is required to check that a macOS wheel's contents match the architecture "
        "it claims, and it was not found"
    )
    claimed = wheel.name.split("-")[-1].removesuffix(".whl")
    # macosx_14_0_arm64 and macosx_11_0_x86_64 both end in the architecture.
    declared = claimed.split("_")[-1]
    if declared == "64" and claimed.endswith("x86_64"):
        declared = "x86_64"

    with tempfile.TemporaryDirectory() as unpacked:
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(unpacked)
        root = Path(unpacked)
        binaries = [
            path
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and not path.is_symlink()
            and path.suffix in (".dylib", ".so")
        ]
        assert binaries, (
            f"the wheel {wheel.name} contains no binaries, so the architecture it claims "
            "cannot be checked against anything"
        )
        mismatched = []
        unreadable = []
        for binary in binaries:
            result = subprocess.run(
                [lipo, "-archs", str(binary)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                # Skipping every unreadable file would let a wheel whose binaries
                # lipo cannot parse pass having inspected none of them. Something
                # merely named .so that is not a Mach-O file is not this check's
                # concern; the magic bytes tell the two apart without depending on
                # how lipo words its refusal.
                with binary.open("rb") as handle:
                    header = handle.read(4)
                if header in _MACH_O_MAGIC:
                    unreadable.append(
                        f"{binary.relative_to(root)}: {result.stderr.strip()[:120]}"
                    )
                continue
            present = result.stdout.split()
            if declared not in present:
                mismatched.append(f"{binary.relative_to(root)} is {' '.join(present)}")
    assert not unreadable, (
        f"lipo could not read these Mach-O files in {wheel.name}, so the architecture "
        f"check covered less than the wheel ships: {unreadable}"
    )
    assert not mismatched, (
        f"the wheel claims architecture {declared} but these binaries are built for "
        f"something else, so it would install where it cannot run: {mismatched}"
    )
    print(f"\u2713 the wheel is tagged for the architecture it contains ({declared})")


def _shipped_object_patterns() -> list[str]:
    """Filename patterns for every shipped loadable object.

    Both suffixes are listed because a Mach-O Python extension is a .so by convention, so a
    dylib-only pattern silently drops the extension on macOS, which is the one artifact these
    checks exist to verify.
    """
    return ["*.dylib*", "*.so*"]


def _dynamic_lib_suffix() -> str:
    """The loadable library suffix on this platform, including the dot."""
    return ".dylib" if sys.platform == "darwin" else ".so"


def _library_file_name(base_name: str) -> str:
    """The file name a component's library has on this platform.

    The component table names libraries without a suffix so one table serves both
    platforms.
    """
    return f"{base_name}{_dynamic_lib_suffix()}"


def _nm_defined_args():
    """The nm flags that list what a library defines.

    GNU nm reads the dynamic symbol table with -D. Mach-O has no separate dynamic
    symbol table, so that flag fails outright there and -gU, global and defined, is
    the equivalent question.
    """
    return ["-gU", "-C"] if sys.platform == "darwin" else ["-DC"]


def _nm_undefined_args():
    """The nm flags that list what a library needs from elsewhere."""
    if sys.platform == "darwin":
        return ["-gu", "-C"]
    return ["-DC", "--undefined-only"]


def _shipped_shared_objects(package_dir: Path):
    """Every shared object the wheel installed.

    Asserts it found some, because a wheel that installed none would let every check that walks this list
    report success having examined nothing.
    """
    found = [
        path
        for path in sorted(
            item
            for pattern in _shipped_object_patterns()
            for item in package_dir.rglob(pattern)
        )
        if path.is_file() and not path.is_symlink()
    ]
    assert (
        found
    ), f"no shared objects found under {package_dir}, so nothing below would be checking them"
    return found


def _shipped_runtime_libraries(package_dir: Path):
    """The libraries the wheel ships under lib/, whatever it names them.

    One place, because three checks previously spelled the pattern themselves as
    `*.so.*` and every one of them silently stopped matching when the build moved to
    unversioned names. The failure was invisible: a check that finds nothing reports
    that the component is absent, which each of those treats as acceptable.

    Matches a versioned name too, so a build that does set SOVERSION is still found.
    """
    lib_dir = package_dir / "lib"
    if not lib_dir.is_dir():
        return []
    return [
        path
        for path in sorted(lib_dir.glob(f"lib*{_dynamic_lib_suffix()}*"))
        if path.is_file() and not path.is_symlink()
    ]


def _defines_symbol(library: Path, symbol: str) -> bool:
    """Whether `library` owns a definition of `symbol`, read from the dynamic table.

    Limited to exported definitions on purpose, because that is all the shipped
    artifacts carry: every library the wheel ships is stripped, so the static symbol
    table `nm -C` would read is gone. Measured on a real wheel, `nm -C` finds zero of
    these sentinels while `nm -DC` finds them, so widening the reader would turn this
    check off rather than strengthen it.

    A duplicate hidden behind non-default visibility would therefore not be seen here.
    That is a real gap in this check: it reads what a library exports, so a second copy
    compiled with hidden visibility is invisible to it. Catching that needs a running
    process, which counts what actually registered rather than what is visible.
    """
    result = subprocess.run(
        [_tool("nm"), *_nm_defined_args(), str(library)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        # A file that is not an object file at all is not this check's concern: something whose
        # name merely ends in .so must not abort the run. A shipped library the reader cannot
        # parse is different, because a real definition could be hiding inside it, and reporting
        # "defines nothing" would let a duplicate pass. The ELF magic bytes tell them apart
        # without depending on the reader's wording.
        with library.open("rb") as handle:
            is_object_file = handle.read(4) == b"\x7fELF"
        assert not is_object_file, (
            f"nm could not read {library.name}, which is a shipped object file, so the symbol "
            f"checks cannot be trusted: {result.stderr.strip()[:200]}"
        )
        return False
    # Mach-O prefixes a C symbol with an underscore, so nm prints _cpuinfo_initialize
    # where ELF prints cpuinfo_initialize. A C++ name demangles to the same text on both,
    # so accepting the prefix is enough and no per-symbol spelling is needed.
    accepted = (symbol, f"_{symbol}") if sys.platform == "darwin" else (symbol,)
    # Matched whole rather than by prefix. A prefix match also accepts a longer
    # symbol that merely begins with this one, and reports a second definer of
    # something no library defines twice. Three suffixes may follow a complete
    # name: nm prints a demangled C++ definition as name(args), a symbol table
    # carrying versions prints name@@version, and a sentinel that names a class
    # appears only as one of its members, name::member, because a class has no
    # symbol of its own.
    for line in result.stdout.splitlines():
        if symbol not in line:
            continue
        match = _DEFINED.match(line)
        if not match or match.group("kind") not in _OWNING_KINDS:
            continue
        name = match.group("name")
        if any(
            name == spelling
            or name.startswith((f"{spelling}(", f"{spelling}@", f"{spelling}::"))
            for spelling in accepted
        ):
            return True
    return False


def _is_export_only(library: Path) -> bool:
    """Whether a library exists to export a model rather than to run one.

    The ahead-of-time operator libraries register kernels into torch so a model can be
    exported, and they link torch to do it. They deliberately carry their own copy of
    the kernels, because the copy a C++ application links is registered into a table
    those libraries never read.

    Excluded from the single-owner check for the one component that genuinely has two
    copies. Counting them there would report a duplicate for something that is not one,
    and the plugin does resolve its registrar from the shipped library now, while the code
    generator still compiles the kernel bodies into it for the dispatcher half.

    Recognised by linking torch, which is the property that makes a library export-side.
    Python extensions link torch too and are not export-side operator libraries, so they
    are excluded by their interpreter suffix rather than by an operator-library name,
    which keeps the test's verdict the same whether or not readelf is installed.
    """
    if ".cpython-" in library.name or library.name.endswith(".pyd"):
        return False
    if library.name.endswith(f"_aot_lib{_dynamic_lib_suffix()}"):
        return True
    dynamic = _dynamic_section(library)
    if dynamic is None:
        return False
    return _library_file_name("libtorch") in dynamic


def _assert_single_definer(
    symbols, what: str, owner: str | None = None, allow_export_copy: bool = False
) -> None:
    """At most one shipped library may define each of `symbols`.

    The owner is named where one is expected, because counting definers alone does
    not prove the split happened: the monolithic layout has exactly one definer too,
    the Python extension. Requiring the symbol to live in the library that is
    supposed to own it is what distinguishes the two.

    A component the wheel does not ship at all is a valid configuration, not a
    fault. Delegates and kernel sets are build options, so a wheel built without one
    has zero definers and is reported as such. What must never happen is two.

    `allow_export_copy` excuses the export-side libraries for one component only. The
    export plugin no longer carries its own registrar, but the code generator still
    compiles the kernels into it for the dispatcher half, so the symbols appear twice:
    once in the runtime library and once in the library torch loads at export time,
    because each side registers into a table the other never reads. Loading both used to
    abort on the second registration, so what this check enforces for that component is
    one owner among the runtime libraries, not the absence of the export copy. Excusing
    every component would disarm the check where duplication is a real fault: two of
    these libraries defined the backend registry symbols in one released wheel and not
    in the release before it, so the duplication this catches does happen.
    """
    assert _tool("nm") is not None, "nm is required to inspect the wheel"

    package_dir = _installed_package_dir()
    libraries = [
        library
        for library in _shipped_shared_objects(package_dir)
        if not (allow_export_copy and _is_export_only(library))
    ]
    assert libraries, f"no shared libraries found under {package_dir}"

    # Every symbol is resolved before anything is reported, so a component that is only
    # half present is described as such rather than looking like one that is absent.
    found = {
        symbol: [lib for lib in libraries if _defines_symbol(lib, symbol)]
        for symbol in symbols
    }
    # A component is either wholly present or wholly absent. Some symbols defined and
    # others not means a partial build, which is neither of those and is a fault.
    present = {symbol for symbol, definers in found.items() if definers}
    if not present:
        # When the caller has already established that the owner library ships, finding none of its
        # symbols is a fault rather than an absence. Returning success here made this a no-op the moment
        # a sentinel symbol was renamed or inlined, which turns the ownership check off without anyone
        # noticing, and one of these sentinels is a two-line accessor.
        assert owner is None, (
            f"the wheel ships {owner}, which owns the {what}, but none of its symbols "
            f"{sorted(found)} are defined anywhere. Either the sentinel symbols were renamed or "
            "inlined, in which case this check needs updating, or the library is empty."
        )
        print(f"- this wheel ships no {what}, nothing to check")
        return
    assert len(present) == len(found), (
        f"the wheel defines only part of the {what}: {sorted(present)} are present "
        f"and {sorted(set(found) - present)} are not, so it is neither shipped nor "
        "absent"
    )

    for symbol, definers in found.items():
        pretty = [str(lib.relative_to(package_dir)) for lib in definers]
        assert len(definers) == 1, (
            f"expected at most one library to define {symbol}, found "
            f"{len(definers)}: {pretty}. More than one definition means the "
            f"process has more than one {what}."
        )
        if owner is not None:
            assert definers[0].name.startswith(owner), (
                f"{symbol} is defined by {pretty[0]}, but it belongs in {owner}. One "
                "definer is not enough on its own: the monolithic layout this change "
                "replaces also had exactly one, inside the Python extension."
            )
    where = f" owned by {owner}" if owner else ""
    print(f"✓ single {what}{where} across {len(libraries)} shipped libraries")


def _wheel_cuda_train() -> str:
    """The CUDA train the installed wheel was built for, or "" for a CPU wheel.

    Read from the local version segment, which is the only place the wheel states what
    it was built for. `1.5.0+cu126` gives "126".
    """
    local = importlib.metadata.version("executorch").partition("+")[2]
    return local[2:] if local.startswith("cu") else ""


# Marker for a row whose owner is required only when the wheel is a CUDA wheel. A
# sentinel rather than a boolean, because the answer is not known until the installed
# wheel is inspected, and a row cannot call that at import time.
_REQUIRED_ON_A_CUDA_WHEEL = "cuda-wheel-only"

# Marker for a row whose owner every Linux wheel carries and no macOS wheel does.
_REQUIRED_ON_LINUX = "linux-only"


def _resolve_required(required):
    """Turn a row's requirement marker into the answer for the installed wheel."""
    if required == _REQUIRED_ON_A_CUDA_WHEEL:
        return bool(_wheel_cuda_train())
    if required == _REQUIRED_ON_LINUX:
        return sys.platform == "linux"
    return required


# The exact dependency names packaging declares per CUDA train, mirroring
# _CUDA_RUNTIME_PACKAGES in setup.py. Listed here rather than imported because setup.py
# runs a build when imported, and duplicated deliberately so a rename on the packaging
# side has to be made here too rather than silently agreeing with itself.
_EXPECTED_CUDA_PACKAGES = {
    "12": ("nvidia-cuda-runtime-cu12>=12,<13",),
    "13": ("nvidia-cuda-runtime>=13,<14",),
}


# Each component the wheel ships as its own library, the symbols that identify it,
# and the library that must own them. `required` says whether the owner has to be
# present: the optimized kernels are optional, because a wheel built without them
# deliberately links the portable ops into the Python extension instead, which is a
# supported configuration rather than a duplicate.
#
# A table rather than one function per component, because the per-function form let
# one of them drift: it looked up its library with its own glob, which silently
# stopped matching when the libraries were renamed while the others kept working.
_OWNED_COMPONENTS = (
    ("backend registry", _REGISTRY_SYMBOLS, _library_file_name("libexecutorch"), True),
    # The platform layer, which two shipped libraries each carried their own copy of, so a
    # register_pal call through one did not reach the other. Listed here so the ownership check
    # that already exists catches a regression rather than a later reader discovering it.
    (
        "platform layer",
        (
            # The strong accessors, not the emit hook next to them. That hook is a weak
            # default so a program supplying none still links, and a weak definition cannot
            # express ownership. A second copy of these two is a genuinely split platform
            # layer.
            "executorch::runtime::register_pal",
            "executorch::runtime::get_pal_impl",
        ),
        _library_file_name("libexecutorch"),
        True,
    ),
    (
        "operator registry",
        _KERNEL_REGISTRY_SYMBOLS,
        _library_file_name("libexecutorch"),
        True,
    ),
    (
        "thread pool",
        _THREADPOOL_SYMBOLS,
        _library_file_name("libexecutorch_threadpool"),
        True,
    ),
    ("profiler", _ETDUMP_SYMBOLS, _library_file_name("libexecutorch_etdump"), True),
    (
        "XNNPACK delegate",
        _XNNPACK_SYMBOLS,
        _library_file_name("libexecutorch_backend_xnnpack"),
        True,
    ),
    (
        "set of CPU kernels",
        _KERNEL_SYMBOLS,
        _library_file_name("libexecutorch_kernels_optimized"),
        False,
    ),
    (
        "set of quantized kernels",
        _QUANTIZED_KERNEL_SYMBOLS,
        _library_file_name("libexecutorch_kernels_quantized"),
        True,
    ),
    # The CUDA components. Required exactly when the wheel says it is a CUDA wheel,
    # which is decided at check time rather than here: a fixed False meant a wheel
    # tagged +cu126 carrying no CUDA library at all passed every check in this file,
    # while a fixed True would fail every CPU wheel. The marker is the string these
    # rows are keyed on below.
    (
        "CUDA delegate",
        _CUDA_BACKEND_SYMBOLS,
        _library_file_name("libexecutorch_backend_cuda"),
        _REQUIRED_ON_A_CUDA_WHEEL,
    ),
    (
        "CUDA stream helper",
        _CUDA_STREAM_SYMBOLS,
        _library_file_name("libexecutorch_extension_cuda"),
        _REQUIRED_ON_A_CUDA_WHEEL,
    ),
    (
        "AOTI shim layer",
        _AOTI_SHIM_SYMBOLS,
        _library_file_name("libaoti_cuda_shims"),
        _REQUIRED_ON_A_CUDA_WHEEL,
    ),
    # The third-party code these libraries bundle, checked separately from the
    # wrappers above. A wrapper can have a single owner while the implementation
    # underneath it is bundled into two of these, which is two real thread pools or
    # two XNNPACK runtimes.
    #
    # One copy among the libraries this wheel ships, which is what this change
    # controls. torch links the same projects and exports the same symbols, so the
    # process still holds two definitions and which one a caller reaches depends on
    # load order. Fixing that needs an explicit export list: hiding them wholesale
    # with --exclude-libs,ALL breaks aarch64, where the optimized kernels resolve
    # cpuinfo_initialize from the thread pool across a library boundary.
    (
        "bundled thread pool implementation",
        _BUNDLED_THREADPOOL_SYMBOLS,
        _library_file_name("libexecutorch_threadpool"),
        True,
    ),
    (
        "bundled XNNPACK runtime",
        _BUNDLED_XNNPACK_SYMBOLS,
        _library_file_name("libexecutorch_backend_xnnpack"),
        True,
    ),
    # Required on Linux, where packaging turns the backend on for every non-minimal
    # build. A fixed False passed a wheel that had compiled the delegate back into the
    # extension: no library ships, so the row skips, and one definer inside the
    # extension is exactly the monolithic layout a definer count cannot distinguish.
    (
        "OpenVINO delegate",
        _OPENVINO_BACKEND_SYMBOLS,
        _library_file_name("libexecutorch_backend_openvino"),
        _REQUIRED_ON_LINUX,
    ),
)

# The one component that legitimately exists twice. The quantized kernels are compiled into the runtime
# library and again into the library torch loads at export time, because each side registers into a
# table the other never reads, so a second definer there is expected rather than a fault. A process
# that loads both used to abort on the second registration, which is why this is named per component and
# the check stays armed for every other component, where a second definer means two registries or two thread
# pools in one process.
_COMPONENTS_WITH_AN_EXPORT_COPY = frozenset({"set of quantized kernels"})


def test_each_component_has_one_owner() -> None:
    """No component may be defined by more than one library the wheel ships.

    This is the property the split exists to create. Two copies of a component mean
    two registries or two thread pools in one process, and a static initializer that
    registers into a table nothing else reads shows up as an operator missing at run
    time rather than as a link error.
    """
    # Every shipped shared object, not only the ones under lib/. One owner, libaoti_cuda_shims.so,
    # ships under backends/cuda/, and scanning lib/ alone reported it as absent, which each row
    # treats as an acceptable state and so would have skipped the check entirely.
    shipped = {path.name for path in _shipped_shared_objects(_installed_package_dir())}
    for what, symbols, owner, required in _OWNED_COMPONENTS:
        # Resolved here rather than in the table, because it depends on the installed
        # wheel. A fixed False let a wheel tagged +cu126 ship with no CUDA library at
        # all and still pass, which is the whole point of the conditional rows.
        required = _resolve_required(required)
        present = any(name.startswith(owner) for name in shipped)
        assert present or not required, (
            f"the wheel ships no {owner}, which owns the {what}. Either packaging "
            "dropped it or the build did not produce it."
        )
        _assert_single_definer(
            symbols,
            what,
            owner if present else None,
            allow_export_copy=what in _COMPONENTS_WITH_AN_EXPORT_COPY,
        )


def test_python_extensions_import() -> None:
    """Every shipped Python extension must import from a clean environment.

    The symbol and dependency checks work on the files. This covers the other
    half: an extension can be packaged correctly and still fail to load because a
    runtime path does not reach one of its dependencies. Run in a subprocess with
    the loader overrides removed so a value from the build environment cannot
    supply a path the shipped library is missing.

    The list is discovered from the installed package rather than written here, so
    an extension added later is covered without anyone remembering to add it. A
    hardcoded list is how the ones this change relinked went untested.
    """
    package_dir = _installed_package_dir()
    modules = []
    for extension in sorted(package_dir.rglob("*.so")):
        # Only Python extensions, which carry the interpreter's suffix. The plain
        # shared libraries under lib/ are checked by the load test instead.
        if ".cpython-" not in extension.name:
            continue
        relative = extension.relative_to(package_dir).parent
        module = extension.name.split(".", 1)[0]
        modules.append(".".join(["executorch", *relative.parts, module]))
    assert modules, "the wheel ships no Python extension, which cannot be right"

    # Torch has to be installed, the same as for the dependency check: these
    # extensions link it, so without it they cannot import for a reason that says
    # nothing about packaging.
    if importlib.util.find_spec("torch") is None:
        print("- torch is not installed, skipping the extension import check")
        return
    environment = _loader_clean_environment()
    for module in modules:
        result = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        if result.returncode == 0:
            print(f"✓ {module} imports from a clean environment")
            continue
        # A Python dependency that is simply not installed here, including torch,
        # says nothing about how the wheel was built. Only a failure to load a
        # native library does.
        # A Python package this environment simply does not have says nothing about
        # how the wheel was built, PROVIDED the wheel does not claim to require it.
        # Match only that shape, so a native load failure reported as
        # ModuleNotFoundError is still caught below.
        missing_python_package = re.search(
            r"ModuleNotFoundError: No module named '(?!executorch)([\w.]+)",
            result.stderr,
        )
        if missing_python_package:
            absent = missing_python_package.group(1).split(".")[0]
            # A package the extension needs must appear in the wheel's declared
            # requirements. Without this the test was silent when a required package
            # was omitted from Requires-Dist: the environment did not have it, so
            # the import failed, and the previous rule skipped anything not already
            # declared. That treats a missing declaration as coverage rather than
            # as the bug it is.
            assert absent in _declared_requirements(), (
                f"{module} cannot import because {absent} is missing, and the wheel "
                "does not declare it. Add it to install_requires, or the extension "
                "silently needs a package a user is not asked to install."
            )
            print(f"- {module} needs {absent}, which this environment lacks, skipping")
            continue
        # Anything else is a real failure to load what the wheel ships: a missing
        # native library, an unresolved symbol, or an ABI mismatch.
        raise AssertionError(
            f"{module} ships in the wheel but does not import: "
            f"{result.stderr.strip()[-500:]}"
        )


_CUSTOM_OP_SOURCE = """\
// A custom operator, built the way an out-of-tree project builds one: against the
// shipped Python extension rather than an ExecuTorch source tree.
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace {

executorch::aten::Tensor& custom_double_out(
    executorch::runtime::KernelRuntimeContext& context,
    const executorch::aten::Tensor& input,
    executorch::aten::Tensor& out) {
  (void)context;
  const float* in = input.const_data_ptr<float>();
  float* dst = out.mutable_data_ptr<float>();
  for (ssize_t i = 0; i < input.numel(); ++i) {
    dst[i] = in[i] * 2.0f;
  }
  return out;
}

} // namespace

// The registration macro is the point of the check: it has to compile and resolve
// against the registry the shipped extension provides.
EXECUTORCH_LIBRARY(wheel_check, "custom_double.out", custom_double_out);
"""


_CUSTOM_OP_CMAKE = """\
cmake_minimum_required(VERSION 3.24)
project(custom_op_check CXX)

find_package(executorch REQUIRED)

add_library(custom_op_check SHARED custom_op.cpp)
# The legacy contract: a custom-op library links the shipped Python extension,
# which owns the operator registry it registers into.
target_link_libraries(custom_op_check PRIVATE _portable_lib)
# The runtime headers include c10 headers, which belong to torch rather than to
# this wheel, so an out-of-tree operator project supplies them the same way it
# supplies torch itself. The package config does not and should not ship them.
#
# The include directory is passed in rather than found with find_package(Torch),
# because that enables the CUDA language and fails on a machine with a CUDA
# toolkit it cannot probe, which has nothing to do with compiling an operator.
target_include_directories(custom_op_check PRIVATE ${TORCH_INCLUDE_DIR})
# Deliberately no target_compile_features here. These headers need C++20, and the
# package config is what has to say so. Setting it here would compile the check
# correctly while leaving a real consumer to fail.
"""


# Libraries that belong to torch rather than to this wheel. A library here resolves when the
# Python package that owns it is imported, so it is not something this wheel can or should ship.
_TORCH_LIBRARY_PREFIXES = (
    "libtorch",
    "libc10",
    "libshm",
    "libgomp",
    "libcudnn",
    "libcublas",
    # Torch declares nvrtc in its own dependency list next to cublas, and the CUDA 12
    # packages put each library in its own directory, so the hop that reaches cudart
    # does not reach this one. It resolves the same way the rest of this list does,
    # once the package that owns it is imported.
    "libnvrtc",
)


def _is_torch_library(name: str) -> bool:
    return name.startswith(_TORCH_LIBRARY_PREFIXES)


def _dyld_load_failure(library: Path, *, with_torch: bool) -> str:
    """Load `library` in a fresh interpreter and return dyld's message, or "".

    macOS has no ldd, and otool reports what a library asks for rather than
    whether the loader can find it, so the load itself has to be the check.
    RTLD_NOW binds every symbol, which is what makes this the counterpart of
    `ldd -r` rather than of plain `ldd`.

    A separate process each time, because dyld satisfies a request by install
    name from what the process has already loaded. Loading in this one would let
    an earlier library stand in for a later library's dependency, and in the
    relocated check it would hand back the original file instead of the copy
    under test, which is the failure that check exists to find.
    """
    prologue = "import torch;" if with_torch else ""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"{prologue}import ctypes,os,sys;ctypes.CDLL(sys.argv[1],mode=os.RTLD_NOW)",
            str(library),
        ],
        capture_output=True,
        text=True,
        check=False,
        # Any loader override in the build environment would paper over a runtime
        # search path the shipped library is actually missing.
        env=_loader_clean_environment(),
    )
    if result.returncode == 0:
        return ""
    return (result.stdout + result.stderr).strip()


def _dyld_missing_names(message: str) -> list[str]:
    """The dependency file names dyld reported it could not load."""
    return [
        Path(entry).name for entry in re.findall(r"Library not loaded: (\S+)", message)
    ]


def _assert_shipped_libraries_load_with_dyld() -> None:
    """The macOS half of the load check, split out only because the tool differs.

    The classification is the Linux one: a dependency the wheel ships must resolve
    here, a dependency it does not ship is the environment's to provide, and an
    unresolved symbol is a library that would fail at first use rather than at
    load.
    """
    package_dir = _installed_package_dir()
    libraries = _shipped_shared_objects(package_dir)
    shipped = {library.name for library in libraries}

    broken = {}
    unreachable = {}
    unresolved = {}
    for library in libraries:
        message = _dyld_load_failure(library, with_torch=True)
        if not message:
            continue
        key = str(library.relative_to(package_dir))
        missing = _dyld_missing_names(message)
        present_but_unreachable = [name for name in missing if name in shipped]
        absent = [
            name
            for name in missing
            if name not in shipped and not _is_torch_library(name)
        ]
        # Interpreter symbols are excluded for the same reason as on Linux: a
        # library Python loads resolves them from the running interpreter, and
        # they say nothing about how the wheel is packaged.
        symbols = [
            symbol
            for symbol in re.findall(r"Symbol not found: (\S+)", message)
            if not re.match(r"_?_?Py", symbol)
        ]
        if present_but_unreachable:
            unreachable[key] = present_but_unreachable
        if absent:
            broken[key] = absent
        if symbols:
            unresolved[key] = symbols[:5]
        # A refusal that names neither a library nor a symbol is still a refusal,
        # and dropping it would turn this check off for whatever caused it.
        if not (present_but_unreachable or absent or symbols):
            broken[key] = [message[:200]]

    assert not broken, (
        "shipped libraries need dependencies that nothing provides, so they will "
        f"fail to load: {broken}"
    )
    assert not unreachable, (
        "shipped libraries need dependencies the wheel ships but the loader "
        "cannot reach from them, which usually means a missing rpath entry: "
        f"{unreachable}"
    )
    assert not unresolved, (
        "shipped libraries reference symbols nothing provides, so they will fail "
        f"at first use rather than at load: {unresolved}"
    )
    print("✓ every shipped library loads in an environment with torch present")


def _macho_rpaths(library: Path) -> list[str]:
    """The LC_RPATH entries a Mach-O file carries, in load command order."""
    listing = subprocess.run(
        [_tool("otool"), "-l", str(library)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    paths = []
    in_rpath = False
    for line in listing.splitlines():
        stripped = line.strip()
        if stripped.startswith("cmd "):
            in_rpath = stripped == "cmd LC_RPATH"
        elif in_rpath and stripped.startswith("path "):
            paths.append(stripped[len("path ") :].rsplit(" (offset", 1)[0])
            in_rpath = False
    return paths


def _assert_shipped_libraries_relocate_with_dyld() -> None:
    """The macOS half of the relocated load check.

    Same shape as the Linux one: mirror the layout somewhere else, take away
    every absolute runtime search path, and see whether what is left still finds
    the libraries the wheel ships.
    """
    # Fatal, not a skip: both come with the developer tools on the only platform that
    # reaches here, so going quiet would report success having examined nothing, which
    # is the failure this check was ported to macOS to end.
    assert _tool("otool") is not None and _tool("install_name_tool") is not None, (
        "otool and install_name_tool are required to relocate a Mach-O file and read "
        "back its runtime search paths"
    )

    package_dir = _installed_package_dir()
    libraries = _shipped_shared_objects(package_dir)
    shipped = {library.name for library in libraries}

    with tempfile.TemporaryDirectory() as work_dir:
        root = Path(work_dir) / package_dir.name
        # Mirror the layout so a relative path such as @loader_path/../../lib
        # still points where it would in a real install.
        for library in libraries:
            target = root / library.relative_to(package_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(library, target)

        broken = {}
        for library in libraries:
            name = str(library.relative_to(package_dir))
            target = root / library.relative_to(package_dir)
            for entry in _macho_rpaths(target):
                # @loader_path and @executable_path are the relative forms this
                # check exists to prove sufficient. Anything else names a
                # directory on the machine that built the wheel.
                if entry.startswith("@"):
                    continue
                subprocess.run(
                    [_tool("install_name_tool"), "-delete_rpath", entry, str(target)],
                    capture_output=True,
                    # A failure here would leave the original absolute build paths
                    # in place, and the load below would then pass by resolving
                    # through them, which is what this check exists to rule out.
                    check=True,
                )
            remaining = [
                entry for entry in _macho_rpaths(target) if not entry.startswith("@")
            ]
            assert not remaining, (
                f"{name} still carries the absolute runtime search paths {remaining} "
                "after they were deleted, so loading it here would prove nothing"
            )

            message = _dyld_load_failure(target, with_torch=True)
            if not message:
                continue
            missing = _dyld_missing_names(message)
            # Only wheel-provided dependencies are asserted on, because an external
            # one is expected to come from the environment. They are still reported,
            # since silently dropping them would hide a library that resolves only
            # through an absolute build path.
            external = [item for item in missing if item not in shipped]
            if external:
                print(f"- {name} also needs {external} from the environment")
            inside = [item for item in missing if item in shipped]
            if inside:
                broken[name] = inside
            elif not missing:
                # A refusal naming no library at all is not about where files were
                # found, so it belongs to the load check above rather than here.
                print(f"- {name} did not load here either: {message[:160]}")

        assert not broken, (
            "shipped libraries only resolve their wheel-provided dependencies "
            "through absolute build paths, so they would fail on any other "
            f"machine: {broken}"
        )
    print("✓ every shipped library resolves without the build tree")


def test_shipped_libraries_load() -> None:
    """Every shipped library must depend only on things that exist.

    The symbol checks prove each component is defined exactly once, but a library
    can still be unloadable if it needs something nothing provides, which is a
    packaging bug rather than a duplication bug.

    A dependency the wheel ships elsewhere is fine even when `ldd` cannot resolve
    it: some extensions are loaded after `import torch` has already brought their
    dependencies into the process, so they intentionally carry no path to them.
    Only a name nothing in the wheel provides is a real problem.
    """
    # Torch has to be installed for this to mean anything: several shipped libraries
    # depend on it and resolve once it is imported. Without it every one of them looks
    # broken, which would report a packaging fault that does not exist.
    if importlib.util.find_spec("torch") is None:
        print("- torch is not installed, skipping the load check")
        return
    if sys.platform == "darwin":
        _assert_shipped_libraries_load_with_dyld()
        return
    if _tool("ldd") is None:
        print("- ldd not available, skipping the load check")
        return

    package_dir = _installed_package_dir()
    libraries = _shipped_shared_objects(package_dir)
    shipped = {library.name for library in libraries}

    # A dependency is only excusable when the wheel ships it AND the loader can
    # actually reach it from the library that needs it. Loaded-later extensions
    # such as the Torch libraries are the real exception: they resolve once the
    # Python package that owns them is imported. Anything the wheel itself ships
    # must resolve here, because a RUNPATH applies to the library carrying it and
    # is not inherited on behalf of a dependency's own dependencies.
    broken = {}
    unreachable = {}
    unresolved = {}
    for library in libraries:
        resolved = subprocess.run(
            # -r resolves data and function symbols too, not just the NEEDED
            # entries. A SHARED link does not error on undefined symbols, so
            # without this an under-linked library passes here and fails at first
            # use instead.
            [_tool("ldd"), "-r", str(library)],
            capture_output=True,
            text=True,
            check=False,
            # Any loader override in the build environment would paper over a
            # RUNPATH the shipped library is actually missing.
            env=_loader_clean_environment(),
        )
        # ldd reports missing libraries on stdout but undefined symbols on stderr,
        # so both streams matter.
        combined = resolved.stdout + resolved.stderr
        # A non-zero exit with none of the expected text means ldd could not inspect
        # the file at all, which a text-only search reads as "nothing wrong". A file
        # under lib/ that is not a loadable object is a packaging defect, so treat it
        # as one rather than passing it.
        if (
            resolved.returncode != 0
            and "not found" not in combined
            and "undefined symbol" not in combined
        ):
            unresolved[str(library.relative_to(package_dir))] = [
                f"ldd could not inspect this file: {combined.strip()[:160]}"
            ]
            continue
        missing = [
            line.split("=>")[0].strip()
            for line in combined.splitlines()
            if "not found" in line
        ]
        # Interpreter symbols are excluded rather than whole files. A library that
        # is loaded by Python, whether a extension module or an ahead-of-time
        # plugin, resolves those only once an interpreter is running, so ldd can
        # never resolve them and their absence says nothing about packaging.
        # Filtering the symbols rather than guessing from the file name keeps the
        # check active for everything else those libraries need.
        undefined = [
            line.strip()
            for line in combined.splitlines()
            if "undefined symbol" in line
            and not re.search(r"undefined symbol:\s+_?Py", line)
        ]
        if undefined:
            unresolved[str(library.relative_to(package_dir))] = undefined[:5]
        # Torch's own libraries are the documented exception. They are not in this wheel, and a
        # library that needs them resolves once the Python package owning them is imported, which
        # is how every accelerator and AOT library in this package is used. Treating them as
        # missing fails a wheel that works, and it fires only where torch installs its libraries
        # somewhere the plain loader search does not reach.
        absent = [
            name
            for name in missing
            if name not in shipped and not _is_torch_library(name)
        ]
        present_but_unreachable = [name for name in missing if name in shipped]
        if absent:
            broken[str(library.relative_to(package_dir))] = absent
        if present_but_unreachable:
            unreachable[str(library.relative_to(package_dir))] = present_but_unreachable

    assert not broken, (
        "shipped libraries need dependencies that nothing provides, so they will "
        f"fail to load: {broken}"
    )
    assert not unreachable, (
        "shipped libraries need dependencies the wheel ships but the loader "
        "cannot reach from them, which usually means a missing RUNPATH entry: "
        f"{unreachable}"
    )
    assert not unresolved, (
        "shipped libraries reference symbols nothing provides, so they will fail "
        f"at first use rather than at load: {unresolved}"
    )
    print("✓ every shipped library resolves in an environment with torch present")


def test_shipped_libraries_resolve_without_build_tree() -> None:
    """A shipped library must resolve using only its relative runtime paths.

    Packaging copies binaries out of the build directory, so they still carry the
    absolute paths they were linked with. On the machine that produced the wheel
    those paths exist, which means a library whose relative path is wrong can still
    resolve and look correct. Anywhere else it would fail.

    Copy each library and its wheel-provided dependencies into a fresh tree that
    mirrors the wheel layout, drop every absolute runtime path, and check what is
    left is enough.
    """
    if sys.platform == "darwin":
        _assert_shipped_libraries_relocate_with_dyld()
        return
    if _tool("ldd") is None or _tool("patchelf") is None:
        print("- ldd or patchelf unavailable, skipping the relocated load check")
        return

    package_dir = _installed_package_dir()
    libraries = _shipped_shared_objects(package_dir)
    environment = _loader_clean_environment()

    with tempfile.TemporaryDirectory() as work_dir:
        root = Path(work_dir) / package_dir.name
        # Mirror the layout so a relative path such as $ORIGIN/../../lib still
        # points where it would in a real install.
        for library in libraries:
            target = root / library.relative_to(package_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(library, target)

        broken = {}
        for library in libraries:
            target = root / library.relative_to(package_dir)
            current = subprocess.run(
                [_tool("patchelf"), "--print-rpath", str(target)],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            relative = [
                entry for entry in current.split(":") if entry.startswith("$ORIGIN")
            ]
            subprocess.run(
                [_tool("patchelf"), "--set-rpath", ":".join(relative), str(target)],
                # A failure here would leave the original absolute build paths in
                # place, and the check below would then pass by resolving through
                # them, which is exactly what this test exists to rule out.
                check=True,
            )
            resolved = subprocess.run(
                [_tool("ldd"), str(target)],
                capture_output=True,
                text=True,
                check=False,
                env=environment,
            ).stdout
            shipped = {item.name for item in libraries}
            all_missing = [
                line.split("=>")[0].strip()
                for line in resolved.splitlines()
                if "not found" in line
            ]
            # Only wheel-provided dependencies are asserted on, because an external
            # one is expected to come from the environment. They are still reported,
            # since silently dropping them would hide a library that resolves only
            # through an absolute build path.
            missing = [name for name in all_missing if name in shipped]
            external = [name for name in all_missing if name not in shipped]
            if external:
                print(
                    f"- {library.relative_to(package_dir)} also needs "
                    f"{external} from the environment"
                )
            if missing:
                broken[str(library.relative_to(package_dir))] = missing

        assert not broken, (
            "shipped libraries only resolve their wheel-provided dependencies "
            "through absolute build paths, so they would fail on any other "
            f"machine: {broken}"
        )
    print("✓ every shipped library resolves without the build tree")


def test_custom_op_compiles(work_dir: Path) -> None:
    """A custom operator compiles and links against the shipped extension.

    This is how an out-of-tree project adds its own kernels, and it points at the
    Python extension rather than the runtime, so it is not covered by the consumer
    check above.
    """
    # Skipped rather than asserted, the same as every other tool this suite needs.
    # A missing compiler says nothing about the wheel, and aborting here would take
    # the whole run down with it rather than reporting the one check it prevents.
    if _tool("cmake") is None:
        print("- cmake unavailable, skipping the custom op check")
        return

    package_dir = _installed_package_dir()
    if not list(package_dir.glob("extension/pybindings/_portable_lib*")):
        print("- the wheel ships no Python extension, skipping the custom op check")
        return

    source_dir = work_dir / "custom-op"
    build_dir = work_dir / "custom-op-build"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "custom_op.cpp").write_text(_CUSTOM_OP_SOURCE)
    (source_dir / "CMakeLists.txt").write_text(_CUSTOM_OP_CMAKE)

    # Torch's include directory is handed over directly, because the runtime headers
    # include c10 headers that belong to torch. A real out-of-tree project supplies
    # them the same way; the package config has no business shipping another
    # project's headers.
    if importlib.util.find_spec("torch") is None:
        print("- torch is not installed, skipping the custom op check")
        return
    import torch

    torch_include = Path(torch.__path__[0]) / "include"

    configure = subprocess.run(
        [
            _tool("cmake"),
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_PREFIX_PATH={package_dir / 'share' / 'cmake'}",
            f"-DTORCH_INCLUDE_DIR={torch_include}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert configure.returncode == 0, (
        "a custom operator project cannot configure against the wheel: "
        f"{(configure.stderr or configure.stdout).strip()[-600:]}"
    )

    compiled = subprocess.run(
        [_tool("cmake"), "--build", str(build_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, (
        "a custom operator does not compile or link against the shipped extension: "
        f"{(compiled.stderr or compiled.stdout).strip()[-800:]}"
    )
    produced = list(build_dir.rglob(_library_file_name("libcustom_op_check"))) or list(
        build_dir.rglob("custom_op_check.dll")
    )
    assert produced, "the custom operator library was not produced"

    # Loaded, not just built. A shared library on Linux is allowed to have
    # unresolved symbols, so an under-linked custom operator links successfully and
    # fails only when something dlopens it and its registration initialiser runs.
    # That is exactly the failure this contract exists to prevent.
    if importlib.util.find_spec("torch") is None:
        print("- torch is not installed, so the custom operator is only built")
        return
    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch\n"
            "from executorch.extension.pybindings import portable_lib\n"
            "from executorch.extension.pybindings.portable_lib import (\n"
            "    _get_operator_names,\n"
            ")\n"
            "before = set(_get_operator_names())\n"
            f"torch.ops.load_library({str(produced[0])!r})\n"
            "after = set(_get_operator_names())\n"
            "arrived = sorted(after - before)\n"
            "print('ARRIVED', arrived)\n"
            # The DELTA, not the post-load set: an operator already registered before the load would
            # satisfy a membership test on `after`, so a library that registered nothing would pass.
            "target = [n for n in arrived if 'custom_double' in n]\n"
            "print('TARGET', target)\n"
            "assert target, (\n"
            "    'loading the custom operator library added no operator named custom_double to '\n"
            "    'the registry the shipped extension queries, so a consumer could not call it. '\n"
            "    'Operators that did arrive: ' + repr(arrived)\n"
            ")\n"
            "print('loaded')",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_loader_clean_environment(),
    )
    assert loaded.returncode == 0, (
        "a custom operator built against the shipped extension cannot be loaded, or it "
        "loaded without registering into the extension's operator registry, so it would "
        "fail at first use rather than at link time: "
        f"{(loaded.stderr or loaded.stdout).strip()[-800:]}"
    )
    # Belt and braces: the child asserts, and the parent confirms the child actually reported the
    # operator rather than exiting 0 down some path that skipped the assertion.
    assert re.search(r"TARGET \[[^\]]*custom_double", loaded.stdout), (
        "the loader child did not report the registered custom operator, so this check "
        "cannot show the registration reached the shipped registry: "
        f"{loaded.stdout.strip()[-400:]}"
    )
    print(
        "✓ a custom operator compiles against and registers into the shipped extension"
    )


def _find_wheel_files() -> list:
    """The built wheel files, searched where a build actually leaves them.

    WHEEL_DIR is honoured when set, but it is not set in the wheel-build job, so the
    usual output directories are searched too. Without this the check has nothing to
    inspect and skips.
    """
    candidates = []
    configured = os.environ.get("WHEEL_DIR")
    if configured:
        candidates.append(Path(configured))
    # The build leaves the wheel in dist/ at the repository root, and this file sits at a
    # fixed depth below that root, so the location follows from __file__ rather than from
    # the current directory. The release job runs the smoke test from the workspace above
    # the repository, where a cwd-relative guess finds nothing.
    #
    # Guarded because a copy of this file can live outside that layout, where indexing
    # past the available parents would raise instead of falling through to the other
    # candidates.
    here = Path(__file__).resolve()
    repository_root = here.parents[3] if len(here.parents) > 3 else here.parent
    candidates += [
        repository_root / "dist",
        Path.cwd() / "dist",
        Path.cwd(),
        repository_root / "wheelhouse",
    ]
    for directory in candidates:
        try:
            found = sorted(directory.glob("executorch-*.whl"))
        except OSError:
            continue
        if found:
            return found
    return []


# The architectures a Linux wheel tag can name. Matched by suffix rather than parsed
# positionally, because the version part differs between spellings (linux_x86_64,
# manylinux_2_28_x86_64, manylinux2014_x86_64) enough that a positional pattern picks
# it up as part of the name.
#
# Linux only, which is what this check reads. `arm64` deliberately absent: it is the
# macOS spelling, Linux uses aarch64, and including it made a macosx_11_0_arm64 tag
# look like something this could compare. A tag naming none of these is reported as
# unreadable rather than as a mismatch.
_WHEEL_ARCHITECTURES = ("x86_64", "aarch64", "i686", "ppc64le", "s390x", "armv7l")


def _wheel_architecture(tag: str):
    """The architecture a platform tag names, or None if it names none of ours."""
    for architecture in _WHEEL_ARCHITECTURES:
        if tag.endswith("_" + architecture):
            return architecture
    return None


def _tag_architectures_match(claimed: str, supported: str):
    """Whether two platform tags name the same architecture.

    Returned rather than asserted so the same decision can be unit tested without a
    wheel. The previous arrangement duplicated the comparison in the test, which meant
    the test could pass while the shipped check was wrong, and that is exactly what
    happened: the original defect was in how the caller compared the two tags, and a
    test that re-implemented the comparison could not see it.

    None for either side means the tag names no architecture this project builds for,
    which is a different failure from a mismatch and is reported separately.
    """
    claimed_arch = _wheel_architecture(claimed)
    supported_arch = _wheel_architecture(supported)
    if claimed_arch is None or supported_arch is None:
        return None
    return claimed_arch == supported_arch


def test_wheel_platform_tag() -> None:
    """The wheel's declared platform tag must name the architecture it was built for.

    Only the architecture. auditwheel cannot certify a glibc baseline for this wheel:
    it depends on torch without vendoring torch's libraries, so the contents reference
    libtorch.so from outside any manylinux policy and auditwheel reports a plain
    linux_<arch>. That is the expected answer for a torch-dependent wheel rather than
    a defect, and the manylinux tag on the file comes from the build image. Asserting
    the baseline here failed every correct wheel.

    The architecture is still worth checking, because a wheel labelled with the wrong
    one installs on machines it cannot run on at all, and that is a mistake this can
    actually catch.
    """
    wheels = _find_wheel_files()
    if not wheels:
        # Failing rather than returning, because this is the only check that reads the declared
        # platform tag at all, and a silent return reads as a pass in the job summary. A caller
        # that genuinely has no wheel says so.
        assert os.environ.get("EXECUTORCH_TEST_WITHOUT_WHEEL"), (
            "no built wheel was found to inspect, so the platform tag was never checked. Run this "
            "from a tree where the wheel was built, or set EXECUTORCH_TEST_WITHOUT_WHEEL to state "
            "that no wheel is expected"
        )
        print("- no wheel file to inspect, skipping the platform tag check")
        return

    # Before auditwheel is provisioned, because the Mach-O route reads the architecture with lipo
    # and never imports it. Installing a tool this platform does not use is one more way for a
    # correct wheel to fail the job.
    if sys.platform == "darwin":
        _assert_mach_o_architecture_matches(wheels[-1])
        return

    if importlib.util.find_spec("auditwheel") is None:
        # Installed here rather than skipped, because auditwheel is not in any CI
        # image and a skip is indistinguishable from a pass in the summary. This
        # check is the only thing that reads the architecture out of the wheel's
        # contents rather than out of its file name, and this change adds five
        # libraries under that tag.
        print("- auditwheel not present, installing it so this check can run")
        installed = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "auditwheel"],
            capture_output=True,
            text=True,
            check=False,
        )
        if installed.returncode != 0 or importlib.util.find_spec("auditwheel") is None:
            raise AssertionError(
                "auditwheel is required to check the wheel's platform tag and could not "
                "be installed. Skipping instead would report a pass, and this is the only "
                "check that reads the architecture out of the wheel's contents rather than "
                f"out of its file name: {installed.stderr.strip()[-200:]}"
            )
        importlib.invalidate_caches()

    result = subprocess.run(
        [sys.executable, "-m", "auditwheel", "show", str(wheels[-1])],
        capture_output=True,
        text=True,
        check=False,
    )
    # auditwheel wraps its verdict across lines, so compare on collapsed
    # whitespace rather than the literal output.
    combined = " ".join((result.stdout + result.stderr).split())
    match = re.search(
        r'consistent with the following platform tag: "([^"]+)"', combined
    )
    assert match, (
        "auditwheel reported no platform tag for the wheel, so its contents could "
        f"not be checked against what it claims: {combined[-400:]}"
    )
    claimed = wheels[-1].name.split("-")[-1].removesuffix(".whl")
    supported = match.group(1)

    claimed_arch = _wheel_architecture(claimed)
    supported_arch = _wheel_architecture(supported)
    matches = _tag_architectures_match(claimed, supported)
    assert matches is not None, (
        f"could not read an architecture from the declared tag {claimed} or from what "
        f"auditwheel reported, {supported}"
    )
    assert matches, (
        f"the wheel claims architecture {claimed_arch} but its contents are built for "
        f"{supported_arch}, so it would install where it cannot run"
    )
    print(f"✓ the wheel is tagged for the architecture it contains ({claimed_arch})")


def _names_a_build_directory(entry: str) -> bool:
    """Whether a runtime path entry points inside the tree that built the wheel.

    Compared as whole path components, matching packaging's own test at setup.py:1288 including its
    anchored lib.<platform>-<pyver> form, because a bare prefix also matched a real user directory such
    as lib.backup.

    The build-directory rule matches packaging. The surrounding check does NOT: packaging is a denylist
    that keeps any absolute path it has no relative answer for, while the caller here allows only four
    suffixes, so a library recording /usr/lib64 or /opt/rocm/lib would fail this check even though
    packaging deliberately preserved it. No shipped wheel records one today. Reconciling the two is a
    separate change; do not read this helper as evidence that they already agree.

    Module scope rather than nested, so its branches do not count against the enclosing test's
    complexity, which lintrunner caps at 12.
    """
    parts = entry.split("/")
    if any(
        part in ("pip-out", "cmake-out")
        or re.fullmatch(r"lib\.[^/]+-(cpython-\d+|\d+(?:\.\d+)*)", part)
        for part in parts
    ):
        return True
    return any(
        part in ("actions-runner", "_work", "_temp", "runner")
        or part.startswith("conda_environment_")
        for part in parts
    )


def test_no_absolute_runtime_paths() -> None:
    """No shipped library may search a directory a user does not have.

    Packaging copies libraries out of the build tree rather than installing them, so
    every directory the linker recorded ships as-is. Two kinds are rejected on every
    shipped library, not only the lib/ payload:

    - a directory inside the build, which names the machine that produced the wheel
    - an empty entry, which the loader reads as the process working directory

    Torch's own directory is accepted. The extensions link torch and resolve it
    through the directory the linker recorded, so that entry is load-bearing rather
    than leftover. Narrowing this check to lib/ once hid seven extensions carrying
    build-tree paths, so the exclusion is by what an entry POINTS AT, never by which
    file carries it.

    The check reads the shipped file directly, with nothing stripped, which is what
    a user actually receives.
    """
    # Fatal, not a skip. Packaging strips these paths best-effort, because it cannot
    # guarantee patchelf on PATH, so this is the only place the guarantee can be
    # enforced. If both went quiet on the same missing tool, a wheel carrying the
    # build machine's directories would ship looking correct.
    # Mach-O keeps its search path in load commands that otool reads, and otool comes
    # with the developer tools, so only the ELF side needs an install step.
    if sys.platform != "darwin" and _tool("patchelf") is None:
        print("- patchelf not present, installing it so this check can run")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "patchelf"],
            capture_output=True,
            text=True,
            check=False,
        )
    reader = _tool("otool") if sys.platform == "darwin" else _tool("patchelf")
    assert reader is not None, (
        "a runtime path reader is required to check the shipped runtime paths and could "
        "not be found. Packaging strips build-tree directories, and without a reader "
        "here neither side would notice that they were left in place."
    )

    package_dir = _installed_package_dir()

    # This project's libraries must not name an absolute directory the wheel has a relative route to. The
    # one that shipped was a CUDA toolkit prefix recorded on the build machine: it sat ahead of the relative
    # hop, so a user with a toolkit at the same prefix resolved the CUDA runtime from there instead of from
    # the declared dependency, and the builder always has one, so nothing exercised the hop.
    #
    # Stated as a property rather than a list of known-bad directories, because a list only catches what
    # someone already thought of and that prefix was not on one.
    #
    # PyTorch's own directory is allowed: the wheel neither declares nor bundles PyTorch, so an absolute
    # path is the only way to reach it. The maths library directories are allowed too. They arrive as
    # -L flags in PyTorch's exported link interface, which CMake mirrors into the runtime path, so every
    # library here that links PyTorch carries them. They point nowhere on any machine: measured on the
    # link line as -L/lib/intel64 -L/lib/intel64_win -L/lib/win-x64, which is a prefix variable that
    # resolved empty leaving the concatenation at the filesystem root.
    #
    # Matched as a suffix, the same way packaging decides what to strip at setup.py:1300. A substring
    # test exempted any path merely CONTAINING one of these, so a directory such as
    # /home/user/torch/lib.backup/stage passed without ever reaching the build-directory classifier.
    # Both "_win" spellings are listed explicitly now that the match is anchored.
    #
    # A torch directory inside a CI worker tree, such as
    # /home/ec2-user/actions-runner/_work/.../pytorch/torch/lib, is rejected rather than allowed: the
    # build-directory classifier sees the worker components and the allowlist never gets to accept the
    # /torch/lib suffix. Packaging strips the same entry, because every extension that names Torch now
    # records a relative route to it.
    allowed_absolute = (
        "/torch/lib",
        "/lib/intel64",
        "/lib/intel64_win",
        "/lib/win-x64",
    )
    # Held for a wheel that bundles PyTorch's libraries rather than declaring them: such a copy records
    # the CUDA toolkit directory of the machine that built IT, which is not this project's to fix.
    #
    # No wheel ships one today. Six wheels across manylinux and macOS contain zero files with these
    # prefixes, because the wheel declares torch as a dependency and there is no auditwheel step, so
    # this clause is currently never false. It stays as a guard for a future bundling change; if that
    # never comes, delete it rather than leaving an unexercised exemption in the check.
    vendored_prefixes = (
        "libtorch",
        "libc10",
        "libshm",
        "libcaffe2",
        "libgomp",
        "libiomp",
    )

    offenders = {}
    inspected = 0
    with_a_runtime_path = 0
    for library in _shipped_shared_objects(package_dir):
        entries = _runtime_search_paths(library)
        if entries is None:
            continue
        inspected += 1
        # A library with nothing to search is fine; the defect is searching somewhere
        # unusable. An absent search path and one holding a single empty entry are the
        # same thing here, and the reader reports both as an empty list.
        if not entries:
            continue
        with_a_runtime_path += 1
        bad = []
        for entry in entries:
            if not entry:
                bad.append("<empty>")
            elif (
                entry.startswith("/")
                and not library.name.startswith(vendored_prefixes)
                and (
                    _names_a_build_directory(entry)
                    or not any(
                        entry.rstrip("/").endswith(allowed)
                        for allowed in allowed_absolute
                    )
                )
            ):
                # Named separately so the message says which kind it is: a build directory and a
                # toolkit prefix are the same defect with different causes.
                kind = (
                    "inside a build of this project"
                    if _names_a_build_directory(entry)
                    else "an absolute directory the wheel has a relative route to"
                )
                bad.append(f"{entry} ({kind})")
        if bad:
            offenders[str(library.relative_to(package_dir))] = bad

    assert not offenders, (
        "shipped libraries search directories a user does not have, either inside "
        "the build tree that produced the wheel or, for an empty entry, the process "
        f"working directory: {offenders}"
    )
    # Counted separately, because a wheel whose libraries all had their runtime paths removed
    # entirely would satisfy a readable-file count while this check examined no path at all.
    assert with_a_runtime_path, (
        f"none of the {inspected} shipped libraries under {package_dir} carries a runtime search path, "
        "so this check examined nothing. The shipped libraries need a relative path to reach each other."
    )
    print(
        f"✓ none of the {with_a_runtime_path} shipped libraries with a runtime path searches a "
        f"build-tree or empty directory ({inspected} inspected)"
    )


def test_extension_contains_no_component() -> None:
    """The Python extension must link the components, not contain them.

    This is the property the change exists to create, and no count of definers
    proves it: the monolithic layout has exactly one definer of every symbol too,
    inside the extension. The direct statement is that the extension defines none of
    what the shipped libraries own, and records a dependency on each instead.
    """
    assert _tool("nm") is not None, "nm is required to inspect the wheel"

    package_dir = _installed_package_dir()
    extensions = sorted(
        (package_dir / "extension" / "pybindings").glob("_portable_lib.*.so")
    )
    assert len(extensions) == 1, f"expected one _portable_lib, found {extensions}"
    extension = extensions[0]

    lib_dir = package_dir / "lib"
    if not lib_dir.is_dir():
        print("- this wheel ships no lib directory, nothing to check")
        return

    # Every symbol group a shipped library owns. The extension holding any of these
    # means it still carries its own copy of that component.
    # Derived from the ownership table rather than restated. A hand-written copy drifts: this once
    # listed five symbol groups while the table covered eleven, so the components added later, including
    # both CUDA ones, were never checked here.
    #
    # The bundled third-party groups are left out on purpose. That code is also linked by torch, and the
    # extension links torch, so seeing those symbols there says nothing about this split.
    # Only components whose owning library is actually in this wheel. A build with an optional component
    # turned off ships no owner, and asserting the extension does not define its symbols would reject a
    # configuration the table itself marks as supported. Required rows are kept regardless, because their
    # owner missing is a packaging fault the owner check reports, and a conditional marker counts as
    # required here so a CPU wheel is still checked for CUDA symbols it should not contain.
    shipped = {path.name for path in _shipped_runtime_libraries(package_dir)}
    # Guarded here rather than further down, so it protects the filter that uses it: a wheel that
    # installed no runtime libraries would otherwise compare the extension against an empty set and pass.
    assert shipped, (
        f"the wheel installed no runtime libraries under {package_dir / 'lib'}, so this check would "
        "compare the extension against nothing and pass"
    )
    owned = tuple(
        symbol
        for _, symbols, owner, required in _OWNED_COMPONENTS
        if symbols not in (_BUNDLED_THREADPOOL_SYMBOLS, _BUNDLED_XNNPACK_SYMBOLS)
        and (required or any(name.startswith(owner) for name in shipped))
        for symbol in symbols
    )
    contained = [symbol for symbol in owned if _defines_symbol(extension, symbol)]
    assert not contained, (
        f"{extension.name} defines {contained}, which the shipped libraries own. The "
        "extension is supposed to link them rather than contain them, so this is the "
        "monolithic layout the split removes."
    )

    # And it has to actually depend on each shipped library. Defining nothing while
    # depending on nothing would be an extension that cannot work at all.
    needed = _recorded_dependencies(extension)
    # Only the libraries whose code the extension used to contain. Those are the ones
    # this split moved out of it, so the extension must now resolve them from outside or
    # a retention option silently failed.
    #
    # Not every shipped library serves Python. The quantized kernels exist for a C++
    # application, since Python registers those operators through the torch-linked
    # ahead-of-time library at export time, and requiring a dependency would demand the
    # extension link code it has no use for.
    #
    # The CUDA delegate is NOT in that category. The build deliberately links it into the
    # extension with a retention option, so it does carry a dependency, and excluding it
    # switched off the one check that would notice if that retention stopped working. The
    # stream helper stays excluded because the extension reaches it only through the
    # delegate's public link, with no retention of its own to protect.
    expected = {
        name
        for name in shipped
        if not any(marker in name for marker in ("kernels_quantized", "extension_cuda"))
    }
    unused = sorted(expected - needed)
    assert not unused, (
        f"the wheel ships {unused} but {extension.name} does not depend on them, so "
        "either they are dead weight or a retention option did not hold"
    )

    # Two shipped libraries register the same quantized operators, one for export and one for a C++
    # application, and the runtime treats a repeat registration as fatal. Reaching both from one process
    # aborts it, and the only thing preventing that is this extension not depending on the run-time one.
    assert not any("kernels_quantized" in name for name in needed), (
        f"{extension.name} depends on the run-time quantized library, which registers the same operators "
        "as the export-time one it already loads. The runtime aborts on a repeat registration, so "
        "importing this extension would kill the process."
    )

    # Positive proof that the extension resolves these from elsewhere, rather than
    # only the absence of a visible definition. A hidden or local copy would not
    # appear in the dynamic symbol table at all, so "defines nothing" on its own is
    # satisfiable by an extension that still carries its own private runtime. An
    # UNDEFINED reference cannot be faked that way: it says the definition is not
    # here and has to come from a dependency.
    undefined = subprocess.run(
        [_tool("nm"), *_nm_undefined_args(), str(extension)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    defined = subprocess.run(
        [_tool("nm"), *_nm_defined_args(), str(extension)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    candidates = (*_REGISTRY_SYMBOLS, *_THREADPOOL_SYMBOLS)
    # A symbol the extension defines itself is the hidden copy this check exists to catch. A
    # symbol it neither imports nor defines is simply unused, which happens for the backend
    # registry when the wheel is built with the optional delegates off.
    carried = [
        symbol for symbol in candidates if symbol not in undefined and symbol in defined
    ]
    assert not carried, (
        f"{extension.name} defines {carried} itself rather than importing it, so it carries a "
        "private copy of a component the wheel also ships as a library"
    )
    # A hidden definition appears in neither table, so the check above cannot see the
    # worst version of this: an extension that whole-archived a private runtime with
    # hidden visibility and kept the shipped one as a dependency it never uses. What it
    # calls has to be imported, and these it calls.
    unimported = [symbol for symbol in _EXTENSION_IMPORTS if symbol not in undefined]
    assert not unimported, (
        f"{extension.name} calls {unimported} but imports none of them, so the definition it "
        "reaches is inside itself and the process holds a second registry the shipped runtime "
        "cannot see"
    )
    print(
        f"✓ {extension.name} ({extension.stat().st_size // 1024} KiB) contains no "
        f"component, imports the runtime symbols it uses, and depends on all "
        f"{len(expected)} shipped libraries it used to contain"
    )


def test_shipped_library_names_are_expected() -> None:
    """Every library in lib/ must be one this build could have produced.

    Packaging copies binaries out of a staging directory rather than running an
    install step, so anything left there from an earlier build ships too. That
    really happened: a wheel picked up three libraries from a different revision
    and still passed every symbol check, because those checks only ask how many
    definers a symbol has, never whether a file belongs in the wheel at all.

    Two properties catch it. Its name is one packaging knows how to produce, which is
    what a leftover from an older layout fails. And its recorded soname matches its
    file name, or a consumer records a dependency the wheel does not contain.

    The names are unversioned, because these libraries ship one file each with no
    symlink chain, and a versioned name without the usual symlinks is harder to load
    rather than safer.
    """
    package_dir = _installed_package_dir()
    lib_dir = package_dir / "lib"
    if not lib_dir.is_dir():
        print("- this wheel ships no lib directory, nothing to check")
        return

    # Regular files only. A symlink here would be a deliberate alias rather than the
    # stale-artifact case this check is about, and a leftover from an earlier build is
    # a real file, so it is still caught.
    shipped = sorted(
        p
        for p in lib_dir.glob(f"*{_dynamic_lib_suffix()}*")
        if p.is_file() and not p.is_symlink()
    )
    assert shipped, f"the wheel ships a lib directory with no libraries: {lib_dir}"

    # The names packaging can put here. Listed rather than derived because setup.py
    # names each one literally, and a file with any other name did not come from
    # this build. Which of them are present depends on the build options, so
    # absence is fine and an unknown name is not.
    #
    # Matched in full rather than by taking the part before the first ".so", because
    # that prefix is satisfied by a name like libexecutorch.so.old.so.1, which is
    # exactly the shape a leftover file takes.
    known = (
        "libexecutorch",
        "libexecutorch_kernels_optimized",
        "libexecutorch_kernels_quantized",
        "libexecutorch_backend_cuda",
        "libexecutorch_extension_cuda",
        # The same library under the name a non-shared build gives it. The shared
        # build renames it to match the other shipped components; every other build
        # leaves this spelling, and the shim layer records whichever one exists as a
        # dependency, so both have to ship and both are expected here.
        "libextension_cuda",
        "libexecutorch_backend_xnnpack",
        "libexecutorch_backend_openvino",
        "libexecutorch_threadpool",
        "libexecutorch_etdump",
    )
    # Unversioned, because the wheel build does not version these. A trailing
    # .<digits> would also be a name packaging did not produce here. These are
    # libraries, so the suffix follows the platform and macOS spells them .dylib.
    permitted = re.compile(rf"(?:{'|'.join(known)})\{_dynamic_lib_suffix()}")
    unknown = sorted(p.name for p in shipped if not permitted.fullmatch(p.name))
    assert not unknown, (
        f"the wheel ships {unknown} under lib/, which packaging does not produce. "
        "A file packaging did not put there came from a stale staging directory, "
        "and it ships while looking correct to every other check."
    )

    # The recorded identity has to match the file name, or a consumer records a
    # dependency on a name the wheel does not contain.
    mismatched = {}
    absolute = {}
    for library in shipped:
        identity = _recorded_identity(library)
        if identity != library.name:
            mismatched[library.name] = identity
        # Checked separately from the name comparison above, which deliberately reduces to a
        # basename so an @rpath/ prefix compares equal. That reduction also makes an absolute
        # install name compare equal, and a consumer copies the recorded string verbatim into its
        # own load command, so an absolute one only resolves on the machine that built the wheel.
        raw = _raw_recorded_identity(library)
        if raw is not None and raw.startswith("/"):
            absolute[library.name] = raw
    # Before the name comparison: this is the more specific condition of the two. On ELF the
    # recorded soname is returned whole, so an absolute one populates both sets, and reporting the
    # name mismatch first would claim the wheel does not ship a file it does ship.
    assert not absolute, (
        "shipped libraries record an absolute install name, so a consumer copies a path that "
        f"exists only on the build machine: {absolute}"
    )
    assert not mismatched, (
        "shipped libraries record an identity that is not their file name, so a "
        f"consumer would look for a file the wheel does not ship: {mismatched}"
    )
    print(f"✓ {len(shipped)} shipped libraries have expected names and identities")


_PARITY_MODEL = '''
import json
import sys

import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)


class Net(torch.nn.Module):
    """Several operator kinds rather than one, so the run exercises the merged CPU
    kernels rather than a single add."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
        self.conv = torch.nn.Conv2d(1, 4, 3, padding=1)

    def forward(self, x, image):
        a = torch.relu(self.linear(x))
        b = self.conv(image).flatten(1)
        return a.sum(dim=1, keepdim=True) + b.mean(dim=1, keepdim=True)


delegate = sys.argv[1] == "delegate"
torch.manual_seed(0)
model = Net().eval()
example = (torch.randn(2, 8), torch.randn(2, 1, 6, 6))
with torch.no_grad():
    expected = model(*example)

partitioners = []
if delegate:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )

    partitioners = [XnnpackPartitioner()]

program = to_edge_transform_and_lower(
    torch.export.export(model, example), partitioner=partitioners
).to_executorch()
buffer = program.buffer

actual = _load_for_executorch_from_buffer(buffer).forward(list(example))[0]
# Compared rather than merely run. The point of the split is that behaviour does
# not change, and only a numeric comparison shows that; a model that returns
# wrong values without erroring passes everything else.
torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

print(json.dumps({"delegated": delegate, "has_xnnpack": b"XnnpackBackend" in bytes(buffer)}))
'''


def test_model_matches_eager_pytorch(work_dir: Path) -> None:
    """A model exported and run through the bindings must match eager PyTorch.

    Twice: once plain, so the CPU kernels resolve from the shared library, and once
    delegated to XNNPACK, so the delegate does. The delegated program is also
    checked for the delegate's own identity, because a partitioner that claimed
    nothing would silently fall back to the CPU kernels and still match.

    Separate processes, because a fault in one export leaves state that makes the
    next look broken when it is not.
    """
    if importlib.util.find_spec("torch") is None:
        print("- torch is not installed, skipping the eager comparison")
        return

    work_dir.mkdir(parents=True, exist_ok=True)
    script = work_dir / "parity.py"
    script.write_text(_PARITY_MODEL)

    # Both halves run. The delegate is a required component in the ownership table
    # above, so a wheel reaching here without it has already failed that check, and
    # tolerating its absence here would only hide a second symptom of the same fault.
    for mode in ["plain", "delegate"]:
        result = subprocess.run(
            [sys.executable, str(script), mode],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(work_dir),
        )
        assert result.returncode == 0, (
            f"the {mode} model does not export, run, and match eager PyTorch: "
            f"{(result.stderr or result.stdout).strip()[-1500:]}"
        )
        report = json.loads(result.stdout.strip().splitlines()[-1])
        if mode == "delegate":
            assert report["has_xnnpack"], (
                "the delegated export produced a program with no XNNPACK partition, "
                "so the delegate was never exercised and the comparison only proves "
                "the CPU kernels work"
            )
        print(f"✓ the {mode} model matches eager PyTorch")


def test_declared_dependencies_match_the_wheel_tag() -> None:
    """A CPU wheel must not declare the CUDA runtime, and a CUDA wheel must declare its own train.

    The tag is what a user resolves against, so a mismatch is a promise the wheel cannot keep in
    either direction: a CPU wheel that pulls the CUDA packages costs a user hundreds of megabytes it
    never loads, and a CUDA wheel that declares nothing leaves the runtime unresolvable.

    Declaring the wrong train is the quiet case, and the reason this checks the names rather than
    only their presence. The CUDA 12 packages are published with a "-cu12" suffix and the CUDA 13
    ones without, so a cu130 wheel that asked for the cu12 packages would install a runtime its
    libraries cannot load, while looking correctly specified.

    This is metadata only, so no library check can see it. A CPU wheel that wrongly declared the CUDA
    runtime passed every other check in this file.
    """
    requirements = importlib.metadata.requires("executorch") or []

    # Split off any environment marker AND any version specifier. The name, the specifier
    # and the marker can arrive as one token, so taking the first whitespace-separated
    # word left "nvidia-cuda-runtime-cu12==12.6.77" as the name and made a correctly
    # specified wheel fail the moment any CUDA dependency gained a pin.
    def distribution_name(requirement: str) -> str:
        return re.split(r"[\s;\[<>=!~(]", requirement.strip(), maxsplit=1)[0]

    # The specifier is part of what is compared, not just the name: a bound is only correct
    # relative to the train the libraries were linked against, so a name-only comparison accepted
    # nvidia-cuda-runtime<13 and ==14.0 on a cu130 wheel. Compared through packaging's own parser
    # rather than as text, because it reorders the clauses it emits: a declared ">=13,<14" appears
    # in METADATA as "<14,>=13". A raw string comparison therefore failed on a correctly built
    # wheel, which is a self-inflicted gate rather than a defect in the wheel.
    def normalized_requirement(requirement: str) -> str:
        parsed = Requirement(requirement)
        return f"{canonicalize_name(parsed.name)}{parsed.specifier}"

    cuda = sorted(
        normalized_requirement(r)
        for r in requirements
        if distribution_name(r).lower().startswith("nvidia")
    )

    # The local version segment of the installed version states what the wheel was built for.
    version = importlib.metadata.version("executorch")
    local = version.partition("+")[2]
    is_cuda_wheel = local.startswith("cu")

    if is_cuda_wheel:
        assert cuda, (
            f"version {version} says this is a CUDA wheel, but it declares no CUDA runtime "
            "packages, so nothing resolves the runtime it links"
        )
        # Compared as sets in both directions rather than as a name suffix: for CUDA 13 the
        # expected suffix is the empty string and every name ends with that, so a suffix test
        # accepted a name from any train whose spelling happened not to be one of the two
        # literals it also excluded. Measured: a cu130 wheel declaring nvidia-cuda-runtime-cu11
        # passed. The reverse check catches the other side of the same defect: a wheel that
        # declares one package and omits the others still cannot load, and one-direction only
        # would accept it.
        train = local[len("cu") : len("cu") + 2]
        expected = {
            normalized_requirement(requirement)
            for requirement in _EXPECTED_CUDA_PACKAGES.get(train, ())
        }
        assert expected, (
            f"version {version} names CUDA train {train}, which this check has no expected "
            f"package list for. Add it beside the packaging list it mirrors."
        )
        actual = set(cuda)
        wrong = sorted(actual - expected)
        missing = sorted(expected - actual)
        # Split by distribution name before the set difference is reported, so the message names the
        # actual fault. Comparing whole requirement strings makes a correct package with a missing
        # bound look identical to a package from the wrong train, and the second message would be
        # false. Diagnostic precision is the reason this comparison is exact rather than a heuristic.
        expected_by_name = {distribution_name(r): r for r in expected}
        misbounded = sorted(
            (declared, expected_by_name[distribution_name(declared)])
            for declared in wrong
            if distribution_name(declared) in expected_by_name
        )
        assert not misbounded, (
            f"version {version} is a CUDA {train} wheel and names the right runtime packages, but "
            f"their version specifiers are not the ones packaging declares: "
            + ", ".join(f"{d!r} should be {e!r}" for d, e in misbounded)
            + ". A user could resolve a CUDA major whose libcudart this wheel's libraries cannot load."
        )
        assert not wrong, (
            f"version {version} is a CUDA {train} wheel, but it declares {wrong}, which belong to "
            f"another CUDA train. Expected only {sorted(expected)}. A user would install a runtime "
            "this wheel's libraries cannot load."
        )
        assert not missing, (
            f"version {version} is a CUDA {train} wheel, but it does not declare {missing} "
            f"(expected {sorted(expected)}). A user installing this wheel would end up without part "
            "of the CUDA runtime the wheel's libraries need."
        )
        print(
            f"✓ this CUDA {train} wheel declares its own runtime ({len(cuda)} packages)"
        )
    else:
        assert not cuda, (
            f"version {version} is not a CUDA wheel, yet it declares {cuda}. A user installing it "
            "would download the CUDA runtime this wheel never loads."
        )
        print("✓ this non-CUDA wheel declares no CUDA runtime")


def run_tests(work_dir: Path) -> None:
    # Ordered by what a failure tells you, because these run in sequence and the
    # first failure stops the rest. The checks that prove the split behaves
    # correctly come first; packaging metadata comes last, so a weak check cannot
    # hide a strong one.
    test_each_component_has_one_owner()
    test_python_extensions_import()
    test_declared_dependencies_match_the_wheel_tag()
    test_extension_contains_no_component()
    test_shipped_library_names_are_expected()
    test_shipped_libraries_load()
    test_shipped_libraries_resolve_without_build_tree()
    test_custom_op_compiles(work_dir)
    test_no_absolute_runtime_paths()
    test_model_matches_eager_pytorch(work_dir)
    # Last: a wrong answer here says the wheel is labelled wrong, not that the
    # split is broken.
    test_wheel_platform_tag()
