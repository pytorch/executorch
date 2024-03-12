# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under both the MIT license found in the
# LICENSE-MIT file in the root directory of this source tree and the Apache
# License, Version 2.0 found in the LICENSE-APACHE file in the root directory
# of this source tree.

load("@prelude//:paths.bzl", "paths")
load("@fbsource//tools/build_defs/buck2:is_buck2.bzl", "is_buck2")

def subdir_glob(glob_specs, exclude = None, prefix = ""):
    """Returns a dict of sub-directory relative paths to full paths.

    The subdir_glob() function is useful for defining header maps for C/C++
    libraries which should be relative the given sub-directory.
    Given a list of tuples, the form of (relative-sub-directory, glob-pattern),
    it returns a dict of sub-directory relative paths to full paths.

    Please refer to native.glob() for explanations and examples of the pattern.

    Args:
      glob_specs: The array of tuples in form of
        (relative-sub-directory, glob-pattern inside relative-sub-directory).
        type: List[Tuple[str, str]]
      exclude: A list of patterns to identify files that should be removed
        from the set specified by the first argument. Defaults to [].
        type: Optional[List[str]]
      prefix: If is not None, prepends it to each key in the dictionary.
        Defaults to None.
        type: Optional[str]

    Returns:
      A dict of sub-directory relative paths to full paths.
    """
    if exclude == None:
        exclude = []

    results = []

    for dirpath, glob_pattern in glob_specs:
        results.append(
            _single_subdir_glob(dirpath, glob_pattern, exclude, prefix),
        )

    return _merge_maps(*results)

def _merge_maps(*file_maps):
    result = {}
    for file_map in file_maps:
        for key in file_map:
            if key in result and result[key] != file_map[key]:
                fail(
                    "Conflicting files in file search paths. " +
                    "\"%s\" maps to both \"%s\" and \"%s\"." %
                    (key, result[key], file_map[key]),
                )

            result[key] = file_map[key]

    return result

def _single_subdir_glob(dirpath, glob_pattern, exclude = None, prefix = None, strict = False):
    if exclude == None:
        exclude = []
    results = {}
    glob_func = (lambda *args, **kwargs: strict_glob(_called_by_subdir_glob = True, *args, **kwargs)) if strict else native.glob
    files = glob_func([paths.join(dirpath, glob_pattern)], exclude = exclude)
    for f in files:
        if dirpath:
            key = f[len(dirpath) + 1:]
        else:
            key = f
        if prefix:
            key = paths.join(prefix, key)
        results[key] = f

    return results

# Using a flat list will trigger build errors on Android.
# cxx_library will generate an apple_library on iOS, a cxx_library on Android.
# Those rules have different behaviors. Using a map will make the behavior consistent.
#
def glob_private_headers(glob_patterns, exclude = []):
    result = {}
    headers = native.glob(glob_patterns, exclude = exclude)
    for header in headers:
        result[paths.basename(header)] = header
    return result

# glob gives buck2 compatible behavior on buck1.
def glob(include, exclude = (), **kwargs):
    if is_buck2():
        return native.glob(include, exclude = exclude, **kwargs)
    buildfile = native.read_config("buildfile", "name", "BUCK")
    subpkgs = [
        target[:-len(buildfile)] + "**/*"
        for target in native.glob(["*/**/" + buildfile])
    ]
    return native.glob(include, exclude = list(exclude) + subpkgs, **kwargs)

def strict_glob(include, _called_by_subdir_glob = False, **kwargs):
    """Has the same API as glob(), but will fail() if any entry in include fails to match."""
    has_glob_pattern = False
    all_results = []
    for entry in include:
        has_glob_pattern = has_glob_pattern or "*" in entry

        # @lint-ignore BUCKRESTRICTEDSYNTAX, no-implicit-native-glob
        result = glob([entry], **kwargs)
        if len(result) == 0:
            fail("Strict glob failed to match files for glob pattern: " + entry)
        if result not in all_results:
            all_results.extend(result)

    for entry in kwargs.get("exclude", []):
        has_glob_pattern = has_glob_pattern or "*" in entry

    # When called by subdir_glob, we pass each entry individually, meaning we won't observe the
    # other entries with "*" when processing a single entry that may not have it.
    if not _called_by_subdir_glob and not has_glob_pattern:
        fail("Using a strict glob without a single '*'. Use a literal list instead of globbing.")

    return all_results

def strict_subdir_glob(include, exclude = None, prefix = ""):
    """
    A version of subdir_glob that fails if any of the glob_specs fail to match an entry after taking
    exclude into consideration.
    """
    if exclude == None:
        exclude = []

    results = []

    for dirpath, glob_pattern in include:
        results.append(
            _single_subdir_glob(dirpath, glob_pattern, exclude, prefix, strict = True),
        )

    return _merge_maps(*results)

glob = native.glob
