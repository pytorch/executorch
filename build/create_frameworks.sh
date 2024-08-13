#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

function usage() {
    echo "This script creates XCFrameworks from libraries in specified directories."
    echo "It merges libraries using libtool and creates an XCFramework using xcodebuild."
    echo ""
    echo "Usage: $0 --directory=<dir> --framework=<lib> [--output=<output>]"
    echo "  --directory: Directory containing the libs"
    echo "  --framework: Framework to create in the format 'target:lib1,lib2:headers'"
    echo "               'target' is the name of the target library."
    echo "               'lib1,lib2' is a comma-separated list of input libraries."
    echo "               'headers' is an optional path to a directory with headers."
    echo "  --output: Optional output directory. Defaults to the current directory."
    echo ""
    echo "Example:"
    echo "$0 --directory=ios-arm64 --directory=ios-arm64-simulator --framework=\"mylib:lib1.a,lib2.a:include\" --output=output/dir"
    exit 1
}

command -v libtool >/dev/null 2>&1 || { echo >&2 "libtool is required but it's not installed.  Aborting."; exit 1; }
command -v xcodebuild >/dev/null 2>&1 || { echo >&2 "xcodebuild is required but it's not installed.  Aborting."; exit 1; }

directories=()
frameworks=()
output=$(pwd)

for arg in "$@"; do
    case $arg in
        -h|--help) usage ;;
        --directory=*) directories+=("${arg#*=}") ;;
        --framework=*) frameworks+=("${arg#*=}") ;;
        --output=*) output="${arg#*=}" ;;
        *)
            echo "Invalid argument: $arg"
            exit 1
        ;;
    esac
done

if [ ${#directories[@]} -eq 0 ] || [ ${#frameworks[@]} -eq 0 ]; then
    echo "Both --directory and --framework options are required"
    usage
fi

mkdir -p "${output}"

create_xcframework() {
    local target_library_name
    target_library_name=$(echo "$1" | cut -d: -f1)
    local libraries_list
    libraries_list=$(echo "$1" | cut -d: -f2 | tr ',' '\n')
    local headers_directory
    headers_directory=$(echo "$1" | cut -d: -f3)
    local dir
    local libraries=()
    local merged_libs=()

    if [[ -n "$headers_directory" && ! -d "$headers_directory" ]]; then
        echo "Headers directory ${headers_directory} does not exist"
        exit 1
    fi

    # For each directory, create a merged library using libtool.
    for dir in "${directories[@]}"; do

        if [ ! -d "${dir}" ]; then
            echo "Directory ${dir} does not exist"
            exit 1
        fi

        local dir_suffix
        dir_suffix=$(echo "$dir" | tr '[:upper:]' '[:lower:]' | sed 's/\//-/g')
        local merged_lib="${output}/lib${target_library_name}-${dir_suffix}.a"

        # Remove the existing .a file if it exists.
        if [ -f "${merged_lib}" ]; then
            echo "Removing existing file ${merged_lib}"
            rm "${merged_lib}"
        fi

        echo -e "\nMerging libraries:\n${libraries_list}\nfrom ${dir}\ninto library ${merged_lib}"

        local lib_paths=()
        for lib in ${libraries_list}; do
            if [ ! -f "${dir}/${lib}" ]; then
                echo "File ${dir}/${lib} does not exist"
                exit 1
            fi
            lib_paths+=("${dir}/${lib}")
        done

        libtool -static -o "${merged_lib}" "${lib_paths[@]}"

        merged_libs+=("${merged_lib}")

        if [[ -n "$headers_directory" ]]; then
            echo -e "\nIncluding headers from ${headers_directory}"
            libraries+=("-library" "${merged_lib}" "-headers" "${headers_directory}")
        else
            libraries+=("-library" "${merged_lib}")
        fi
    done

    # Remove the existing .xcframework if it exists.
    local xcframework="${output}/${target_library_name}.xcframework"
    if [ -d "${xcframework}" ]; then
        echo -e "\nRemoving existing XCFramework ${xcframework}"
        rm -rf "${xcframework}"
    fi

    echo -e "\nCreating XCFramework ${xcframework}"

    xcodebuild -create-xcframework "${libraries[@]}" -output "${xcframework}"

    echo -e "\nDeleting intermediate libraries:"
    for merged_lib in "${merged_libs[@]}"; do
        if [[ -f "${merged_lib}" ]]; then
            echo "Deleting ${merged_lib}"
            rm "${merged_lib}"
        fi
    done
}

# Create an XCFramework for each target library.
for target_lib in "${frameworks[@]}"; do
    create_xcframework "$target_lib"
done
