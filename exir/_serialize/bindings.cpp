/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <flatbuffers/flatc.h> // @manual=fbsource//third-party/flatbuffers:flatc_library
#include <flatbuffers/idl.h> // @manual=fbsource//third-party/flatbuffers:flatc_library
#include <pybind11/pybind11.h> // @manual=fbsource//third-party/pybind11:pybind11
#include <pybind11/stl.h> // @manual=fbsource//third-party/pybind11:pybind11

namespace exir {
namespace {
void Warn(
    const flatbuffers::FlatCompiler* /* flatc */,
    const std::string& warn,
    bool /* show_exe_name */) {
  printf("flatc compiler warning: %s\n", warn.c_str());
}

void Error(
    const flatbuffers::FlatCompiler* /* flatc */,
    const std::string& err,
    bool /* usage */,
    bool /* show_exe_name */) {
  throw std::runtime_error("Caught error in flatc compiler: " + err);
}

} // namespace

PYBIND11_MODULE(_bindings, m) {
  m.def(
       "flatc_compile",
       [&](const std::string& outputPath,
           const std::string& schemaPath,
           const std::string& jsonPath) {
         static const flatbuffers::FlatCompiler::Generator generators[] = {
             {flatbuffers::GenerateBinary,
              "-b",
              "--binary",
              "binary",
              false,
              nullptr,
              flatbuffers::IDLOptions::kBinary,
              "Generate wire format binaries for any data definitions",
              flatbuffers::BinaryMakeRule}};

         flatbuffers::FlatCompiler::InitParams params;
         params.generators = generators;
         params.num_generators = sizeof(generators) / sizeof(generators[0]);
         params.warn_fn = Warn;
         params.error_fn = Error;

         flatbuffers::FlatCompiler flatc(params);
         std::array<const char*, 5> argv = {
             "--binary",
             "-o",
             outputPath.c_str(),
             schemaPath.c_str(),
             jsonPath.c_str()};
         return flatc.Compile(argv.size(), argv.data());
       })
      .def(
          "flatc_decompile",
          [&](const std::string& outputPath,
              const std::string& schemaPath,
              const std::string& binPath) {
            static const flatbuffers::FlatCompiler::Generator generators[] = {
                {flatbuffers::GenerateTextFile,
                 "-t",
                 "--json",
                 "text",
                 false,
                 nullptr,
                 flatbuffers::IDLOptions::kJson,
                 "Generate text output for any data definitions",
                 flatbuffers::TextMakeRule}};

            flatbuffers::FlatCompiler::InitParams params;
            params.generators = generators;
            params.num_generators = sizeof(generators) / sizeof(generators[0]);
            params.warn_fn = Warn;
            params.error_fn = Error;

            flatbuffers::FlatCompiler flatc(params);

            std::array<const char*, 8> argv = {
                "--json",
                "--defaults-json",
                "--strict-json",
                "-o",
                outputPath.c_str(),
                schemaPath.c_str(),
                "--",
                binPath.c_str()};
            return flatc.Compile(argv.size(), argv.data());
          });
}

} // namespace exir
