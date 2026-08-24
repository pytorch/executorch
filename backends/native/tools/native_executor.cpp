// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// native_executor: a standalone harness for native-runtime graphs.
//
// Scope today: load a serialized native graph (*.nptg) and its optional
// out-of-line constant file into memory, and report what was loaded. This is
// the seed of a central tool that will grow load / inspect / benchmark /
// validate / profile subcommands. It has no ExecuTorch or torch dependency;
// until the C++ program reader lands it treats both files as raw bytes.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <executorch/backends/native/runtime/Program.h>
#include <gflags/gflags.h>

DEFINE_string(program, "", "Path to the *.nptg to load. Required.");
DEFINE_string(constants, "", "Path to the out-of-line constant file.");
DEFINE_bool(dot, false, "Emit Graphviz DOT to stdout, then exit.");
DEFINE_bool(methods, false, "Dump every deserialized method, then exit.");

namespace {

// FlatBuffer file_identifier for a native Program, at byte offset 4 (see
// native_graph.fbs: file_identifier "NPTG").
constexpr char kProgramMagic[4] = {'N', 'P', 'T', 'G'};
constexpr size_t kMagicOffset = 4;

// Read an entire file into a byte buffer; nullopt (with a message) on failure.
std::optional<std::vector<uint8_t>> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::fprintf(stderr, "error: cannot open %s\n", path.c_str());
    return std::nullopt;
  }
  const std::streamsize size = f.tellg();
  if (size < 0) {
    std::fprintf(stderr, "error: cannot size %s\n", path.c_str());
    return std::nullopt;
  }
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> buf(static_cast<size_t>(size));
  if (size > 0 && !f.read(reinterpret_cast<char*>(buf.data()), size)) {
    std::fprintf(stderr, "error: cannot read %s\n", path.c_str());
    return std::nullopt;
  }
  return buf;
}

bool has_program_magic(const std::vector<uint8_t>& buf) {
  if (buf.size() < kMagicOffset + sizeof(kProgramMagic)) {
    return false;
  }
  return std::memcmp(
             buf.data() + kMagicOffset, kProgramMagic, sizeof(kProgramMagic)) ==
      0;
}

// Emit `text` on stdout. False on a short write, so a full disk truncating a
// redirected dump is reported rather than exiting 0 on partial output.
bool write_stdout(const std::string& text) {
  return std::fwrite(text.data(), 1, text.size(), stdout) == text.size();
}

} // namespace

int main(int argc, char** argv) {
  gflags::SetUsageMessage(
      "Load a serialized native graph and report what it holds.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_program.empty()) {
    std::fprintf(stderr, "error: --program is required\n");
    return 1;
  }

  const std::optional<std::vector<uint8_t>> program = read_file(FLAGS_program);
  if (!program) {
    return 1;
  }

  // --dot: parse + verify via the program reader, emit Graphviz DOT to stdout
  // (nothing else), then exit. Pipe to a file and render with `dot -Tpng`.
  if (FLAGS_dot) {
    try {
      const ptn::Program prog =
          ptn::Program::load(program->data(), program->size());
      if (!write_stdout(prog.to_dot())) {
        std::fprintf(stderr, "error: short write emitting DOT\n");
        return 1;
      }
      return 0;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: %s\n", e.what());
      return 2;
    }
  }

  // --methods: deserialize each method into the in-memory IR and dump it.
  if (FLAGS_methods) {
    try {
      const ptn::Program prog =
          ptn::Program::load(program->data(), program->size());
      const std::vector<std::string> names = prog.method_names();
      std::printf("methods: %zu\n", names.size());
      for (const std::string& name : names) {
        const ptn::Method& method = prog.get_method(name);
        std::printf("\n===== method: %s =====\n", name.c_str());
        if (!write_stdout(method.to_string())) {
          std::fprintf(stderr, "error: short write dumping %s\n", name.c_str());
          return 1;
        }
      }
      return 0;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: %s\n", e.what());
      return 2;
    }
  }

  const bool magic_ok = has_program_magic(*program);
  std::printf("program:   %s\n", FLAGS_program.c_str());
  std::printf("  bytes:   %zu\n", program->size());
  std::printf(
      "  magic:   %s (expect 'NPTG' at offset 4)\n",
      magic_ok ? "NPTG ok" : "MISSING/BAD");

  if (!FLAGS_constants.empty()) {
    const std::optional<std::vector<uint8_t>> constants =
        read_file(FLAGS_constants);
    if (!constants) {
      return 1;
    }
    std::printf("constants: %s\n", FLAGS_constants.c_str());
    std::printf("  bytes:   %zu\n", constants->size());
    std::printf(
        "  note:    opaque bytes only (torch pickle; not parsed in "
        "standalone C++)\n");
  }

  if (!magic_ok) {
    std::fprintf(
        stderr, "error: program is not a native graph (bad NPTG magic)\n");
    return 2;
  }
  std::printf("ok: loaded into memory\n");
  return 0;
}
