// Read-only single-owner Gemma decode diagnosis; never used for timing.
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {
int next_index = 0;
int pending_index = -1;

const char* directory() {
  return std::getenv("ET_DIAGNOSTIC_ATTENTION_OUTPUT");
}

void checked(cudaError_t error) {
  if (error != cudaSuccess) std::abort();
}

int integer(void** args, int index) {
  return *static_cast<int*>(args[index]);
}

char* pointer(void** args, int index) {
  return *static_cast<char**>(args[index]);
}

void write(const std::string& name, const char* data, size_t bytes) {
  std::ofstream stream(std::string(directory()) + "/" + name, std::ios::binary);
  stream.write(data, bytes);
  if (!stream.good()) std::abort();
}

void copy(const std::string& name, const void* data, size_t bytes) {
  std::vector<char> host(bytes);
  checked(cudaMemcpy(host.data(), data, bytes, cudaMemcpyDeviceToHost));
  write(name, host.data(), host.size());
}
} // namespace

extern "C" CUresult CUDAAPI cuLaunchKernel(
    CUfunction function, unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz, unsigned shared,
    CUstream stream, void** args, void** extra) {
  using Launch = CUresult(CUDAAPI*)(CUfunction, unsigned, unsigned, unsigned,
      unsigned, unsigned, unsigned, unsigned, CUstream, void**, void**);
  static auto real = reinterpret_cast<Launch>(dlsym(RTLD_NEXT, "cuLaunchKernel"));
  if (!real) std::abort();
  const char* name = nullptr;
  if (directory() && args && cuFuncGetName(&name, function) == CUDA_SUCCESS && name) {
    const bool ring = std::strcmp(name, "_decode_reference_order") == 0;
    const bool baseline = std::strcmp(name, "_sdpa_varlen_decode_kernel") == 0;
    if ((ring || baseline) && gy == 8) {
      if (gx != 1 || gz != 12) std::abort();
      checked(cudaStreamSynchronize(stream));
      const int h = ring ? 14 : 8;
      const int heads = integer(args, h);
      const int qh = integer(args, h + 1);
      const int qt = integer(args, h + 2);
      if (heads != 16 || qh != 256) std::abort();
      std::vector<int> info(162 * 7);
      checked(cudaMemcpy(info.data(), pointer(args, ring ? 13 : 7),
                         info.size() * sizeof(int), cudaMemcpyDeviceToHost));
      const int end = info[3], columns = info[4], column = info[2];
      if (info[0] != 0 || info[1] != 1 || end < 1 || end > 8192 ||
          columns != 2 || column != 0) std::abort();
      for (int segment = 1; segment < 162; ++segment) {
        if (info[segment * 7 + 1] != 0) std::abort();
      }
      const int id = next_index++;
      pending_index = id;
      const std::string prefix = std::to_string(id) + "-";
      copy(prefix + "Q.bf16", pointer(args, 0), 16 * 256 * 2);
      write(prefix + "info.i32", reinterpret_cast<char*>(info.data()),
            info.size() * sizeof(int));
      const int first = std::max(0, end - 1024);
      for (int kind = 0; kind < 2; ++kind) {
        const int column_stride = integer(args, ring ? (kind ? 21 : 18) : (kind ? 15 : 12));
        const int head_stride = integer(args, ring ? (kind ? 22 : 19) : (kind ? 16 : 13));
        if (column_stride != 2048 || head_stride != 256) std::abort();
        char* history = pointer(args, kind ? 2 : 1);
        std::vector<char> host((end - first) * 4096);
        for (int position = first; position < end; ++position) {
          if (ring && position == end - 1) {
            char* current = pointer(args, kind ? 4 : 3);
            const int stride = integer(args, kind ? 7 : 5);
            for (int head = 0; head < 8; ++head) {
              checked(cudaMemcpy(host.data() + (position - first) * 4096 + head * 512,
                                 current + head * stride * 2, 512, cudaMemcpyDeviceToHost));
            }
          } else {
            const int row = position % (ring ? 1024 : 3071);
            const char* input = history + (row * columns + column) * column_stride * 2;
            checked(cudaMemcpy(host.data() + (position - first) * 4096,
                               input, 4096, cudaMemcpyDeviceToHost));
          }
        }
        write(prefix + (kind ? "V.bf16" : "K.bf16"), host.data(), host.size());
      }
      std::ofstream metadata(std::string(directory()) + "/" + prefix + "metadata.json");
      metadata << "{\"kernel\":\"" << name << "\",\"index\":" << id
               << ",\"logical_end\":" << end << ",\"first_position\":" << first
               << ",\"query_head_stride\":" << qh << ",\"query_token_stride\":" << qt
               << ",\"columns\":" << columns << ",\"splits\":" << gz << "}\n";
      if (!metadata.good()) std::abort();
    }
  }
  const CUresult result = real(function, gx, gy, gz, bx, by, bz, shared, stream, args, extra);
  if (result == CUDA_SUCCESS && directory() && args && name && pending_index >= 0 &&
      (std::strcmp(name, "_reduce") == 0 ||
       std::strcmp(name, "_sdpa_varlen_reduce_kernel") == 0)) {
    checked(cudaStreamSynchronize(stream));
    copy(std::to_string(pending_index) + "-attention-out.bf16", pointer(args, 3), 16 * 256 * 2);
    pending_index = -1;
  }
  return result;
}
