#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/platform/runtime.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>
#include <pytorch/tokenizers/sentencepiece.h>
#include <pytorch/tokenizers/tiktoken.h>

namespace py = pybind11;
using namespace executorch::extension::llm;

PYBIND11_MODULE(runner, m) {
    m.doc() = "ExecutorTorch LLM Runner Python bindings";
    // Bind safe_printf function
    m.def("safe_printf", &safe_printf, "Prints a string safely",
          py::arg("piece"));
}
