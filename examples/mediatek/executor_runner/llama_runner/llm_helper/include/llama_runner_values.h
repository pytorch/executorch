#pragma once

namespace torch::executor {
  using llm_helper::LLMType;

  // Sizes
  const size_t PROMPT_TOKEN_BATCH_SIZE = 128;
  const size_t CACHE_SIZE = 512;
  const size_t HIDDEN_SIZE = 4096;
  const size_t NUM_HEAD = 32;
  const size_t NUM_LAYER = 32;
  const size_t MAX_TOKEN_LENGTH = 8192;
  const double ROT_EMB_BASE = 500000;

  // Types
  const LLMType MODEL_INPUT_TYPE = LLMType::FP32;
  const LLMType MODEL_OUTPUT_TYPE = LLMType::FP32;
  const LLMType CACHE_TYPE = LLMType::FP32;
  const LLMType MASK_TYPE = LLMType::FP32;
  const LLMType ROT_EMB_TYPE = LLMType::FP32;

  // Paths
  const std::string TOKENIZER_PATH="/data/local/tmp/llama3/tokenizer.model";
  const std::string TOKEN_EMBEDDING_PATH="/data/local/tmp/llama3/embedding_llama3_8b_instruct_fp32.bin";

  // Comma-Separated Paths
  const std::string PROMPT_MODEL_PATHS="\
  /data/local/tmp/llama3/llama3_8b_SC_sym4W_sym16A_4_chunks_Overall_128t512c_0.pte,\
  /data/local/tmp/llama3/llama3_8b_SC_sym4W_sym16A_4_chunks_Overall_128t512c_1.pte,\
  /data/local/tmp/llama3/llama3_8b_SC_sym4W_sym16A_4_chunks_Overall_128t512c_2.pte,\
  /data/local/tmp/llama3/llama3_8b_SC_sym4W_sym16A_4_chunks_Overall_128t512c_3.pte,";

  // Comma-Separated Paths
  const std::string GEN_MODEL_PATHS="\
  /data/local/tmp/llama3/llama3_8b_SC_sym4W_sym16A_4_chunks_Overall_1t512c_0.pte,\
  /data/local/tmp/llama3/llama3_8b_SC_sym4W_sym16A_4_chunks_Overall_1t512c_1.pte,\
  /data/local/tmp/llama3/llama3_8b_SC_sym4W_sym16A_4_chunks_Overall_1t512c_2.pte,\
  /data/local/tmp/llama3/llama3_8b_SC_sym4W_sym16A_4_chunks_Overall_1t512c_3.pte,";

} // namespace torch::executor
