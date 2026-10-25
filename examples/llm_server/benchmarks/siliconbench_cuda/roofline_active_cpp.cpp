/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "roofline_active_common.h"

#include <llama.h>

int main(int argc, char** argv) {
  try {
    roofline::require(argc == 4, "Usage: roofline_cpp model.gguf output.json repetitions");
    int repetitions = std::stoi(argv[3]);
    roofline::require(repetitions >= 4, "At least four repetitions required");
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = -1;
    auto* model = llama_model_load_from_file(argv[1], mp);
    roofline::require(model != nullptr, "llama model load failed");
    int vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(model));
    roofline::require(vocabulary > 190256, "Unexpected MG vocabulary");
    std::vector<llama_sampler_seq_config> sampling;
    for (int owner = 0; owner < roofline::kOwners; ++owner) {
      auto* chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
      llama_sampler_chain_add(chain, llama_sampler_init_greedy());
      sampling.push_back({owner, chain});
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = roofline::kCapacity * roofline::kOwners;
    cp.n_seq_max = roofline::kOwners;
    cp.n_batch = roofline::kBatch;
    cp.n_ubatch = roofline::kBatch;
    cp.n_outputs_max = roofline::kOwners;
    cp.n_outputs_max_per_seq = 1;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_BF16;
    cp.type_v = GGML_TYPE_BF16;
    cp.offload_kqv = true;
    cp.op_offload = true;
    cp.kv_unified = true;
    cp.samplers = sampling.data();
    cp.n_samplers = sampling.size();
    auto* ctx = llama_init_from_model(model, cp);
    roofline::require(ctx != nullptr, "llama context initialization failed");
    roofline::require(llama_n_seq_max(ctx) >= roofline::kOwners,
                      "llama context cannot hold 32 owners");
    auto batch = llama_batch_init(roofline::kBatch, 0, 1);
    auto run = [&](const std::vector<roofline::Row>& input, int outputs) {
      batch.n_tokens = input.size();
      for (std::size_t i = 0; i < input.size(); ++i) {
        batch.token[i] = input[i].token;
        batch.pos[i] = input[i].position;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = input[i].owner;
        batch.logits[i] = input[i].output;
      }
      roofline::require(llama_decode(ctx, batch) == 0, "llama_decode failed");
      llama_synchronize(ctx);
      std::vector<std::int64_t> ids;
      for (int i = 0; i < outputs; ++i) {
        // Official backend sampling API returns token IDs, avoiding full
        // vocabulary logits D2H transfer in the timed forward.
        auto id = llama_get_sampled_token_ith(ctx, i);
        roofline::require(id >= 0 && id < vocabulary,
                          "CUDA backend greedy sampling unavailable or invalid");
        ids.push_back(id);
      }
      return ids;
    };
    auto memory = llama_get_memory(ctx);
    int active_owners = 0;
    auto setup = [&](const roofline::Mix& mix) {
      // Public API clears prior history metadata and KV data, outside timing.
      llama_memory_clear(memory, true);
      active_owners = mix.decode + (mix.prefill ? 1 : 0);
      for (int owner = 0; owner < active_owners; ++owner) {
        for (int start = 0; start < roofline::kHistory; start += 512) {
          std::vector<roofline::Row> input;
          for (int pos = start; pos < start + 512; ++pos)
            input.push_back({roofline::token(owner, pos), pos, owner, false});
          run(input, 0);
        }
      }
      for (int owner = active_owners; owner < roofline::kOwners; ++owner)
        roofline::require(llama_memory_seq_pos_max(memory, owner) == -1, "Idle history was retained");
      roofline::sync();
    };
    auto rewind = [&] {
      for (int owner = 0; owner < active_owners; ++owner) {
        roofline::require(llama_memory_seq_rm(memory, owner, roofline::kHistory, -1),
                          "llama tail rewind failed");
        roofline::require(llama_memory_seq_pos_max(memory, owner) == roofline::kHistory - 1,
                          "llama context before step changed");
      }
    };
    std::ofstream out(argv[2]);
    roofline::require(out.good(), "Output file cannot be opened");
    out << "{\"framework\":\"llama.cpp\",\"runtime_api\":\"llama_decode\","
        << "\"timing\":\"synchronized API forward plus device greedy selected-token return\","
        << "\"history_tokens\":1024,\"maximum_independent_owners\":32,\"history_population\":\"only-current-mix; reset-between-cases\","
        << "\"runtime_context_capacity_per_owner\":2048,"
        << "\"KV_dtype\":\"BF16\",\"prefix_reuse\":false,"
        << "\"n_batch\":1024,\"n_ubatch\":1024,\"flash_attention\":true,"
        << "\"kv_unified\":true,\"CUDA_backend_greedy_required\":true,"
        << "\"native_runtime_source_modified\":false,"
        << "\"synthetic_input_definition\":\"roofline_active_common.h::token\","
        << "\"actual_total_context_capacity\":" << llama_n_ctx(ctx) << ',';
    auto evidence = [&](std::ostream& stream, const roofline::Mix& mix, int calls) {
      for (int owner = 0; owner < mix.decode; ++owner) {
        roofline::require(llama_memory_seq_pos_max(memory, owner) == roofline::kHistory,
                          "llama decoder history did not advance exactly one token");
      }
      if (mix.prefill) {
        roofline::require(llama_memory_seq_pos_max(memory, mix.decode) ==
                              roofline::kHistory + mix.prefill - 1,
                          "llama prefill history did not advance by the chunk length");
      }
      stream << "\"public_API_forward_calls\":" << calls
             << ",\"requested_logit_rows\":" << mix.decode
             << ",\"physical_ubatch_partition\":\"native; inspect trace/log, not asserted single\"";
    };
    roofline::measure(out, repetitions, run, rewind, evidence, setup);
    out << ",\"completed\":true}\n";
    out.close();
    llama_batch_free(batch);
    llama_free(ctx);
    for (auto& config : sampling) llama_sampler_free(config.sampler);
    llama_model_free(model);
    llama_backend_free();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAILED " << e.what() << '\n';
    return 1;
  }
}
