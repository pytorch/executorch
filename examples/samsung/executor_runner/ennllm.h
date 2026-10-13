/*
 * Copyright (c) 2026 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * Public C API for the Exynos LLM runner.
 *
 * Only these symbols are exported from libennllm.so. The underlying
 * implementation (model loading, ENN backend, tokenizer, KV-cache) is kept
 * internal to the library.
 *
 * Model preparation happens once in enn_llm_create. enn_llm_generate can then
 * be called repeatedly (with or without an image) on the same handle without
 * reloading or re-preparing the models.
 */

#ifndef ENNLLM_H_
#define ENNLLM_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
#define ENN_API __attribute__((visibility("default")))
#else
#define ENN_API
#endif

/** Values returned by the API functions. 0 = OK, non-zero = error. */
enum {
  ENN_OK = 0,
  ENN_ERR_INVALID_ARG = 1,
  ENN_ERR_LOAD = 2,
  ENN_ERR_GENERATE = 3,
};

/**
 * Callback invoked with each decoded text piece as tokens are generated.
 * text points to len bytes of UTF-8 text (may not be NUL-terminated); copy it
 * if you keep it beyond the call. Called on the same thread that calls
 * enn_llm_generate; must not block.
 */
typedef void (*enn_token_cb)(const char* text, int32_t len, void* userdata);

typedef struct enn_runner enn_runner;

/**
 * Load and prepare the model(s). Does all one-time work: decoder .pte/.ptd,
 * tokenizer, positional embedding, optional vision encoder, and KV-cache
 * allocation. Repeated calls are not allowed on the same handle.
 *
 * @param model_path Decoder .pte file (required).
 * @param data_path Decoder .ptd weights, or NULL if none.
 * @param tokenizer_path Tokenizer model file (required).
 * @param pos_embedding_path Precomputed rotary embeddings, or NULL.
 * @param config_path Runner config file (required).
 * @param vision_model_path Vision encoder .pte, or NULL => text-only runner.
 * @return A handle on success, NULL on failure.
 */
ENN_API enn_runner* enn_llm_create(
    const char* model_path,
    const char* data_path,
    const char* tokenizer_path,
    const char* pos_embedding_path,
    const char* config_path,
    const char* vision_model_path);

/** Destroy the runner and release all resources. Safe to call with NULL. */
ENN_API void enn_llm_destroy(enn_runner* runner);

/**
 * Generate a response for prompt (optionally with an image). The models stay
 * prepared across calls; KV-cache is reset before each generation so every
 * prompt starts fresh.
 *
 * @param runner Handle from enn_llm_create.
 * @param prompt User prompt text (required, non-empty).
 * @param image_path Image file for this turn, or NULL => text-only.
 * @param prompt_before_image Text placed before the image in the user turn
 *   (Gemma 4 multimodal); ignored when image_path is NULL.
 * @param seq_len Total token budget (prompt + generated), capped by the model's
 *   max context length; -1 => use the model's max context length.
 * @param temperature Sampling temperature.
 * @param num_bos BOS tokens to prepend to the prompt.
 * @param num_eos EOS tokens to append to the prompt.
 * @param verbose Non-zero also streams tokens to stdout.
 * @param on_token Per-piece callback, or NULL to discard output.
 * @param userdata Opaque pointer passed through to on_token.
 * @return ENN_OK on success, an ENN_ERR_* code otherwise.
 */
ENN_API int32_t enn_llm_generate(
    enn_runner* runner,
    const char* prompt,
    const char* image_path,
    const char* prompt_before_image,
    int32_t seq_len,
    float temperature,
    int32_t num_bos,
    int32_t num_eos,
    int32_t verbose,
    enn_token_cb on_token,
    void* userdata);

#ifdef __cplusplus
}
#endif

#endif // ENNLLM_H_
