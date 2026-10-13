/*
 * Copyright (c) 2026 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * Sample CLI consuming the public exynos_llm_wrapper C API
 * (exynos_llm_wrapper.h). This mirrors how an Android app drives
 * libexynos_llm_wrapper.so: create once, generate repeatedly, destroy at
 * shutdown. Models stay prepared across generate() calls.
 *
 * Usage (paths are on-device):
 *   exynos_llm_wrapper_sample <model> <ptd> <tok> <pos> <cfg> [vision]
 *                             [max_output_tokens]
 *
 * Reads prompts from stdin (one per line) and generates on the SAME handle,
 * so the model stays prepared across lines. Each line may be prefixed with
 * "IMAGE:<path>" to feed an image followed by the rest as the text prompt.
 *
 * STREAMING=1 (default): the on_token callback prints each token as generated.
 * STREAMING=0: the on_token callback appends to a std::string and the full
 * answer is printed once when generation ends. Override with -DSTREAMING=0.
 *
 * Stop test: pass max_output_tokens=N (> 0; use "" for vision to skip it) to
 * call enn_llm_stop() right after the N-th output token of each prompt and
 * check that no more tokens arrive. Use prompts whose answer is longer than N.
 * Without it, a stopper thread calls enn_llm_stop() once after 3 seconds.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "exynos_llm_wrapper.h"

#ifndef STREAMING
#define STREAMING 1
#endif

struct GenContext {
  enn_runner* runner;
  int32_t max_output_tokens;
  int32_t num_pieces = 0;
  int32_t num_tokens = 0;
  int32_t tokens_after_stop = 0;
  std::string acc;
};

static void
count_for_stop_test(GenContext* ctx, const char* text, int32_t len) {
  // The first callback is the echoed input prompt, not an output token.
  if (ctx->num_pieces++ == 0) {
    return;
  }
  // The runner emits a trailing "\n" after generation ends; ignore it.
  if (ctx->num_tokens >= ctx->max_output_tokens) {
    if (!(len == 1 && text[0] == '\n')) {
      ++ctx->tokens_after_stop;
    }
    return;
  }
  if (++ctx->num_tokens == ctx->max_output_tokens) {
    enn_llm_stop(ctx->runner);
    printf("\n[STOP TEST] enn_llm_stop called at token %d\n", ctx->num_tokens);
    fflush(stdout);
  }
}

static void on_token(const char* text, int32_t len, void* userdata) {
  auto* ctx = static_cast<GenContext*>(userdata);
#if STREAMING
  fwrite(text, 1, (size_t)len, stdout);
  fflush(stdout);
#else
  ctx->acc.append(text, (size_t)len);
#endif
  if (ctx->max_output_tokens > 0) {
    count_for_stop_test(ctx, text, len);
  }
}

static int
generate_one(enn_runner* r, const char* line, int32_t max_output_tokens) {
  const char* img = NULL;
  const char* prompt = line;
  if (strncmp(line, "IMAGE:", 6) == 0) {
    img = line + 6;
    prompt = "Describe this image.";
  }
  printf("----- generate (image=%s) -----\n", img ? img : "none");

  GenContext ctx{r, max_output_tokens};
  int rc = enn_llm_generate(
      r,
      prompt,
      NULL,
      img,
      NULL,
      196,
      0.0f,
      0,
      0,
      /*echo=*/1,
      0,
      on_token,
      &ctx);
#if !STREAMING
  printf("%s\n", ctx.acc.c_str());
  fflush(stdout);
#endif
  printf("\n[rc=%d]\n", rc);
  if (rc == ENN_OK && max_output_tokens > 0) {
    bool pass =
        ctx.num_tokens == max_output_tokens && ctx.tokens_after_stop == 0;
    printf(
        "[STOP TEST] %s: requested stop at %d, received %d, after stop %d\n",
        pass ? "PASS" : "FAIL",
        max_output_tokens,
        ctx.num_tokens,
        ctx.tokens_after_stop);
    if (!pass) {
      rc = ENN_ERR_GENERATE;
    }
  }
  return rc;
}

int main(int argc, char** argv) {
  if (argc < 6) {
    fprintf(
        stderr,
        "usage: %s model ptd tok pos cfg [vision] [max_output_tokens]\n",
        argv[0]);
    fprintf(
        stderr, "then read prompts from stdin; 'IMAGE:<path>' for an image.\n");
    fprintf(
        stderr,
        "max_output_tokens=N (> 0): stop test, call enn_llm_stop() right after "
        "the N-th output token and check nothing more arrives.\n");
    return 2;
  }
  const char* vision = (argc > 6 && argv[6][0]) ? argv[6] : NULL;
  const int32_t max_output_tokens = argc > 7 ? atoi(argv[7]) : 0;

  printf("[sample] creating runner...\n");
  enn_runner* r =
      enn_llm_create(argv[1], argv[2], argv[3], argv[4], argv[5], vision, 0);
  if (!r) {
    fprintf(stderr, "enn_llm_create FAILED\n");
    return 1;
  }
  printf("[sample] created. enter prompts (Ctrl-D to end):\n");

  // Stopper thread that invokes enn_llm_stop to imitate a user interrupt.
  // Replaced by the token-count stop test when max_output_tokens is given.
  std::thread stopper_thread;
  if (max_output_tokens <= 0) {
    stopper_thread = std::thread([r]() {
      std::cout << "[Stopper Thread] Waiting 3 seconds before interrupting..."
                << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(3));
      std::cout << "[Stopper Thread] Triggering enn_llm_stop()..." << std::endl;
      enn_llm_stop(r);
    });
  }

  std::cout << "[CLI] Starting token generation loop..." << std::endl;
  int rc_total = 0;
  char buf[4096];
  while (fgets(buf, sizeof(buf), stdin)) {
    size_t n = strlen(buf);
    while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) {
      buf[--n] = '\0';
    }
    if (n == 0) {
      continue;
    }
    rc_total |= generate_one(r, buf, max_output_tokens);
  }

  if (stopper_thread.joinable()) {
    stopper_thread.join();
  }

  printf("[sample] destroying runner...\n");
  enn_llm_destroy(r);
  return rc_total ? 1 : 0;
}
