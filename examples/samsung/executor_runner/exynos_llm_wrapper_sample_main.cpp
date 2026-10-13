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
 *
 * Reads prompts from stdin (one per line) and generates on the SAME handle,
 * so the model stays prepared across lines. Each line may be prefixed with
 * "IMAGE:<path>" to feed an image followed by the rest as the text prompt.
 *
 * STREAMING=1 (default): the on_token callback prints each token as generated.
 * STREAMING=0: the on_token callback appends to a std::string and the full
 * answer is printed once when generation ends. Override with -DSTREAMING=0.
 */
#include <stdio.h>
#include <string.h>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "exynos_llm_wrapper.h"

#ifndef STREAMING
#define STREAMING 1
#endif

static void on_token(const char* text, int32_t len, void* userdata) {
#if STREAMING
  (void)userdata;
  fwrite(text, 1, (size_t)len, stdout);
  fflush(stdout);
#else
  auto* acc = static_cast<std::string*>(userdata);
  acc->append(text, (size_t)len);
#endif
}

static int generate_one(enn_runner* r, const char* line) {
  const char* img = NULL;
  const char* prompt = line;
  if (strncmp(line, "IMAGE:", 6) == 0) {
    img = line + 6;
    prompt = "Describe this image.";
  }
  printf("----- generate (image=%s) -----\n", img ? img : "none");

#if STREAMING
  int rc = enn_llm_generate(
      r, prompt, img, NULL, 196, 0.0f, 0, 0, 0, on_token, NULL);
#else
  std::string acc;
  int rc = enn_llm_generate(
      r, prompt, img, NULL, 196, 0.0f, 0, 0, 0, on_token, &acc);
  printf("%s\n", acc.c_str());
  fflush(stdout);
#endif
  printf("\n[rc=%d]\n", rc);
  return rc;
}

int main(int argc, char** argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: %s model ptd tok pos cfg [vision]\n", argv[0]);
    fprintf(
        stderr, "then read prompts from stdin; 'IMAGE:<path>' for an image.\n");
    return 2;
  }
  const char* vision = (argc > 6 && argv[6][0]) ? argv[6] : NULL;

  printf("[sample] creating runner...\n");
  enn_runner* r =
      enn_llm_create(argv[1], argv[2], argv[3], argv[4], argv[5], vision);
  if (!r) {
    fprintf(stderr, "enn_llm_create FAILED\n");
    return 1;
  }
  printf("[sample] created. enter prompts (Ctrl-D to end):\n");

  // Stopper thread that invokes enn_llm_stop to imitate a user interrupt.
  std::thread stopper_thread([r]() {
    std::cout << "[Stopper Thread] Waiting 3 seconds before interrupting..."
              << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(3));
    std::cout << "[Stopper Thread] Triggering enn_llm_stop()..." << std::endl;
    enn_llm_stop(r);
  });

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
    rc_total |= generate_one(r, buf);
  }

  if (stopper_thread.joinable()) {
    stopper_thread.join();
  }

  printf("[sample] destroying runner...\n");
  enn_llm_destroy(r);
  return rc_total ? 1 : 0;
}
