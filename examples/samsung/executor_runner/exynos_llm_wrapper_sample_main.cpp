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
 *     [--raw_prompt <str>] [--raw_prompt_file <file>]
 *     [--system_prompt <str>] [--system_prompt_file <file>]
 *     [--user_prompt <str>] [--user_prompt_file <file>]
 *     [--assistant_prefill_prompt <str>] [--assistant_prefill_prompt_file
 * <file>]
 *     [--image_file <path>]
 *     [--max_output_tokens <N>] [--stop_after_sec <N>] [--echo]
 *
 * Each generation is one en-invocation:
 *   - `--raw_prompt <str>` / `--raw_prompt_file <file>` : feed verbatim
 *     (raw_prompt=1).
 *   - `--*_prompt <str>`      : the next argument IS the text itself.
 *   - `--*_prompt_file <file>`: read the file and use its contents.
 *   - `--image_file <path>` : the value IS the image path passed as image_path.
 *   The system/user/prefill texts are fed together (raw_prompt=0) as one
 *   generate call, along with any image. Any may be omitted. Repeating runs
 *   sequentially on the same handle, so the model stays prepared across calls.
 *
 * STREAMING=1 (default): the on_token callback prints each token as generated.
 * STREAMING=0: the on_token callback appends to a std::string and the full
 * answer is printed once when generation ends. Override with -DSTREAMING=0.
 *
 * Stop test (exercises enn_llm_stop(); both apply to every generation):
 *   --max_output_tokens N (> 0): call enn_llm_stop() right after the N-th
 *     output token of each generation and check that no more tokens arrive.
 *     Use prompts whose answer is longer than N. Prints PASS/FAIL per run.
 *   --stop_after_sec N (> 0): spawn a stopper thread that calls enn_llm_stop()
 *     N seconds into each generation, imitating a user interrupt.
 */
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

#include "exynos_llm_wrapper.h"

#ifndef STREAMING
#define STREAMING 1
#endif

// Option codes for the long-option table (start past ASCII so getopt_long
// distinguishes them from any single-char option).
enum {
  OPT_RAW = 256,
  OPT_RAW_FILE,
  OPT_SYS,
  OPT_SYS_FILE,
  OPT_USR,
  OPT_USR_FILE,
  OPT_PRE,
  OPT_PRE_FILE,
  OPT_IMG_FILE,
  OPT_MAX_TOKENS,
  OPT_STOP_SEC,
  OPT_ECHO,
};

struct GenContext {
  enn_runner* runner;
  int32_t max_output_tokens;
  bool echo;
  int32_t num_pieces = 0;
  int32_t num_tokens = 0;
  int32_t tokens_after_stop = 0;
  std::string acc;
};

static void
count_for_stop_test(GenContext* ctx, const char* text, int32_t len) {
  // When echo is on, the first callback is the echoed input prompt, not an
  // output token, so skip it. With echo off every piece is an output token.
  if (ctx->echo && ctx->num_pieces++ == 0) {
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

static bool read_file(const char* path, std::string* out) {
  std::ifstream f(path);
  if (!f.is_open()) {
    fprintf(stderr, "failed to open %s\n", path);
    return false;
  }
  std::stringstream ss;
  ss << f.rdbuf();
  *out = ss.str();
  return true;
}

static int run_one(
    enn_runner* r,
    const std::string& raw,
    const std::string& sys,
    const std::string& usr,
    const std::string& pre,
    const std::string& image,
    int32_t max_output_tokens,
    int32_t stop_after_sec,
    bool echo) {
  int32_t raw_prompt = !raw.empty();
  const char* prompt = raw_prompt ? raw.c_str() : usr.c_str();
  // raw prompts already contain the leading <bos> token; the templated path
  // must add it so both modes feed the model the same prompt.
  int32_t num_bos = raw_prompt ? 0 : 1;

  GenContext ctx{r, max_output_tokens, echo};

  // Timer-based stopper thread imitating a user interrupt. Replaced by the
  // token-count stop test when max_output_tokens is given.
  std::thread stopper;
  if (stop_after_sec > 0) {
    stopper = std::thread([r, stop_after_sec]() {
      std::this_thread::sleep_for(std::chrono::seconds(stop_after_sec));
      printf(
          "\n[Stopper] triggering enn_llm_stop() after %d s\n", stop_after_sec);
      fflush(stdout);
      enn_llm_stop(r);
    });
  }

  int rc = enn_llm_generate(
      r,
      prompt,
      raw_prompt ? NULL : (sys.empty() ? NULL : sys.c_str()),
      raw_prompt ? NULL : (pre.empty() ? NULL : pre.c_str()),
      image.empty() ? NULL : image.c_str(),
      NULL,
      raw_prompt,
      3840,
      0.0f,
      num_bos,
      0,
      echo ? 1 : 0,
      0,
      on_token,
      &ctx);

  if (stopper.joinable()) {
    stopper.join();
  }

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
        "usage: %s model ptd tok pos cfg [vision] "
        "[--raw_prompt STR | --raw_prompt_file F | "
        "--system_prompt STR | --system_prompt_file F | "
        "--user_prompt STR | --user_prompt_file F | "
        "--assistant_prefill_prompt STR | --assistant_prefill_prompt_file F | "
        "--image_file PATH]* "
        "[--max_output_tokens N] [--stop_after_sec N] [--echo]\n",
        argv[0]);
    return 2;
  }
  const char* vision =
      (argc > 6 && argv[6][0] && argv[6][0] != '-') ? argv[6] : NULL;
  int idx = (argc > 6 && argv[6][0] && argv[6][0] != '-') ? 7 : 6;

  static const struct option long_opts[] = {
      {"raw_prompt", required_argument, 0, OPT_RAW},
      {"raw_prompt_file", required_argument, 0, OPT_RAW_FILE},
      {"system_prompt", required_argument, 0, OPT_SYS},
      {"system_prompt_file", required_argument, 0, OPT_SYS_FILE},
      {"user_prompt", required_argument, 0, OPT_USR},
      {"user_prompt_file", required_argument, 0, OPT_USR_FILE},
      {"assistant_prefill_prompt", required_argument, 0, OPT_PRE},
      {"assistant_prefill_prompt_file", required_argument, 0, OPT_PRE_FILE},
      {"image_file", required_argument, 0, OPT_IMG_FILE},
      {"max_output_tokens", required_argument, 0, OPT_MAX_TOKENS},
      {"stop_after_sec", required_argument, 0, OPT_STOP_SEC},
      {"echo", no_argument, 0, OPT_ECHO},
      {0, 0, 0, 0},
  };

  printf("[sample] creating runner...\n");
  enn_runner* r =
      enn_llm_create(argv[1], argv[2], argv[3], argv[4], argv[5], vision, 0);
  if (!r) {
    fprintf(stderr, "enn_llm_create FAILED\n");
    return 1;
  }

  // Options accumulate into one generation spec. A spec is either a raw
  // (standalone) or the system/user/prefill triple combined into ONE generate
  // call; each may come as a bare string or a file. Flush the accumulated spec
  // only when a new raw begins (a raw group never shares a call with others).
  // --max_output_tokens / --stop_after_sec / --echo are global and apply to
  // every generation.
  std::string raw, sys, usr, pre, image;
  bool have = false;
  int rc_total = 0;
  int32_t max_output_tokens = 0;
  int32_t stop_after_sec = 0;
  bool echo = false;

  optind = idx; // resume scanning after the positional model arguments
  int opt;
  while ((opt = getopt_long(argc, argv, "", long_opts, NULL)) != -1) {
    const bool is_file = opt == OPT_RAW_FILE || opt == OPT_SYS_FILE ||
        opt == OPT_USR_FILE || opt == OPT_PRE_FILE;
    std::string content;
    if (is_file) {
      if (!read_file(optarg, &content)) {
        rc_total = 1;
        break;
      }
    } else if (opt != OPT_ECHO) {
      content = optarg;
    }

    switch (opt) {
      case OPT_RAW:
      case OPT_RAW_FILE:
        if (have) {
          rc_total |= run_one(
              r,
              raw,
              sys,
              usr,
              pre,
              image,
              max_output_tokens,
              stop_after_sec,
              echo);
          raw.clear();
          sys.clear();
          usr.clear();
          pre.clear();
          image.clear();
          have = false;
        }
        raw = content;
        have = true;
        break;
      case OPT_SYS:
      case OPT_SYS_FILE:
        sys = content;
        have = true;
        break;
      case OPT_USR:
      case OPT_USR_FILE:
        usr = content;
        have = true;
        break;
      case OPT_PRE:
      case OPT_PRE_FILE:
        pre = content;
        have = true;
        break;
      case OPT_IMG_FILE:
        image = content;
        have = true;
        break;
      case OPT_MAX_TOKENS:
        max_output_tokens = atoi(content.c_str());
        break;
      case OPT_STOP_SEC:
        stop_after_sec = atoi(content.c_str());
        break;
      case OPT_ECHO:
        echo = true;
        break;
      default:
        fprintf(stderr, "unknown argument: %s\n", argv[optind - 1]);
        rc_total = 1;
        break;
    }
    if (rc_total)
      break;
  }
  if (have && !rc_total) {
    rc_total |= run_one(
        r, raw, sys, usr, pre, image, max_output_tokens, stop_after_sec, echo);
  }

  printf("[sample] destroying runner...\n");
  enn_llm_destroy(r);
  return rc_total ? 1 : 0;
}
