/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/qaihub_scripts/stable_diffusion/runner/runner.h>
#include <executorch/runtime/platform/runtime.h>
#include <gflags/gflags.h>

DEFINE_string(
    text_encoder_path,
    "qaihub_stable_diffusion_text_encoder.pte",
    "Text Encoder Model serialized in flatbuffer format.");
DEFINE_string(
    unet_path,
    "qaihub_stable_diffusion_unet.pte",
    "Unet Model serialized in flatbuffer format.");
DEFINE_string(
    vae_path,
    "qaihub_stable_diffusion_vae.pte",
    "Vae Model serialized in flatbuffer format.");
DEFINE_string(
    output_folder_path,
    "outputs",
    "Executorch inference data output path.");
DEFINE_string(
    input_list_path,
    "input_list.txt",
    "Input list storing time embedding.");
DEFINE_string(
    vocab_json,
    "vocab.json",
    "Json path to retrieve a list of vocabs.");
DEFINE_string(
    prompt,
    "a photo of an astronaut riding a horse on mars",
    "User input prompt");
DEFINE_int32(num_time_steps, 20, "Number of time steps.");
DEFINE_double(guidance_scale, 7.5, "Guidance Scale");

DEFINE_double(text_encoder_output_scale, 0.0, "Text encoder output scale");
DEFINE_int32(text_encoder_output_offset, 0, "Text encoder output offset");
DEFINE_double(unet_input_latent_scale, 0.0, "Unet input latent scale");
DEFINE_int32(unet_input_latent_offset, 0, "Unet input latent offset");
DEFINE_double(unet_input_text_emb_scale, 0.0, "Unet input text emb scale");
DEFINE_int32(unet_input_text_emb_offset, 0, "Unet input text emb offset");
DEFINE_double(unet_output_scale, 0.0, "Unet output scale");
DEFINE_int32(unet_output_offset, 0, "Unet output offset");
DEFINE_double(vae_input_scale, 0.0, "Vae input scale");
DEFINE_int32(vae_input_offset, 0, "Vae input offset");
DEFINE_double(vae_output_scale, 0.0, "Vae output scale");
DEFINE_int32(vae_output_offset, 0, "Vae output offset");
DEFINE_bool(
    fix_latents,
    false,
    "Enable this option to fix the latents in the unet diffuse step.");

void usage_message() {
  std::string usage_message =
      "This is a sample executor runner capable of executing stable diffusion models."
      "Users will need binary .pte program files for text_encoder, unet, and vae. Below are the options to retrieve required .pte program files:\n"
      "For further information on how to generate the .pte program files and example command to execute this runner, please refer to qaihub_stable_diffsion.py.";
  gflags::SetUsageMessage(usage_message);
}

using executorch::runtime::Error;

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();
  usage_message();
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  bool is_default =
      gflags::GetCommandLineFlagInfoOrDie("text_encoder_output_scale")
          .is_default ||
      gflags::GetCommandLineFlagInfoOrDie("text_encoder_output_offset")
          .is_default ||
      gflags::GetCommandLineFlagInfoOrDie("unet_input_latent_scale")
          .is_default ||
      gflags::GetCommandLineFlagInfoOrDie("unet_input_latent_offset")
          .is_default ||
      gflags::GetCommandLineFlagInfoOrDie("unet_input_text_emb_scale")
          .is_default ||
      gflags::GetCommandLineFlagInfoOrDie("unet_input_text_emb_offset")
          .is_default ||
      gflags::GetCommandLineFlagInfoOrDie("unet_output_scale").is_default ||
      gflags::GetCommandLineFlagInfoOrDie("unet_output_offset").is_default ||
      gflags::GetCommandLineFlagInfoOrDie("vae_input_scale").is_default ||
      gflags::GetCommandLineFlagInfoOrDie("vae_input_offset").is_default ||
      gflags::GetCommandLineFlagInfoOrDie("vae_output_scale").is_default ||
      gflags::GetCommandLineFlagInfoOrDie("vae_output_offset").is_default;

  ET_CHECK_MSG(
      !is_default,
      "Please provide scale and offset for unet latent input, unet output, and vae input/output."
      "Please refer to qaihub_stable_diffusion.py if you are unsure how to retrieve these values.");

  ET_LOG(Info, "Stable Diffusion runner started");
  std::vector<std::string> models_path = {
      FLAGS_text_encoder_path, FLAGS_unet_path, FLAGS_vae_path};

  // Create stable_diffusion_runner
  example::Runner runner(
      models_path,
      FLAGS_num_time_steps,
      FLAGS_guidance_scale,
      FLAGS_text_encoder_output_scale,
      FLAGS_text_encoder_output_offset,
      FLAGS_unet_input_latent_scale,
      FLAGS_unet_input_latent_offset,
      FLAGS_unet_input_text_emb_scale,
      FLAGS_unet_input_text_emb_offset,
      FLAGS_unet_output_scale,
      FLAGS_unet_output_offset,
      FLAGS_vae_input_scale,
      FLAGS_vae_input_offset,
      FLAGS_vae_output_scale,
      FLAGS_vae_output_offset,
      FLAGS_output_folder_path,
      FLAGS_fix_latents);

  ET_CHECK_MSG(
      runner.init_tokenizer(FLAGS_vocab_json) == Error::Ok,
      "Runner failed to init tokenizer");

  ET_CHECK_MSG(runner.load() == Error::Ok, "Runner failed to load method");

  ET_CHECK_MSG(
      runner.parse_input_list(FLAGS_input_list_path) == Error::Ok,
      "Failed to parse time embedding input list");
  ET_CHECK_MSG(
      runner.generate(FLAGS_prompt) == Error::Ok, "Runner failed to generate");

  ET_CHECK_MSG(
      runner.print_performance() == Error::Ok,
      "Runner failed to print performance");

  return 0;
}
