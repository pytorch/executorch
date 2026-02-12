/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gflags/gflags.h>

#include "audio_stream.h"
#include "tokenizer_utils.h"
#include "types.h"
#include "video_stream.h"
#include "yolo_detector.h"

#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>

DEFINE_string(
    asr_model_path,
    "parakeet.pte",
    "Path to Parakeet ASR model (.pte).");
DEFINE_string(
    yolo_model_path,
    "yolo26n.pte",
    "Path to YOLO detection model (.pte).");
DEFINE_string(
    tokenizer_path,
    "tokenizer.model",
    "Path to SentencePiece tokenizer model file.");
DEFINE_string(
    asr_data_path,
    "",
    "Path to data file (.ptd) for ASR delegate data (optional, for CUDA).");
DEFINE_string(
    yolo_data_path,
    "",
    "Path to data file (.ptd) for YOLO delegate data (optional, for CUDA).");
DEFINE_int32(audio_device_index, -1, "Audio input device index (-1 for default).");
DEFINE_int32(video_device_index, 0, "Video input device index (0 for default).");
DEFINE_bool(list_devices, false, "List available devices and exit.");
DEFINE_double(yolo_score_threshold, 0.45, "YOLO detection score threshold.");
DEFINE_bool(
    show_video,
    true,
    "Show video window (set to false for headless mode).");

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

using ::parakeet::Token;
using ::parakeet::TokenId;

namespace {

const std::vector<int> DURATIONS = {0, 1, 2, 3, 4};

// Helper to get expected scalar type for a method input
::executorch::runtime::Result<::executorch::aten::ScalarType>
get_input_scalar_type(
    Module& model,
    const char* method_name,
    size_t input_index) {
  auto method_meta_result = model.method_meta(method_name);
  if (!method_meta_result.ok()) {
    return method_meta_result.error();
  }
  auto method_meta = method_meta_result.get();
  if (method_meta.num_inputs() <= input_index) {
    return ::executorch::runtime::Error::InvalidArgument;
  }
  auto input_meta_result = method_meta.input_tensor_meta(input_index);
  if (input_meta_result.error() != ::executorch::runtime::Error::Ok) {
    return input_meta_result.error();
  }
  return input_meta_result.get().scalar_type();
}

// Batch transcription using the same flow as main.cpp
std::vector<Token> greedy_decode(
    Module& model,
    const ::executorch::aten::Tensor& f_proj,
    int64_t encoder_len,
    int64_t blank_id,
    int64_t num_rnn_layers,
    int64_t pred_hidden) {
  std::vector<Token> hypothesis;
  size_t proj_dim = static_cast<size_t>(f_proj.sizes()[2]);

  auto h_dtype_result = get_input_scalar_type(model, "decoder_step", 1);
  auto c_dtype_result = get_input_scalar_type(model, "decoder_step", 2);
  if (!h_dtype_result.ok() || !c_dtype_result.ok()) {
    return hypothesis;
  }
  auto h_dtype = h_dtype_result.get();
  auto c_dtype = c_dtype_result.get();

  auto f_dtype_result = get_input_scalar_type(model, "joint", 0);
  auto g_dtype_result = get_input_scalar_type(model, "joint", 1);
  if (!f_dtype_result.ok() || !g_dtype_result.ok()) {
    return hypothesis;
  }
  auto f_dtype = f_dtype_result.get();
  auto g_dtype = g_dtype_result.get();

  size_t h_elem_size = ::executorch::runtime::elementSize(h_dtype);
  size_t c_elem_size = ::executorch::runtime::elementSize(c_dtype);
  size_t g_elem_size = ::executorch::runtime::elementSize(g_dtype);
  size_t f_elem_size = ::executorch::runtime::elementSize(f_dtype);
  size_t num_elements =
      static_cast<size_t>(num_rnn_layers) * static_cast<size_t>(pred_hidden);

  std::vector<uint8_t> h_data(num_elements * h_elem_size, 0);
  std::vector<uint8_t> c_data(num_elements * c_elem_size, 0);

  auto h = from_blob(
      h_data.data(),
      {static_cast<::executorch::aten::SizesType>(num_rnn_layers),
       1,
       static_cast<::executorch::aten::SizesType>(pred_hidden)},
      h_dtype);
  auto c = from_blob(
      c_data.data(),
      {static_cast<::executorch::aten::SizesType>(num_rnn_layers),
       1,
       static_cast<::executorch::aten::SizesType>(pred_hidden)},
      c_dtype);

  // Prime decoder with SOS token (blank_id)
  std::vector<int64_t> sos_token_data = {blank_id};
  auto sos_token = from_blob(
      sos_token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);
  auto decoder_init_result =
      model.execute("decoder_step", std::vector<EValue>{sos_token, h, c});
  if (!decoder_init_result.ok()) {
    ET_LOG(Error, "decoder_step (SOS) failed");
    return hypothesis;
  }

  auto& init_outputs = decoder_init_result.get();
  auto g_proj_init = init_outputs[0].toTensor();
  auto new_h_init = init_outputs[1].toTensor();
  auto new_c_init = init_outputs[2].toTensor();
  std::memcpy(h_data.data(), new_h_init.const_data_ptr(), h_data.size());
  std::memcpy(c_data.data(), new_c_init.const_data_ptr(), c_data.size());

  size_t g_proj_num_bytes =
      static_cast<size_t>(g_proj_init.numel()) * g_elem_size;
  std::vector<uint8_t> g_proj_data(g_proj_num_bytes);
  std::memcpy(
      g_proj_data.data(), g_proj_init.const_data_ptr(), g_proj_num_bytes);

  int64_t t = 0;
  int64_t symbols_on_frame = 0;
  const uint8_t* f_proj_ptr =
      static_cast<const uint8_t*>(f_proj.const_data_ptr());
  size_t f_t_num_bytes = proj_dim * f_elem_size;

  while (t < encoder_len) {
    std::vector<uint8_t> f_t_data(f_t_num_bytes);
    std::memcpy(
        f_t_data.data(),
        f_proj_ptr + static_cast<size_t>(t) * f_t_num_bytes,
        f_t_num_bytes);

    auto f_t = from_blob(
        f_t_data.data(),
        {1, 1, static_cast<::executorch::aten::SizesType>(proj_dim)},
        f_dtype);

    auto g_proj = from_blob(
        g_proj_data.data(),
        {1, 1, static_cast<::executorch::aten::SizesType>(proj_dim)},
        g_dtype);

    auto joint_result =
        model.execute("joint", std::vector<EValue>{f_t, g_proj});
    if (!joint_result.ok()) {
      ET_LOG(Error, "joint failed at t=%lld", static_cast<long long>(t));
      return hypothesis;
    }

    int64_t k = joint_result.get()[0].toTensor().const_data_ptr<int64_t>()[0];
    int64_t dur_idx =
        joint_result.get()[1].toTensor().const_data_ptr<int64_t>()[0];
    int64_t dur = DURATIONS[dur_idx];

    if (k == blank_id) {
      t += std::max(dur, static_cast<int64_t>(1));
      symbols_on_frame = 0;
    } else {
      hypothesis.push_back({static_cast<TokenId>(k), t, dur});

      std::vector<int64_t> token_data = {k};
      auto token = from_blob(
          token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);

      auto decoder_result =
          model.execute("decoder_step", std::vector<EValue>{token, h, c});
      if (!decoder_result.ok()) {
        ET_LOG(Error, "decoder_step failed");
        return hypothesis;
      }

      auto& outputs = decoder_result.get();
      auto new_g_proj = outputs[0].toTensor();
      auto new_h = outputs[1].toTensor();
      auto new_c = outputs[2].toTensor();

      std::memcpy(h_data.data(), new_h.const_data_ptr(), h_data.size());
      std::memcpy(c_data.data(), new_c.const_data_ptr(), c_data.size());
      std::memcpy(
          g_proj_data.data(), new_g_proj.const_data_ptr(), g_proj_data.size());

      t += dur;

      if (dur == 0) {
        symbols_on_frame++;
        if (symbols_on_frame >= 10) {
          t++;
          symbols_on_frame = 0;
        }
      } else {
        symbols_on_frame = 0;
      }
    }
  }

  return hypothesis;
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_list_devices) {
    std::cout << "Available audio input devices:" << std::endl;
    auto audio_devices = parakeet::AudioStream::list_devices();
    for (const auto& device : audio_devices) {
      std::cout << "  " << device << std::endl;
    }

    std::cout << "\nAvailable video input devices:" << std::endl;
    auto video_devices = parakeet::VideoStream::list_devices();
    for (const auto& device : video_devices) {
      std::cout << "  " << device << std::endl;
    }
    return 0;
  }

  std::cout << "=== Multimodal Runner: Parakeet ASR + YOLO Detection ==="
            << std::endl;

  // Load ASR model (defer full load until after recording to avoid
  // initializing Metal backend during video streaming)
  ET_LOG(Info, "Loading ASR model from: %s", FLAGS_asr_model_path.c_str());
  std::unique_ptr<Module> asr_model;
  if (!FLAGS_asr_data_path.empty()) {
    asr_model = std::make_unique<Module>(
        FLAGS_asr_model_path, FLAGS_asr_data_path, Module::LoadMode::Mmap);
  } else {
    asr_model =
        std::make_unique<Module>(FLAGS_asr_model_path, Module::LoadMode::Mmap);
  }

  // Only query sample_rate (a constant method, no Metal init needed)
  std::vector<EValue> empty_inputs;
  auto sample_rate_result = asr_model->execute("sample_rate", empty_inputs);
  if (!sample_rate_result.ok()) {
    ET_LOG(Error, "Failed to query sample rate");
    return 1;
  }
  int64_t sample_rate = sample_rate_result.get()[0].toInt();
  ET_LOG(Info, "ASR sample rate: %lld", static_cast<long long>(sample_rate));

  // Load tokenizer
  ET_LOG(Info, "Loading tokenizer from: %s", FLAGS_tokenizer_path.c_str());
  auto tokenizer =
      ::executorch::extension::llm::load_tokenizer(FLAGS_tokenizer_path);
  if (!tokenizer || !tokenizer->is_loaded()) {
    ET_LOG(Error, "Failed to load tokenizer");
    return 1;
  }

  // Load YOLO model
  ET_LOG(Info, "Loading YOLO model from: %s", FLAGS_yolo_model_path.c_str());
  parakeet::YOLOConfig yolo_config;
  yolo_config.class_names = parakeet::YOLODetector::get_coco_classes();
  yolo_config.score_threshold = static_cast<float>(FLAGS_yolo_score_threshold);

  parakeet::YOLODetector yolo_detector(
      FLAGS_yolo_model_path, yolo_config, FLAGS_yolo_data_path);

  if (!yolo_detector.initialize()) {
    ET_LOG(Error, "Failed to initialize YOLO detector");
    return 1;
  }

  // Audio collector: just accumulate all samples from the mic
  std::vector<float> all_audio;
  std::mutex audio_mutex;

  // Create audio stream
  parakeet::AudioStreamConfig audio_config;
  audio_config.sample_rate = static_cast<int32_t>(sample_rate);
  audio_config.channels = 1;
  audio_config.frames_per_buffer = 512;

  auto audio_stream = parakeet::create_audio_stream(audio_config);
  audio_stream->set_callback(
      [&all_audio, &audio_mutex](const float* samples, size_t num_samples) {
        std::lock_guard<std::mutex> lock(audio_mutex);
        all_audio.insert(all_audio.end(), samples, samples + num_samples);
      });

  if (!audio_stream->open(FLAGS_audio_device_index)) {
    ET_LOG(Error, "Failed to open audio stream");
    return 1;
  }

  if (!audio_stream->start()) {
    ET_LOG(Error, "Failed to start audio stream");
    return 1;
  }

  // Create video stream
  parakeet::VideoStreamConfig video_config;
  video_config.width = 640;
  video_config.height = 480;
  video_config.fps = 30;

  auto video_stream = parakeet::create_video_stream(video_config);

  if (!video_stream->open(FLAGS_video_device_index)) {
    ET_LOG(Error, "Failed to open video stream");
    return 1;
  }

  if (!video_stream->start()) {
    ET_LOG(Error, "Failed to start video stream");
    return 1;
  }

  std::cout << "\n=== System Active ===" << std::endl;
  std::cout << "Audio: " << FLAGS_audio_device_index << " @ " << sample_rate
            << "Hz" << std::endl;
  std::cout << "Video: " << FLAGS_video_device_index << " @ "
            << video_config.width << "x" << video_config.height << std::endl;
  std::cout << "Press 'q' in video window to quit\n" << std::endl;
  std::cout << "Audio is being recorded. Transcription will run after quit."
            << std::endl;

  // Load ASR model in background while recording
  std::atomic<bool> asr_model_loaded(false);
  Error asr_load_error = Error::Ok;
  int64_t num_rnn_layers = 0, pred_hidden = 0, blank_id = 0;
  std::thread asr_load_thread([&]() {
    asr_load_error = asr_model->load();
    if (asr_load_error != Error::Ok) {
      return;
    }
    auto num_rnn_layers_result =
        asr_model->execute("num_rnn_layers", empty_inputs);
    auto pred_hidden_result = asr_model->execute("pred_hidden", empty_inputs);
    auto blank_id_result = asr_model->execute("blank_id", empty_inputs);
    if (num_rnn_layers_result.ok() && pred_hidden_result.ok() &&
        blank_id_result.ok()) {
      num_rnn_layers = num_rnn_layers_result.get()[0].toInt();
      pred_hidden = pred_hidden_result.get()[0].toInt();
      blank_id = blank_id_result.get()[0].toInt();
      asr_model_loaded.store(true);
    }
  });

  const char* window_name = "Parakeet Multimodal: Video + Detection";
  if (FLAGS_show_video) {
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
  }

  std::atomic<bool> running(true);

  // Main video processing loop (YOLO only, no streaming ASR)
  while (running.load() && video_stream->is_active()) {
    cv::Mat frame;
    if (!video_stream->get_frame(frame)) {
      break;
    }

    if (frame.empty()) {
      continue;
    }

    // Run YOLO detection
    auto detections = yolo_detector.detect(frame);

    // Draw detections on frame
    parakeet::YOLODetector::draw_detections(
        frame, detections, cv::Scalar(0, 255, 0));

    // Show video
    if (FLAGS_show_video) {
      cv::imshow(window_name, frame);
      char key = static_cast<char>(cv::waitKey(1));
      if (key == 'q' || key == 27) {
        running.store(false);
        break;
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  std::cout << "\n\n=== Shutting Down ===" << std::endl;

  // Stop streams
  audio_stream->stop();
  audio_stream->close();
  video_stream->stop();
  video_stream->close();

  if (FLAGS_show_video) {
    cv::destroyAllWindows();
  }

  // Grab collected audio
  std::vector<float> audio_data;
  {
    std::lock_guard<std::mutex> lock(audio_mutex);
    audio_data = std::move(all_audio);
  }

  double audio_seconds =
      static_cast<double>(audio_data.size()) / static_cast<double>(sample_rate);
  std::cout << "Collected " << audio_data.size() << " audio samples ("
            << audio_seconds << "s)" << std::endl;

  // Audio diagnostics
  {
    float peak = 0.0f;
    double sum_sq = 0.0;
    for (float s : audio_data) {
      float abs_s = std::abs(s);
      if (abs_s > peak) {
        peak = abs_s;
      }
      sum_sq += static_cast<double>(s) * static_cast<double>(s);
    }
    double rms = std::sqrt(sum_sq / static_cast<double>(audio_data.size()));
    std::cout << "Audio stats: peak=" << peak << ", rms=" << rms << std::endl;
    if (peak < 0.001f) {
      std::cout << "WARNING: Audio appears to be silence. Check microphone."
                << std::endl;
    }
  }

  if (audio_data.empty()) {
    std::cout << "No audio recorded." << std::endl;
    return 0;
  }

  // Wait for ASR model to finish loading
  asr_load_thread.join();
  if (asr_load_error != Error::Ok || !asr_model_loaded.load()) {
    ET_LOG(Error, "Failed to load ASR model");
    return 1;
  }

  std::cout << "Running preprocessor..." << std::endl;
  auto audio_tensor = from_blob(
      audio_data.data(),
      {static_cast<::executorch::aten::SizesType>(audio_data.size())},
      ::executorch::aten::ScalarType::Float);
  std::vector<int64_t> audio_len_data = {
      static_cast<int64_t>(audio_data.size())};
  auto audio_len_tensor = from_blob(
      audio_len_data.data(), {1}, ::executorch::aten::ScalarType::Long);

  auto proc_result = asr_model->execute(
      "preprocessor", std::vector<EValue>{audio_tensor, audio_len_tensor});
  if (!proc_result.ok()) {
    ET_LOG(Error, "Preprocessor failed");
    return 1;
  }

  auto& proc_outputs = proc_result.get();
  auto mel = proc_outputs[0].toTensor();
  auto mel_len_tensor_out = proc_outputs[1].toTensor();
  int64_t mel_len_value = mel_len_tensor_out.const_data_ptr<int64_t>()[0];

  std::vector<int64_t> mel_len_data = {mel_len_value};
  auto mel_len =
      from_blob(mel_len_data.data(), {1}, ::executorch::aten::ScalarType::Long);

  std::cout << "Running encoder..." << std::endl;
  auto enc_result =
      asr_model->execute("encoder", std::vector<EValue>{mel, mel_len});
  if (!enc_result.ok()) {
    ET_LOG(Error, "Encoder failed");
    return 1;
  }

  auto& enc_outputs = enc_result.get();
  auto f_proj = enc_outputs[0].toTensor();
  int64_t encoded_len =
      enc_outputs[1].toTensor().const_data_ptr<int64_t>()[0];

  std::cout << "Running greedy decode..." << std::endl;
  auto decoded_tokens = greedy_decode(
      *asr_model, f_proj, encoded_len, blank_id, num_rnn_layers, pred_hidden);

  std::cout << "Decoded " << decoded_tokens.size() << " tokens" << std::endl;

  std::string text = parakeet::tokenizer_utils::decode_token_sequence(
      decoded_tokens, *tokenizer);
  std::cout << "Transcription: " << text << std::endl;

  return 0;
}
