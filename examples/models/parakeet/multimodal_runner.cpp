/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <atomic>
#include <chrono>
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
DEFINE_double(chunk_seconds, 2.0, "Audio chunk duration in seconds.");
DEFINE_double(left_context_seconds, 10.0, "Left context duration in seconds.");
DEFINE_double(yolo_score_threshold, 0.45, "YOLO detection score threshold.");
DEFINE_double(yolo_nms_threshold, 0.50, "YOLO NMS threshold.");
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

// Streaming transcription engine (same as streaming_runner)
class StreamingTranscriber {
 public:
  StreamingTranscriber(
      Module& model,
      const tokenizers::Tokenizer& tokenizer,
      int64_t sample_rate,
      double chunk_seconds,
      double left_context_seconds)
      : model_(model),
        tokenizer_(tokenizer),
        sample_rate_(sample_rate),
        chunk_size_(static_cast<size_t>(chunk_seconds * sample_rate)),
        left_context_size_(
            static_cast<size_t>(left_context_seconds * sample_rate)),
        is_initialized_(false) {
    audio_buffer_.reserve(left_context_size_ + chunk_size_);
  }

  bool initialize() {
    std::vector<EValue> empty_inputs;
    auto num_rnn_layers_result = model_.execute("num_rnn_layers", empty_inputs);
    auto pred_hidden_result = model_.execute("pred_hidden", empty_inputs);
    auto blank_id_result = model_.execute("blank_id", empty_inputs);

    if (!num_rnn_layers_result.ok() || !pred_hidden_result.ok() ||
        !blank_id_result.ok()) {
      ET_LOG(Error, "Failed to query ASR model metadata");
      return false;
    }

    num_rnn_layers_ = num_rnn_layers_result.get()[0].toInt();
    pred_hidden_ = pred_hidden_result.get()[0].toInt();
    blank_id_ = blank_id_result.get()[0].toInt();

    auto h_dtype_result = get_input_scalar_type(model_, "decoder_step", 1);
    auto c_dtype_result = get_input_scalar_type(model_, "decoder_step", 2);
    if (!h_dtype_result.ok() || !c_dtype_result.ok()) {
      return false;
    }
    h_dtype_ = h_dtype_result.get();
    c_dtype_ = c_dtype_result.get();

    auto f_dtype_result = get_input_scalar_type(model_, "joint", 0);
    auto g_dtype_result = get_input_scalar_type(model_, "joint", 1);
    if (!f_dtype_result.ok() || !g_dtype_result.ok()) {
      return false;
    }
    f_dtype_ = f_dtype_result.get();
    g_dtype_ = g_dtype_result.get();

    size_t h_elem_size = ::executorch::runtime::elementSize(h_dtype_);
    size_t c_elem_size = ::executorch::runtime::elementSize(c_dtype_);
    size_t g_elem_size = ::executorch::runtime::elementSize(g_dtype_);
    size_t num_elements =
        static_cast<size_t>(num_rnn_layers_) * static_cast<size_t>(pred_hidden_);

    h_data_.resize(num_elements * h_elem_size, 0);
    c_data_.resize(num_elements * c_elem_size, 0);

    std::vector<int64_t> sos_token_data = {blank_id_};
    auto sos_token = from_blob(
        sos_token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);

    auto h = from_blob(
        h_data_.data(),
        {static_cast<::executorch::aten::SizesType>(num_rnn_layers_),
         1,
         static_cast<::executorch::aten::SizesType>(pred_hidden_)},
        h_dtype_);
    auto c = from_blob(
        c_data_.data(),
        {static_cast<::executorch::aten::SizesType>(num_rnn_layers_),
         1,
         static_cast<::executorch::aten::SizesType>(pred_hidden_)},
        c_dtype_);

    auto decoder_init_result = model_.execute(
        "decoder_step", std::vector<EValue>{sos_token, h, c});
    if (!decoder_init_result.ok()) {
      ET_LOG(Error, "Failed to initialize decoder");
      return false;
    }

    auto& init_outputs = decoder_init_result.get();
    auto g_proj_init = init_outputs[0].toTensor();
    auto new_h_init = init_outputs[1].toTensor();
    auto new_c_init = init_outputs[2].toTensor();

    std::memcpy(h_data_.data(), new_h_init.const_data_ptr(), h_data_.size());
    std::memcpy(c_data_.data(), new_c_init.const_data_ptr(), c_data_.size());

    size_t g_proj_num_bytes =
        static_cast<size_t>(g_proj_init.numel()) * g_elem_size;
    g_proj_data_.resize(g_proj_num_bytes);
    std::memcpy(
        g_proj_data_.data(), g_proj_init.const_data_ptr(), g_proj_num_bytes);

    is_initialized_ = true;
    ET_LOG(Info, "ASR transcriber initialized");
    return true;
  }

  void process_audio_chunk(const float* samples, size_t num_samples) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);

    audio_buffer_.insert(
        audio_buffer_.end(), samples, samples + num_samples);

    if (audio_buffer_.size() >= chunk_size_) {
      std::vector<float> chunk_to_process;

      size_t start_idx = 0;
      if (audio_buffer_.size() > chunk_size_ + left_context_size_) {
        start_idx = audio_buffer_.size() - chunk_size_ - left_context_size_;
      }

      chunk_to_process.assign(
          audio_buffer_.begin() + start_idx, audio_buffer_.end());

      std::thread([this, chunk = std::move(chunk_to_process)]() mutable {
        this->process_chunk_internal(chunk);
      }).detach();

      if (audio_buffer_.size() > left_context_size_) {
        audio_buffer_.erase(
            audio_buffer_.begin(),
            audio_buffer_.end() - left_context_size_);
      }
    }
  }

  std::string get_current_text() const {
    std::lock_guard<std::mutex> lock(text_mutex_);
    return current_text_;
  }

 private:
  void process_chunk_internal(std::vector<float>& audio_chunk) {
    if (!is_initialized_) {
      return;
    }

    auto audio_tensor = from_blob(
        audio_chunk.data(),
        {static_cast<::executorch::aten::SizesType>(audio_chunk.size())},
        ::executorch::aten::ScalarType::Float);
    std::vector<int64_t> audio_len_data = {
        static_cast<int64_t>(audio_chunk.size())};
    auto audio_len_tensor = from_blob(
        audio_len_data.data(), {1}, ::executorch::aten::ScalarType::Long);

    auto proc_result = model_.execute(
        "preprocessor", std::vector<EValue>{audio_tensor, audio_len_tensor});
    if (!proc_result.ok()) {
      return;
    }

    auto& proc_outputs = proc_result.get();
    auto mel = proc_outputs[0].toTensor();
    auto mel_len_tensor_out = proc_outputs[1].toTensor();
    int64_t mel_len_value = mel_len_tensor_out.const_data_ptr<int64_t>()[0];

    std::vector<int64_t> mel_len_data = {mel_len_value};
    auto mel_len =
        from_blob(mel_len_data.data(), {1}, ::executorch::aten::ScalarType::Long);

    auto enc_result =
        model_.execute("encoder", std::vector<EValue>{mel, mel_len});
    if (!enc_result.ok()) {
      return;
    }

    auto& enc_outputs = enc_result.get();
    auto f_proj = enc_outputs[0].toTensor();
    int64_t encoded_len = enc_outputs[1].toTensor().const_data_ptr<int64_t>()[0];

    auto tokens = decode_greedy(f_proj, encoded_len);

    if (!tokens.empty()) {
      std::string new_text =
          parakeet::tokenizer_utils::decode_token_sequence(tokens, tokenizer_);

      std::lock_guard<std::mutex> lock(text_mutex_);
      current_text_ = new_text;
    }
  }

  std::vector<Token> decode_greedy(
      const ::executorch::aten::Tensor& f_proj,
      int64_t encoder_len) {
    std::vector<Token> hypothesis;
    size_t proj_dim = static_cast<size_t>(f_proj.sizes()[2]);
    size_t f_elem_size = ::executorch::runtime::elementSize(f_dtype_);
    size_t f_t_num_bytes = proj_dim * f_elem_size;

    int64_t t = 0;
    int64_t symbols_on_frame = 0;
    const uint8_t* f_proj_ptr =
        static_cast<const uint8_t*>(f_proj.const_data_ptr());

    auto h = from_blob(
        h_data_.data(),
        {static_cast<::executorch::aten::SizesType>(num_rnn_layers_),
         1,
         static_cast<::executorch::aten::SizesType>(pred_hidden_)},
        h_dtype_);
    auto c = from_blob(
        c_data_.data(),
        {static_cast<::executorch::aten::SizesType>(num_rnn_layers_),
         1,
         static_cast<::executorch::aten::SizesType>(pred_hidden_)},
        c_dtype_);

    while (t < encoder_len) {
      std::vector<uint8_t> f_t_data(f_t_num_bytes);
      std::memcpy(
          f_t_data.data(),
          f_proj_ptr + static_cast<size_t>(t) * f_t_num_bytes,
          f_t_num_bytes);

      auto f_t = from_blob(
          f_t_data.data(),
          {1, 1, static_cast<::executorch::aten::SizesType>(proj_dim)},
          f_dtype_);

      auto g_proj = from_blob(
          g_proj_data_.data(),
          {1, 1, static_cast<::executorch::aten::SizesType>(proj_dim)},
          g_dtype_);

      auto joint_result = model_.execute("joint", std::vector<EValue>{f_t, g_proj});
      if (!joint_result.ok()) {
        return hypothesis;
      }

      int64_t k = joint_result.get()[0].toTensor().const_data_ptr<int64_t>()[0];
      int64_t dur_idx =
          joint_result.get()[1].toTensor().const_data_ptr<int64_t>()[0];
      int64_t dur = DURATIONS[dur_idx];

      if (k == blank_id_) {
        t += std::max(dur, static_cast<int64_t>(1));
        symbols_on_frame = 0;
      } else {
        hypothesis.push_back({static_cast<TokenId>(k), t, dur});

        std::vector<int64_t> token_data = {k};
        auto token = from_blob(
            token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);

        auto decoder_result =
            model_.execute("decoder_step", std::vector<EValue>{token, h, c});
        if (!decoder_result.ok()) {
          return hypothesis;
        }

        auto& outputs = decoder_result.get();
        auto new_g_proj = outputs[0].toTensor();
        auto new_h = outputs[1].toTensor();
        auto new_c = outputs[2].toTensor();

        std::memcpy(h_data_.data(), new_h.const_data_ptr(), h_data_.size());
        std::memcpy(c_data_.data(), new_c.const_data_ptr(), c_data_.size());
        std::memcpy(
            g_proj_data_.data(), new_g_proj.const_data_ptr(), g_proj_data_.size());

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

  Module& model_;
  const tokenizers::Tokenizer& tokenizer_;
  int64_t sample_rate_;
  size_t chunk_size_;
  size_t left_context_size_;

  int64_t num_rnn_layers_;
  int64_t pred_hidden_;
  int64_t blank_id_;

  ::executorch::aten::ScalarType h_dtype_;
  ::executorch::aten::ScalarType c_dtype_;
  ::executorch::aten::ScalarType f_dtype_;
  ::executorch::aten::ScalarType g_dtype_;

  std::vector<uint8_t> h_data_;
  std::vector<uint8_t> c_data_;
  std::vector<uint8_t> g_proj_data_;

  std::vector<float> audio_buffer_;
  mutable std::mutex buffer_mutex_;

  std::string current_text_;
  mutable std::mutex text_mutex_;

  bool is_initialized_;
};

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

  // Load ASR model
  ET_LOG(Info, "Loading ASR model from: %s", FLAGS_asr_model_path.c_str());
  std::unique_ptr<Module> asr_model;
  if (!FLAGS_asr_data_path.empty()) {
    asr_model = std::make_unique<Module>(
        FLAGS_asr_model_path, FLAGS_asr_data_path, Module::LoadMode::Mmap);
  } else {
    asr_model =
        std::make_unique<Module>(FLAGS_asr_model_path, Module::LoadMode::Mmap);
  }

  auto asr_load_error = asr_model->load();
  if (asr_load_error != Error::Ok) {
    ET_LOG(Error, "Failed to load ASR model");
    return 1;
  }

  // Query ASR sample rate
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

  // Create ASR transcriber
  StreamingTranscriber transcriber(
      *asr_model,
      *tokenizer,
      sample_rate,
      FLAGS_chunk_seconds,
      FLAGS_left_context_seconds);

  if (!transcriber.initialize()) {
    ET_LOG(Error, "Failed to initialize ASR transcriber");
    return 1;
  }

  // Load YOLO model
  ET_LOG(Info, "Loading YOLO model from: %s", FLAGS_yolo_model_path.c_str());
  parakeet::YOLOConfig yolo_config;
  yolo_config.class_names = parakeet::YOLODetector::get_coco_classes();
  yolo_config.score_threshold = static_cast<float>(FLAGS_yolo_score_threshold);
  yolo_config.nms_threshold = static_cast<float>(FLAGS_yolo_nms_threshold);

  parakeet::YOLODetector yolo_detector(
      FLAGS_yolo_model_path, yolo_config, FLAGS_yolo_data_path);

  if (!yolo_detector.initialize()) {
    ET_LOG(Error, "Failed to initialize YOLO detector");
    return 1;
  }

  // Create audio stream
  parakeet::AudioStreamConfig audio_config;
  audio_config.sample_rate = static_cast<int32_t>(sample_rate);
  audio_config.channels = 1;
  audio_config.frames_per_buffer = 512;

  auto audio_stream = parakeet::create_audio_stream(audio_config);
  audio_stream->set_callback(
      [&transcriber](const float* samples, size_t num_samples) {
        transcriber.process_audio_chunk(samples, num_samples);
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

  const char* window_name = "Parakeet Multimodal: Video + Detection";
  if (FLAGS_show_video) {
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
  }

  std::atomic<bool> running(true);
  std::string last_transcription;

  // Main video processing loop
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

    // Update transcription display
    std::string current_transcription = transcriber.get_current_text();
    if (current_transcription != last_transcription) {
      last_transcription = current_transcription;
      std::cout << "\r\033[K"; // Clear line
      std::cout << "Transcription: " << current_transcription << std::flush;
    }

    // Small delay to avoid busy-waiting
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  std::cout << "\n\n=== Shutting Down ===" << std::endl;

  // Cleanup
  audio_stream->stop();
  audio_stream->close();
  video_stream->stop();
  video_stream->close();

  if (FLAGS_show_video) {
    cv::destroyAllWindows();
  }

  std::cout << "Final transcription: " << transcriber.get_current_text()
            << std::endl;

  return 0;
}
