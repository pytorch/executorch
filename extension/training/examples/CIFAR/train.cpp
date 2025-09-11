/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/flat_tensor/serialize/serialize.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/training/module/training_module.h>
#include <executorch/extension/training/optimizer/sgd.h>
#include <fcntl.h>
#include <gflags/gflags.h>
#include <sys/stat.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <numeric>
#include <random>
#include <string>

// Define namespace aliases for cleaner code
using executorch::extension::training::optimizer::SGD; // Stochastic Gradient
                                                       // Descent optimizer
using executorch::extension::training::optimizer::SGDOptions; // Options for SGD
                                                              // optimizer
using executorch::runtime::Error; // Error handling

// Define command-line flags
DEFINE_string(
    model_path,
    "/data/sandcastle/boxes/fbsource/fbcode/executorch/extension/training/"
    "examples/CIFAR/cifar10_model.pte",
    "Model serialized in flatbuffer format."); // Path to the model file
DEFINE_string(
    ptd_path,
    "",
    "Model weights serialized in flatbuffer format."); // Path to trained
                                                       // weights (optional)
DEFINE_string(
    train_data_path,
    "/data/sandcastle/boxes/fbsource/fbcode/executorch/extension/training/"
    "examples/CIFAR/cifar-10/extracted_data/train_data.bin",
    "Path to the combined training data file."); // Path to the combined train
                                                 // data file
DEFINE_string(
    test_data_path,
    "/data/sandcastle/boxes/fbsource/fbcode/executorch/extension/"
    "training/examples/CIFAR/cifar-10/extracted_data/test_data.bin",
    "Path to the combined test data file."); // Path to the combined
                                             // test data file

DEFINE_string(
    ptd_save_path,
    "/data/sandcastle/boxes/fbsource/fbcode/executorch/extension/training/"
    "examples/CIFAR/CPP/",
    "Path to save the cpp model trained weights."); // Path to save the trained
                                                    // weights

DEFINE_int32(
    batch_size,
    4,
    "Batch size for training."); // Batch size for training (must match
                                 // export batch size)

DEFINE_int32(
    num_epochs,
    1,
    "Number of epochs to train."); // Number of epochs to train

DEFINE_double(
    learning_rate,
    0.001,
    "Learning rate for SGD optimizer."); // Learning rate

DEFINE_double(momentum, 0.9,
              "Momentum for SGD optimizer."); // Momentum

// Constants for the CIFAR-10 dataset
const size_t IMAGE_C = 3; // Number of color channels
const size_t IMAGE_H = 32; // Image height
const size_t IMAGE_W = 32; // Image width
const size_t IMAGE_TENSOR_SIZE = IMAGE_C * IMAGE_H * IMAGE_W; // Size of image

void train_model(
    executorch::extension::training::TrainingModule& mod,
    const std::vector<std::pair<
        executorch::extension::TensorPtr,
        executorch::extension::TensorPtr>>& dataset,
    SGD& optimizer,
    std::mt19937& g) {
  ET_LOG(
      Info,
      "Starting training for %d epochs with batch size %d...",
      FLAGS_num_epochs,
      FLAGS_batch_size);

  for (int epoch = 0; epoch < FLAGS_num_epochs; epoch++) {
    auto epoch_start = std::chrono::high_resolution_clock::now();

    float epoch_loss = 0.0;
    size_t correct_predictions = 0;
    size_t total_samples = 0;

    // Shuffling the dataset indices for each epoch for better learning
    std::vector<size_t> indices(dataset.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), g);

    // Process data in batches
    size_t num_batches = 0;
    for (size_t i = 0; i < dataset.size(); i += FLAGS_batch_size) {
      // Skip incomplete batches at the end
      if (i + FLAGS_batch_size > dataset.size()) {
        break;
      }

      // Start timing data batch preparation
      auto data_prep_start = std::chrono::high_resolution_clock::now();

      // Create batch tensors
      auto batch_image_buffer = std::make_shared<std::vector<float>>(
          FLAGS_batch_size * IMAGE_C * IMAGE_H * IMAGE_W);
      auto batch_label_buffer =
          std::make_shared<std::vector<int32_t>>(FLAGS_batch_size);

      // Fill batch tensors with data from batch size samples
      for (int j = 0; j < FLAGS_batch_size; j++) {
        size_t idx = indices.at(i + j);
        auto& data = dataset[idx];

        // Copy image data
        const float* src_img = data.first->const_data_ptr<float>();
        float* dst_img =
            batch_image_buffer->data() + (j * IMAGE_C * IMAGE_H * IMAGE_W);
        std::memcpy(
            dst_img, src_img, IMAGE_C * IMAGE_H * IMAGE_W * sizeof(float));

        // Copy label data
        batch_label_buffer->at(j) = data.second->const_data_ptr<int32_t>()[0];
      }

      // Create batch tensors
      executorch::extension::TensorPtr batch_image_tensor =
          executorch::extension::make_tensor_ptr<float>(
              {FLAGS_batch_size, IMAGE_C, IMAGE_H, IMAGE_W},
              *batch_image_buffer);

      // Convert int32_t labels to int64_t as expected by the model
      auto batch_label_buffer_int64 =
          std::make_shared<std::vector<int64_t>>(FLAGS_batch_size);
      for (int j = 0; j < FLAGS_batch_size; j++) {
        batch_label_buffer_int64->at(j) =
            static_cast<int64_t>(batch_label_buffer->at(j));
      }

      executorch::extension::TensorPtr batch_label_tensor =
          executorch::extension::make_tensor_ptr<int64_t>(
              {FLAGS_batch_size}, *batch_label_buffer_int64);

      // End timing data batch preparation
      auto data_prep_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> data_prep_time =
          data_prep_end - data_prep_start;

      // Start timing model training
      auto train_start = std::chrono::high_resolution_clock::now();

      // Execute forward and backward pass on the batch
      const auto& results = mod.execute_forward_backward(
          "forward", {*batch_image_tensor, *batch_label_tensor});
      if (results.error() != Error::Ok) {
        ET_LOG(
            Error,
            "Failed to execute the forward method on batch starting at "
            "sample %zu",
            i);
        return;
      }

      // Process results
      float loss = results.get()[0].toTensor().const_data_ptr<float>()[0];
      epoch_loss += loss;

      // Count correct predictions in the batch
      const int64_t* predictions =
          results.get()[1].toTensor().const_data_ptr<int64_t>();
      for (int j = 0; j < FLAGS_batch_size; j++) {
        if (predictions[j] == static_cast<int64_t>(batch_label_buffer->at(j))) {
          correct_predictions++;
        }
      }
      total_samples += FLAGS_batch_size;

      // Get gradients and update parameters
      auto grads = mod.named_gradients("forward");
      if (grads.error() != Error::Ok) {
        ET_LOG(Error, "Failed to get named gradients");
        return;
      }
      optimizer.step(grads.get());

      // End timing model training
      auto train_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> train_time =
          train_end - train_start;

      num_batches++;

      // Log for tracking progress
      if (num_batches % 100 == 0) {
        ET_LOG(
            Info,
            "Epoch [%d/%d], Batch [%zu/%zu], Loss: %.4f, Data prep: %.2f "
            "ms, Train: %.2f ms",
            epoch + 1,
            FLAGS_num_epochs,
            num_batches,
            dataset.size() / FLAGS_batch_size,
            loss,
            data_prep_time.count(),
            train_time.count());
      }
    }

    auto epoch_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> epoch_time = epoch_end - epoch_start;

    // Log epoch summary
    float avg_loss = epoch_loss / num_batches;
    float accuracy = 100.0f * correct_predictions / total_samples;
    ET_LOG(
        Info,
        "Epoch %d/%d Summary: Avg Loss: %.4f, Accuracy: %.2f%% (%zu/%zu), "
        "Time: %.2f s",
        epoch + 1,
        FLAGS_num_epochs,
        avg_loss,
        accuracy,
        correct_predictions,
        total_samples,
        epoch_time.count());
  }

  ET_LOG(Info, "Training finished...");
}

void evaluate_on_test_set(
    executorch::extension::training::TrainingModule& mod,
    const std::vector<std::pair<
        executorch::extension::TensorPtr,
        executorch::extension::TensorPtr>>& test_dataset) {
  ET_LOG(Info, "Starting final evaluation on test set...");
  auto eval_start = std::chrono::high_resolution_clock::now();

  float test_loss = 0.0;
  size_t test_correct = 0;
  size_t test_total = 0;
  size_t test_batches = 0;

  for (size_t i = 0; i < test_dataset.size(); i += FLAGS_batch_size) {
    if (i + FLAGS_batch_size > test_dataset.size()) {
      break;
    }

    // Create batch tensors for test data
    auto batch_image_buffer = std::make_shared<std::vector<float>>(
        FLAGS_batch_size * IMAGE_C * IMAGE_H * IMAGE_W);
    auto batch_label_buffer =
        std::make_shared<std::vector<int32_t>>(FLAGS_batch_size);

    // Fill batch tensors with test data
    for (int j = 0; j < FLAGS_batch_size; j++) {
      auto& data = test_dataset[i + j];

      // Copy image data
      const float* src_img = data.first->const_data_ptr<float>();
      float* dst_img =
          batch_image_buffer->data() + (j * IMAGE_C * IMAGE_H * IMAGE_W);
      std::memcpy(
          dst_img, src_img, IMAGE_C * IMAGE_H * IMAGE_W * sizeof(float));

      // Copy label data
      batch_label_buffer->at(j) = data.second->const_data_ptr<int32_t>()[0];
    }

    // Create batch tensors
    executorch::extension::TensorPtr batch_image_tensor =
        executorch::extension::make_tensor_ptr<float>(
            {FLAGS_batch_size, IMAGE_C, IMAGE_H, IMAGE_W}, *batch_image_buffer);

    // Convert int32_t labels to int64_t as expected by the model
    auto batch_label_buffer_int64 =
        std::make_shared<std::vector<int64_t>>(FLAGS_batch_size);
    for (int j = 0; j < FLAGS_batch_size; j++) {
      batch_label_buffer_int64->at(j) =
          static_cast<int64_t>(batch_label_buffer->at(j));
    }

    executorch::extension::TensorPtr batch_label_tensor =
        executorch::extension::make_tensor_ptr<int64_t>(
            {FLAGS_batch_size}, *batch_label_buffer_int64);

    const auto& results = mod.execute_forward_backward(
        "forward", {*batch_image_tensor, *batch_label_tensor});
    if (results.error() != Error::Ok) {
      ET_LOG(
          Error,
          "Failed to execute forward pass on test batch starting at sample %zu",
          i);
      continue;
    }

    // Process results
    float loss = results.get()[0].toTensor().const_data_ptr<float>()[0];
    test_loss += loss;

    // Count correct predictions
    const int64_t* predictions =
        results.get()[1].toTensor().const_data_ptr<int64_t>();
    for (int j = 0; j < FLAGS_batch_size; j++) {
      if (predictions[j] == static_cast<int64_t>(batch_label_buffer->at(j))) {
        test_correct++;
      }
    }
    test_total += FLAGS_batch_size;
    test_batches++;
  }

  auto eval_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> eval_time = eval_end - eval_start;

  float test_avg_loss = test_loss / test_batches;
  float test_accuracy = 100.0f * test_correct / test_total;

  ET_LOG(
      Info,
      "Final Test Results: Avg Loss: %.4f, Accuracy: %.2f%% (%zu/%zu), "
      "Time: %.2f s",
      test_avg_loss,
      test_accuracy,
      test_correct,
      test_total,
      eval_time.count());
}

torch::executor::Error load_data_from_combined_binary(
    const std::string& data_path,
    std::vector<std::pair<
        executorch::extension::TensorPtr,
        executorch::extension::TensorPtr>>& data_set) {
  std::ifstream data_file(data_path, std::ios::binary);

  if (!data_file.is_open()) {
    ET_LOG(Error, "Failed to open data file: %s", data_path.c_str());
    return torch::executor::Error::InvalidState;
  }

  ET_LOG(
      Info,
      "Loading the dataset from the combined binary file: %s",
      data_path.c_str());

  data_file.seekg(0, std::ios::end);
  std::streampos file_size = data_file.tellg();
  data_file.seekg(0, std::ios::beg);

  // Debug: Read first 32 bytes to understand file format
  char debug_bytes[32];
  data_file.read(debug_bytes, 32);
  data_file.seekg(0, std::ios::beg); // Reset to beginning

  // Try CIFAR-10 format: label (1 byte) + image (3072 bytes)
  // This is the standard CIFAR-10 binary format
  size_t cifar_sample_size =
      1 + IMAGE_TENSOR_SIZE; // 1 byte label + 3072 bytes image
  size_t cifar_max_samples = file_size / cifar_sample_size;

  for (size_t i = 0; i < cifar_max_samples; i++) {
    // Read label (1 byte)
    uint8_t label_byte;
    data_file.read(reinterpret_cast<char*>(&label_byte), 1);
    if (data_file.gcount() != 1) {
      ET_LOG(Error, "Failed to read label byte at sample %zu", i);
      return torch::executor::Error::InvalidState;
    }

    // Read image data (3072 bytes as uint8_t, then convert to float)
    std::vector<uint8_t> image_bytes(IMAGE_TENSOR_SIZE);
    data_file.read(
        reinterpret_cast<char*>(image_bytes.data()), IMAGE_TENSOR_SIZE);
    if (data_file.gcount() != IMAGE_TENSOR_SIZE) {
      ET_LOG(Error, "Failed to read image bytes at sample %zu", i);
      return torch::executor::Error::InvalidState;
    }

    // Validate label range
    if (label_byte > 9) {
      ET_LOG(
          Error,
          "Invalid label value %u at sample %zu (expected 0-9)",
          label_byte,
          i);
      return torch::executor::Error::InvalidState;
    }

    // Convert image bytes to floats (normalize to 0-1 range)
    auto image_buffer = std::make_shared<std::vector<float>>(IMAGE_TENSOR_SIZE);
    for (size_t j = 0; j < IMAGE_TENSOR_SIZE; j++) {
      (*image_buffer)[j] = static_cast<float>(image_bytes[j]) / 255.0f;
    }

    // Create label buffer
    auto label_buffer = std::make_shared<std::vector<int32_t>>(1);
    (*label_buffer)[0] = static_cast<int32_t>(label_byte);

    // Store the image and label buffers
    data_set.emplace_back(
        executorch::extension::make_tensor_ptr<float>(
            {1, IMAGE_C, IMAGE_H, IMAGE_W}, *image_buffer),
        executorch::extension::make_tensor_ptr<int32_t>({1}, *label_buffer));
  }

  ET_LOG(
      Info,
      "Successfully loaded %zu samples using CIFAR-10 format.",
      data_set.size());
  return Error::Ok;
}

int main(int argc, char** argv) {
  // Parse command-line flags
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Load the model: The following code works for loading the pte model
  executorch::runtime::Result<executorch::extension::FileDataLoader>
      loader_res =
          executorch::extension::FileDataLoader::from(FLAGS_model_path.c_str());
  if (loader_res.error() != Error::Ok) {
    ET_LOG(Error, "Failed to open model file: %s", FLAGS_model_path.c_str());
    return 1;
  } else {
    ET_LOG(
        Info, "Successfully opened model file: %s", FLAGS_model_path.c_str());
  }

  auto loader = std::make_unique<executorch::extension::FileDataLoader>(
      std::move(loader_res.get()));

  std::unique_ptr<executorch::extension::FileDataLoader> ptd_loader = nullptr;
  if (!FLAGS_ptd_path.empty()) {
    executorch::runtime::Result<executorch::extension::FileDataLoader>
        ptd_loader_res =
            executorch::extension::FileDataLoader::from(FLAGS_ptd_path.c_str());
    if (ptd_loader_res.error() != Error::Ok) {
      ET_LOG(Error, "Failed to open ptd file: %s", FLAGS_ptd_path.c_str());
      return 1;
    } else {
      ET_LOG(
          Info,
          "Successfully opened trained weights file: %s",
          FLAGS_ptd_path.c_str());
    }
    ptd_loader = std::make_unique<executorch::extension::FileDataLoader>(
        std::move(ptd_loader_res.get()));
  }

  auto mod = executorch::extension::training::TrainingModule(
      std::move(loader), nullptr, nullptr, nullptr, std::move(ptd_loader));

  // Load the training dataset from combined binary file
  std::vector<std::pair<
      executorch::extension::TensorPtr,
      executorch::extension::TensorPtr>>
      dataset;
  Error data_load_res =
      load_data_from_combined_binary(FLAGS_train_data_path, dataset);
  if (data_load_res != Error::Ok) {
    return 1;
  }

  // Confirm that the dataset has been loaded correctly
  ET_LOG(
      Info,
      "Successfully loaded the dataset with %zu samples.",
      dataset.size());

  // Create optimizer.
  // Get the params and names
  auto param_res = mod.named_parameters("forward");
  if (param_res.error() != Error::Ok) {
    ET_LOG(
        Error,
        "Failed to get named parameters, error: %d",
        static_cast<int>(param_res.error()));
    return 1;
  }

  SGDOptions options{FLAGS_learning_rate, FLAGS_momentum};
  SGD optimizer(param_res.get(), options);

  ET_LOG(
      Info,
      "Successfully created the optimizer with lr=%.4f, momentum=%.2f.",
      FLAGS_learning_rate,
      FLAGS_momentum);

  // Initialize random number generator for shuffling
  std::random_device rd;
  std::mt19937 g(rd());

  train_model(mod, dataset, optimizer, g);

  // Load test dataset for evaluation
  std::vector<std::pair<
      executorch::extension::TensorPtr,
      executorch::extension::TensorPtr>>
      test_dataset;
  Error test_data_load_res =
      load_data_from_combined_binary(FLAGS_test_data_path, test_dataset);
  if (test_data_load_res != Error::Ok) {
    ET_LOG(Error, "Failed to load test dataset, skipping evaluation");
  } else {
    ET_LOG(
        Info,
        "Successfully loaded test dataset with %zu samples.",
        test_dataset.size());

    evaluate_on_test_set(mod, test_dataset);
  }

  // Save the trained weights
  std::map<std::string, executorch::aten::Tensor> param_map;
  for (auto& param : param_res.get()) {
    param_map.insert({std::string(param.first.data()), param.second});
  }

  // Define the directory path for saving the model
  const std::string model_path = FLAGS_ptd_save_path + "trained_cifar_cpp.ptd";

  // Create the directory if it doesn't exist
  int dir_fd = open(FLAGS_ptd_save_path.c_str(), O_RDONLY);
  if (dir_fd == -1) {
    // Directory doesn't exist or can't be accessed, create it
    ET_LOG(Info, "Creating directory: %s", FLAGS_ptd_save_path.c_str());
    int result = mkdir(
        FLAGS_ptd_save_path.c_str(),
        0755); // Create with permissions rwxr-xr-x
    if (result != 0) {
      ET_LOG(
          Error, "Failed to create directory: %s", FLAGS_ptd_save_path.c_str());
      return 1;
    }
  } else {
    // Directory exists, check if it's actually a directory
    struct stat info {};
    if (fstat(dir_fd, &info) == 0 && !(info.st_mode & S_IFDIR)) {
      close(dir_fd);
      ET_LOG(
          Error,
          "Path exists but is not a directory: %s",
          FLAGS_ptd_save_path.c_str());
      return 1;
    }
    close(dir_fd);
  }

  executorch::extension::flat_tensor::save_ptd(
      model_path.c_str(), param_map, 16);
  ET_LOG(Info, "Trained weights saved to %s", model_path.c_str());

  return 0;
}
