//
//  ExecutorchModule.m
//  ExecutorchMobileNet
//
//  Created by Chen Lai on 8/10/23.
//


#import "ExecutorchModule.h"
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <cstddef>
#include <string>
//
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/util/util.h>
#include <executorch/util/read_file.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>

#include <iostream>
#include <sstream>
#import <sys/stat.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace torch::executor;
using torch::executor::util::FileDataLoader;
using torch::executor::testing::TensorFactory;


static constexpr size_t kRuntimeMemorySize = 64021120;
static constexpr size_t kMemoryAmount = 64021120;

static uint8_t runtime_pool[kRuntimeMemorySize];
static uint8_t activation_pool[kMemoryAmount];

@implementation ExecutorchModule {
@protected
    std::string executorch_module_name;
}

struct IndexValuePair {
    int index;
    float value;

    IndexValuePair(int index, float value) : index(index), value(value) {}
};

// Comparator function to sort IndexValuePair objects based on values
bool compareValues(const IndexValuePair& a, const IndexValuePair& b) {
    return a.value > b.value; // Sort in descending order of values
}

std::vector<IndexValuePair> findTopK(const std::vector<float>& nums, int k) {
    std::vector<IndexValuePair> indexValuePairs;

    // Store index-value pairs in the custom data structure
    for (int i = 0; i < nums.size(); ++i) {
        indexValuePairs.emplace_back(i, nums[i]);
    }

    // Sort the index-value pairs based on values
    std::sort(indexValuePairs.begin(), indexValuePairs.end(), compareValues);

    // Return the top k index-value pairs
    return std::vector<IndexValuePair>(indexValuePairs.begin(), indexValuePairs.begin() + k);
}

std::vector<std::string> get_image_net_classes() {
    NSString *imagenet_classes_path = [[NSBundle mainBundle] pathForResource:@"imagenet_classes" ofType:@"txt"];
    NSLog(@"   imagenet_classes_path: %@", imagenet_classes_path);

    std::vector<std::string> stringList; // Create a vector to store the strings

    // Open the file for reading
    std::string imagenet_classes_name = [imagenet_classes_path UTF8String];
    std::ifstream inputFile(imagenet_classes_name);

    if (inputFile.is_open()) {
        std::string line;

        // Read each line from the file and add it to the vector
        while (std::getline(inputFile, line)) {
            stringList.push_back(line);
        }

        // Close the file after reading
        inputFile.close();

    } else {
        std::cerr << "Failed to open the file." << std::endl;
    }
    return  stringList;
}


- (nullable instancetype)initWithFileAtPath:(NSString*)filePath {
    printf("start initing...");
    self = [super init];
    if (self) {
        try {
            executorch_module_name = filePath.UTF8String;
        } catch (const std::exception& exception) {
            NSLog(@"%s", exception.what());
            return nil;
        }
    }
    return self;
}

- (char*)segmentImage:(void *)imageBuffer withWidth:(int)width withHeight:(int)height {
    float* floatData = static_cast<float*>(imageBuffer);

    runtime_init();
    Error status;

    TensorFactory<ScalarType::Float> tensor_inputs;
    const std::vector<int32_t> sizes = {1, 3, 224, 224};
    std::vector<float> floatVector(floatData, floatData + 1 * 3 * 224 * 224);
    EValue evalue_inputs(tensor_inputs.make(sizes, /*data=*/floatVector));

    NSString *filePath = [[NSBundle mainBundle] pathForResource:@"mv2_softmax" ofType:@"pte"];
    NSLog(@"   filePath: %@", filePath);
    std::string log_message = "Start logging...\n";
    ET_LOG(Info, "Hello world.");
    NSString *objcString = @(log_message.c_str());

    MemoryAllocator const_allocator{MemoryAllocator(0, nullptr)};
    const_allocator.enable_profiling("const allocator");

    MemoryAllocator runtime_allocator{
        MemoryAllocator(kRuntimeMemorySize, runtime_pool)};
    runtime_allocator.enable_profiling("runtime allocator");

    MemoryAllocator temp_allocator{MemoryAllocator(0, nullptr)};
    temp_allocator.enable_profiling("temp allocator");

    MemoryAllocator non_const_allocators[1]{
        MemoryAllocator(kMemoryAmount, activation_pool)};
    non_const_allocators[0].enable_profiling("non_const_allocators");

    HierarchicalAllocator non_const_allocator{
        HierarchicalAllocator(1, non_const_allocators)};

    MemoryManager memory_manager{MemoryManager(
        &const_allocator,
        &non_const_allocator,
        &runtime_allocator,
        &temp_allocator)};

    const char *file_name = [filePath UTF8String];

    // Open file
    FILE* file = fopen(file_name, "rb");
    unsigned long fileLen = ftell(file);
    if (!file) {
      ET_LOG(Error, "Unable to open file %s\n", file_name);
    }
    ET_LOG(Info, "Open file Finish.");
    NSString *existingString = @"open file from ";

    // Get file length
    fseek(file, 0, SEEK_END);
    size_t file_length = ftell(file);
    fseek(file, 0, SEEK_SET);
    ET_LOG(Info, "Get file len Finish.");

    // Allocate memory
    std::shared_ptr<char> file_data = std::shared_ptr<char>(new char[file_length + 1], std::default_delete<char[]>());
    if (!file_data) {
      ET_LOG(Error, "Unable to allocate memory to read file %s\n", file_name);
      fclose(file);
    }
    ET_LOG(Info, "Allocate memory Finish.");

    fread(file_data.get(), file_length, 1, file);
    ET_LOG(Info, "Load file Finish.");

    const void * program_data = file_data.get();

    const auto program = torch::executor::Program(program_data);

    if (!program.is_valid()) {
      ET_LOG(Info, "Failed to parse model file %s", file_name);
    }

    // Use the first method in the program.
    const char* method_name = nullptr;
    {
      const auto method_name_result = program.get_method_name(0);
      ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
      method_name = *method_name_result;
    }
    ET_LOG(Info, "Loading method %s", method_name);
    log_message = log_message + "Loading method " + method_name + "\n";

    Result<Method> method = program.load_method(method_name, &memory_manager);

    ET_CHECK(method.ok());
    ET_LOG(Info, "Method loaded.");
    method->set_input(evalue_inputs, 0);

    ET_LOG(Info, "Inputs prepared.");


    status = method->execute();
    ET_CHECK(status == Error::Ok);
    ET_LOG(Info, "Model executed successfully.");

    auto output_list =
        runtime_allocator.allocateList<EValue>(method->outputs_size());

    status = method->get_outputs(output_list, method->outputs_size());
    ET_CHECK(status == Error::Ok);
    //    torch::executor::util::FreeInputs(inputs);

    std::vector<std::string> categories = get_image_net_classes();
    std::stringstream category_result;
    char delimiter = ',';

    for (size_t i = 0; i < method->outputs_size(); i++) {
      auto output_tensor = output_list[i].toTensor();
      auto data_output = output_tensor.const_data_ptr<float>();
      for (size_t j = 0; j < output_list[i].toTensor().numel(); ++j) {
        ET_LOG(Info, "%f", data_output[j]);
      }
        const std::vector<float> probabilities(data_output, data_output + output_list[i].toTensor().numel());
        int k = 5;
        std::vector<IndexValuePair> top5 = findTopK(probabilities, k);
        for(int i = 0; i < k; i++) {
            printf("index: %d, value: %f , category: %s \n", top5[i].index, top5[i].value, categories[top5[i].index].c_str());
            if (i > 0) {
                category_result << delimiter;
            }
            category_result << categories[top5[i].index];

        }
    }

    ET_LOG(Info, "Finish.");


    fclose(file);
    NSMutableData* data = [NSMutableData dataWithLength:sizeof(char) * 6];
    char* buffer = (char*)[data mutableBytes];
    std::string final_text = category_result.str();
    std::strcpy(buffer, category_result.str().c_str());

    return buffer;
}


@end
