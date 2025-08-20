#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <fstream>

// ExecutorTorch LLM Runner headers
#include <extension/llm/runner/multimodal_runner.h>
#include <extension/llm/runner/multimodal_input.h>
#include <extension/llm/runner/llm_runner_helper.h>
#include <extension/llm/runner/irunner.h>

using namespace executorch::extension::llm;

// Simple PPM image loader (P6 format)
// This is a basic example - for production use, consider OpenCV, STB, or other libraries
Image load_ppm_image(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }
    
    std::string magic;
    int width, height, max_val;
    
    file >> magic >> width >> height >> max_val;
    
    if (magic != "P6") {
        throw std::runtime_error("Only P6 PPM format is supported");
    }
    
    if (max_val != 255) {
        throw std::runtime_error("Only 8-bit images are supported");
    }
    
    // Skip the newline after the header
    file.ignore();
    
    Image image;
    image.width = width;
    image.height = height;
    image.channels = 3; // RGB
    
    size_t data_size = width * height * 3;
    image.data.resize(data_size);
    
    file.read(reinterpret_cast<char*>(image.data.data()), data_size);
    
    if (file.gcount() != static_cast<std::streamsize>(data_size)) {
        throw std::runtime_error("Failed to read complete image data");
    }
    
    std::cout << "Loaded image: " << width << "x" << height << " pixels" << std::endl;
    return image;
}

// Simple raw RGB image loader (assumes 224x224x3 format)
Image load_raw_rgb_image(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }
    
    // Assume 224x224x3 RGB format (common for vision models)
    const int width = 224;
    const int height = 224;
    const int channels = 3;
    
    Image image;
    image.width = width;
    image.height = height;
    image.channels = channels;
    
    size_t data_size = width * height * channels;
    image.data.resize(data_size);
    
    file.read(reinterpret_cast<char*>(image.data.data()), data_size);
    
    if (file.gcount() != static_cast<std::streamsize>(data_size)) {
        throw std::runtime_error("Failed to read complete image data. Expected " + 
                                std::to_string(data_size) + " bytes, got " + 
                                std::to_string(file.gcount()));
    }
    
    std::cout << "Loaded raw RGB image: " << width << "x" << height << " pixels" << std::endl;
    return image;
}

// Create a dummy test image
Image create_test_image() {
    Image image;
    image.width = 224;
    image.height = 224;
    image.channels = 3;
    
    // Create a simple gradient pattern
    image.data.resize(image.width * image.height * image.channels);
    
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            int idx = (y * image.width + x) * 3;
            image.data[idx + 0] = (x * 255) / image.width;      // Red gradient
            image.data[idx + 1] = (y * 255) / image.height;     // Green gradient
            image.data[idx + 2] = 128;                          // Constant blue
        }
    }
    
    std::cout << "Created test gradient image: " << image.width << "x" << image.height << " pixels" << std::endl;
    return image;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <model.pte> <tokenizer_path> [image_path] [image_format]" << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  model.pte      - Path to the multimodal model file" << std::endl;
    std::cout << "  tokenizer_path - Path to the tokenizer file" << std::endl;
    std::cout << "  image_path     - (Optional) Path to image file" << std::endl;
    std::cout << "  image_format   - (Optional) Image format: 'ppm', 'raw', or 'test'" << std::endl;
    std::cout << std::endl;
    std::cout << "Image formats:" << std::endl;
    std::cout << "  ppm  - PPM P6 format image" << std::endl;
    std::cout << "  raw  - Raw RGB data (224x224x3 bytes)" << std::endl;
    std::cout << "  test - Generate a test gradient image (default if no image provided)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << " model.pte tokenizer.model" << std::endl;
    std::cout << "  " << program_name << " model.pte tokenizer.model image.ppm ppm" << std::endl;
    std::cout << "  " << program_name << " model.pte tokenizer.model image.raw raw" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 5) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string tokenizer_path = argv[2];
    std::string image_path = (argc >= 4) ? argv[3] : "";
    std::string image_format = (argc >= 5) ? argv[4] : "test";
    
    try {
        std::cout << "=== Multimodal Runner Example ===" << std::endl;
        std::cout << "Model: " << model_path << std::endl;
        std::cout << "Tokenizer: " << tokenizer_path << std::endl;
        std::cout << std::endl;
        
        // Load the tokenizer
        std::cout << "Loading tokenizer..." << std::endl;
        auto tokenizer = load_tokenizer(tokenizer_path);
        if (!tokenizer || !tokenizer->is_loaded()) {
            std::cerr << "Failed to load tokenizer from: " << tokenizer_path << std::endl;
            return 1;
        }
        std::cout << "Tokenizer loaded successfully" << std::endl;
        
        // Create the multimodal runner
        std::cout << "Creating multimodal runner..." << std::endl;
        auto runner = create_multimodal_runner(model_path, std::move(tokenizer));
        if (!runner) {
            std::cerr << "Failed to create multimodal runner" << std::endl;
            return 1;
        }
        
        // Load the model
        std::cout << "Loading model..." << std::endl;
        auto load_result = runner->load();
        if (load_result != executorch::runtime::Error::Ok) {
            std::cerr << "Failed to load model" << std::endl;
            return 1;
        }
        std::cout << "Model loaded successfully!" << std::endl;
        std::cout << std::endl;
        
        // Load or create image
        Image image;
        if (!image_path.empty()) {
            std::cout << "Loading image: " << image_path << " (format: " << image_format << ")" << std::endl;
            if (image_format == "ppm") {
                image = load_ppm_image(image_path);
            } else if (image_format == "raw") {
                image = load_raw_rgb_image(image_path);
            } else {
                std::cerr << "Unknown image format: " << image_format << std::endl;
                return 1;
            }
        } else {
            std::cout << "Creating test image..." << std::endl;
            image = create_test_image();
        }
        
        // Create multimodal inputs
        std::vector<MultimodalInput> inputs;
        
        // Add text prompts and image
        inputs.emplace_back(make_text_input("Describe what you see in this image:"));
        inputs.emplace_back(make_image_input(std::move(image)));
        inputs.emplace_back(make_text_input(" Please provide a detailed description."));
        
        // Configure generation parameters
        GenerationConfig config;
        config.max_new_tokens = 150;       // Generate at most 150 new tokens
        config.temperature = 0.7f;         // Sampling temperature for creativity
        config.echo = true;                // Echo the input prompt
        config.seq_len = 2048;             // Maximum sequence length
        
        // Set up token callback to print tokens as they're generated
        auto token_callback = [](const std::string& token) {
            std::cout << token << std::flush;
        };
        
        // Set up stats callback to print generation statistics
        auto stats_callback = [](const Stats& stats) {
            std::cout << std::endl << std::endl;
            std::cout << "=== Generation Statistics ===" << std::endl;
            std::cout << "Generated tokens: " << stats.num_generated_tokens << std::endl;
            
            if (stats.model_load_end_ms >= stats.model_load_start_ms) {
                std::cout << "Model load time: " << (stats.model_load_end_ms - stats.model_load_start_ms) << "ms" << std::endl;
            }
            
            if (stats.inference_end_ms >= stats.inference_start_ms) {
                double inference_time_ms = stats.inference_end_ms - stats.inference_start_ms;
                std::cout << "Inference time: " << inference_time_ms << "ms" << std::endl;
                
                if (inference_time_ms > 0) {
                    double tokens_per_second = stats.num_generated_tokens * 1000.0 / inference_time_ms;
                    std::cout << "Tokens per second: " << tokens_per_second << std::endl;
                }
            }
            std::cout << "==============================" << std::endl;
        };
        
        std::cout << std::endl;
        std::cout << "Starting multimodal text generation..." << std::endl;
        std::cout << "Output:" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        // Generate text
        auto generate_result = runner->generate(inputs, config, token_callback, stats_callback);
        
        if (generate_result != executorch::runtime::Error::Ok) {
            std::cerr << std::endl << "Error during generation" << std::endl;
            return 1;
        }
        
        std::cout << std::endl;
        std::cout << "Generation completed successfully!" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}