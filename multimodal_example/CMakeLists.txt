cmake_minimum_required(VERSION 3.19)

# Add multimodal runner executable
add_executable(run_multimodal_runner ../run_multimodal_runner.cpp)

# Use the same include directories as the main build
target_include_directories(run_multimodal_runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link against the built libraries using full paths
target_link_libraries(run_multimodal_runner PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/../extension/llm/runner/libextension_llm_runner.a
    ${CMAKE_CURRENT_BINARY_DIR}/../extension/llm/tokenizers/libtokenizers.a
    ${CMAKE_CURRENT_BINARY_DIR}/../extension/module/libextension_module.a
    ${CMAKE_CURRENT_BINARY_DIR}/../libexecutorch.a
    pthread
)