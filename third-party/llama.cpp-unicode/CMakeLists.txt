cmake_minimum_required(VERSION 3.15)

project(unicode LANGUAGES CXX)

add_library(unicode STATIC
    src/unicode.cpp
    src/unicode-data.cpp
)

target_include_directories(unicode PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(unicode PUBLIC cxx_std_17)

install(TARGETS unicode
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)
