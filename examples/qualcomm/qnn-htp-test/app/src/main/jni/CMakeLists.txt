cmake_minimum_required(VERSION 3.22.1)
project("qnn_test_jni")

add_library(qnn_test_jni SHARED qnn_test.cpp)

find_library(log-lib log)
target_link_libraries(qnn_test_jni ${log-lib} dl)
