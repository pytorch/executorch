rm -r build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DUSE_XNNPACK_BACKEND=ON -DUSE_OPENVINO_BACKEND=ON ..
make -j 30
