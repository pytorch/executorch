rm -r build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/home/dlyakhov/Projects/executorch_ov/executorch/cmake-out/" -DUSE_XNNPACK_BACKEND=ON -DUSE_OPENVINO_BACKEND=ON ..
make -j 30
