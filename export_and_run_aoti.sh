./install_executorch.sh
python export_add.py
./install_executorch.sh --clean
mkdir -p cmake-out
cd cmake-out
cmake -DEXECUTORCH_BUILD_AOTI=ON \
      -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
      ..
cd ..
cmake --build cmake-out -j9
./cmake-out/executor_runner --model_path add.pte
