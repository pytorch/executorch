cmake --preset llm \
      -DEXECUTORCH_BUILD_CUDA=ON \
      -DCMAKE_INSTALL_PREFIX=cmake-out \
      -DCMAKE_BUILD_TYPE=Release \
      -DEXECUTORCH_ENABLE_LOGGING=ON \
      -Bcmake-out -S.
cmake --build cmake-out -j$(nproc) --target install --config Release

# Build the Gemma3 runner
cmake -DEXECUTORCH_BUILD_CUDA=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DEXECUTORCH_ENABLE_LOGGING=ON \
      -Sexamples/models/gemma3 \
      -Bcmake-out/examples/models/gemma3/
cmake --build cmake-out/examples/models/gemma3 --target gemma3_e2e_runner --config Release

./cmake-out/examples/models/gemma3/gemma3_e2e_runner \
  --model_path /home/gasoonjia/gemma/cuda/int4/model.pte \
  --data_path /home/gasoonjia/gemma/cuda/int4/aoti_cuda_blob.ptd \
  --tokenizer_path /home/gasoonjia/gemma/cuda/tokenizer.json \
  --image_path docs/source/_static/img/et-logo.png \
  --temperature 0
