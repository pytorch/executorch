1. Make sure you have setup the Ethos-U ExecuTorch dependencies by running the examples/arm/setup.sh script. See the [readme](../../README.md) for instructions on how to do the setup.

2. Build executorch from the `examples/arm` folder, cross compiled for a Cortex-M device.
```$ cmake --preset arm-baremetal \
-DCMAKE_BUILD_TYPE=Release \
-B../../cmake-out-arm ../..
cmake --build ../../cmake-out-arm --target install -j$(nproc) ````

3. Set up the build system. You need to provide path to the DEiT-Tiny pte generated in the
`examples/arm/image_classification_example/export` folder. You also need to provide an image of a dog, you can download such
image from the [HuggingFace Oxford iiit pet dataset](https://huggingface.co/datasets/timm/oxford-iiit-pet).
```
$ cmake -DCMAKE_TOOLCHAIN_FILE=$(pwd)/ethos-u-setup/arm-none-eabi-gcc.cmake -DET_PTE_FILE_PATH=<path to DEiT-Tiny compiled for Ethos-U85-256> -DIMAGE_PATH=<path to a JPG image> -Bsimple_app_deit_tiny image_classification_example/runtime
```

4. Compile the application.
```
$ cmake --build simple_app_deit_tiny -j$(nproc) -- img_class_example
```

5. Deploy the application on the Corstone-320 Fixed Virtual Platform. Assuming you have the Corstone-320 installed on your path, do the following command to deploy the application.
```
$ FVP_Corstone_SSE-320 -C mps4_board.subsystem.ethosu.num_macs=256 -C mps4_board.visualisation.disable-visualisation=1 -C vis_hdlcd.disable_visualisation=1 -C mps4_board.telnetterminal0.start_telnet=0 -C mps4_board.uart0.out_file='-' -C mps4_board.uart0.shutdown_on_eot=1 -a simple_app_deit_tiny/img_class_example -C mps4_board.subsystem.ethosu.extra_args="--fast"
```