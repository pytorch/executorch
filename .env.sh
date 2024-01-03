conda activate executorch
export QNN_SDK_ROOT=/opt/qcom/aistack/qnn/2.14.2.230905
export EXECUTORCH_ROOT=/home/megankuo/repos/executorch
export ANDROID_NDK=/home/megankuo/android-ndk
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang/:$LD_LIBRARY_PATH
export PYTHONPATH=$EXECUTORCH_ROOT/..
export PATH="$EXECUTORCH_ROOT/build_x86_64/third-party/flatbuffers:$PATH"

