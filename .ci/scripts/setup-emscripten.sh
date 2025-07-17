
set -ex

install_emscripten() {
    git clone https://github.com/emscripten-core/emsdk.git
    pushd emsdk || return
    ./emsdk install 4.0.10
    ./emsdk activate 4.0.10
    source ./emsdk_env.sh
    popd || return
}

install_emscripten
