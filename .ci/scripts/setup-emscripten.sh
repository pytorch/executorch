
set -ex

install_nodejs() {
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
    \. "$HOME/.nvm/nvm.sh"
    nvm install 22
}

install_emscripten() {
    git clone https://github.com/emscripten-core/emsdk.git
    pushd emsdk || return
    ./emsdk install 4.0.10
    ./emsdk activate 4.0.10
    source ./emsdk_env.sh
    popd || return
}

install_nodejs
install_emscripten
