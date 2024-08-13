ARG OS_VERSION

FROM ubuntu:${OS_VERSION}

ARG OS_VERSION

ENV DEBIAN_FRONTEND noninteractive

# Install common dependencies
COPY ./common/install_base.sh install_base.sh
RUN bash ./install_base.sh && rm install_base.sh

COPY ./common/install_openssl.sh install_openssl.sh
RUN bash ./install_openssl.sh && rm install_openssl.sh
ENV OPENSSL_ROOT_DIR /opt/openssl
ENV OPENSSL_DIR /opt/openssl

# Install clang
ARG CLANG_VERSION
COPY ./common/install_clang.sh install_clang.sh
RUN bash ./install_clang.sh && rm install_clang.sh

# Install gcc
ARG GCC_VERSION
COPY ./common/install_gcc.sh install_gcc.sh
RUN bash ./install_gcc.sh && rm install_gcc.sh

# Setup buck
ARG BUCK2_VERSION
COPY ./common/install_buck.sh install_buck.sh
RUN bash ./install_buck.sh && rm install_buck.sh

# Setup user
COPY ./common/install_user.sh install_user.sh
RUN bash ./install_user.sh && rm install_user.sh

# Install docs
ARG BUILD_DOCS
COPY ./common/install_docs_reqs.sh install_docs_reqs.sh
RUN bash ./install_docs_reqs.sh && rm install_docs_reqs.sh

# Install conda and other dependencies
ARG MINICONDA_VERSION
ARG PYTHON_VERSION
ENV PYTHON_VERSION=$PYTHON_VERSION
ENV PATH /opt/conda/envs/py_$PYTHON_VERSION/bin:/opt/conda/bin:$PATH
COPY requirements-ci.txt /opt/conda/
COPY conda-env-ci.txt /opt/conda/
COPY ./common/install_conda.sh install_conda.sh
COPY ./common/utils.sh utils.sh
RUN bash ./install_conda.sh && rm install_conda.sh utils.sh /opt/conda/requirements-ci.txt /opt/conda/conda-env-ci.txt

# Install sccache before building torch
COPY ./common/install_cache.sh install_cache.sh
ENV PATH /opt/cache/bin:$PATH
COPY ./common/utils.sh utils.sh
RUN bash ./install_cache.sh && rm install_cache.sh utils.sh
ENV SCCACHE_BUCKET ossci-compiler-cache-circleci-v2
ENV SCCACHE_S3_KEY_PREFIX executorch

ARG TORCH_VERSION
COPY ./common/install_pytorch.sh install_pytorch.sh
COPY ./common/utils.sh utils.sh
RUN bash ./install_pytorch.sh && rm install_pytorch.sh utils.sh

ARG LINTRUNNER
# Install lintrunner if needed
COPY ./common/install_linter.sh install_linter.sh
COPY ./common/utils.sh utils.sh
COPY ./requirements-lintrunner.txt requirements-lintrunner.txt
RUN if [ -n "${LINTRUNNER}" ]; then bash ./install_linter.sh; fi
RUN rm install_linter.sh utils.sh requirements-lintrunner.txt

ARG ANDROID_NDK_VERSION
# Install lintrunner if needed
COPY ./common/install_android.sh install_android.sh
RUN if [ -n "${ANDROID_NDK_VERSION}" ]; then bash ./install_android.sh; fi
RUN rm install_android.sh

ARG ARM_SDK
COPY --chown=ci-user:ci-user ./arm /opt/arm
# Set up ARM SDK if needed
RUN if [ -n "${ARM_SDK}" ]; then git config --global user.email "ossci@example.com"; git config --global user.name "OSS CI"; bash /opt/arm/setup.sh --i-agree-to-the-contained-eula /opt/arm-sdk; chown -R ci-user:ci-user /opt/arm-sdk; fi

USER ci-user
CMD ["bash"]
