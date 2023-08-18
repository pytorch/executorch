ARG OS_VERSION

FROM ubuntu:${OS_VERSION}

ARG OS_VERSION

ENV DEBIAN_FRONTEND noninteractive

# Install common dependencies
COPY ./common/install_base.sh install_base.sh
RUN bash ./install_base.sh && rm install_base.sh

# Install clang
ARG CLANG_VERSION
COPY ./common/install_clang.sh install_clang.sh
RUN bash ./install_clang.sh && rm install_clang.sh

# Setup buck
ARG BUCK2_VERSION
COPY ./common/install_buck.sh install_buck.sh
RUN bash ./install_buck.sh && rm install_buck.sh

# Setup user
COPY ./common/install_user.sh install_user.sh
RUN bash ./install_user.sh && rm install_user.sh

# Install conda and other dependencies
ARG MINICONDA_VERSION
ARG PYTHON_VERSION
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
ENV PYTHON_VERSION=$PYTHON_VERSION
ENV PATH /opt/conda/envs/py_$PYTHON_VERSION/bin:/opt/conda/bin:$PATH
COPY requirements-ci.txt /opt/conda/
COPY conda-env-ci.txt /opt/conda/
COPY ./common/install_conda.sh install_conda.sh
COPY ./common/utils.sh utils.sh
RUN bash ./install_conda.sh && rm install_conda.sh utils.sh /opt/conda/requirements-ci.txt /opt/conda/conda-env-ci.txt

USER ci-user
CMD ["bash"]
