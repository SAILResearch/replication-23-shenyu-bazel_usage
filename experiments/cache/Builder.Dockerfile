# Image with Bazelisk installed, and will be used to built projects.

FROM ubuntu:20.04

ENV PROJECT_URL=""
ENV TARGET=//:all
ENV SUBCOMMAND=build
ENV CMD_ARGS=""
ENV SHA=""

ENV TZ=Canada/Eastern

SHELL ["/bin/bash", "-c"]

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


# Install basic tools
RUN \
    # This makes add-apt-repository available.
    apt-get update && \
    apt-get -y install \
        make curl apt-utils \
        python \
        python3 \
        python-pkg-resources \
        python3-pkg-resources \
        software-properties-common \
        unzip \
        git \
        build-essential \
        nodejs npm \
        openjdk-17-jdk

ADD https://go.dev/dl/go1.19.7.linux-amd64.tar.gz go1.19.7.linux-amd64.tar.gz
RUN tar -C /usr/local -xzf go1.19.7.linux-amd64.tar.gz
ENV PATH="/usr/local/go/bin:${PATH}"

# install rust toolchains
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc
ENV PATH="/root/.cargo/bin:${PATH}"
RUN cargo --help

# Install Bazelisk
ADD https://github.com/bazelbuild/bazelisk/releases/download/v1.16.0/bazelisk-linux-amd64 /usr/local/bin/bazel


# Make Bazel executable
RUN \
    chmod +x /usr/local/bin/bazel && \
    bazel version

WORKDIR /root

ADD bazel_runner.sh /root/bazel_runner.sh
RUN chmod +x /root/bazel_runner.sh

CMD CMD_ARGS=${CMD_ARGS}; /root/bazel_runner.sh -p ${PROJECT_URL} -t ${TARGET} -s ${SUBCOMMAND} -b ${SHA}
