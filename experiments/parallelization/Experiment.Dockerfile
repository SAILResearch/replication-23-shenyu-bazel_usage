# container used to run the experiments

FROM ubuntu:20.04

ENV TZ=Canada/Eastern
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


# Install basic tools
RUN \
    # This makes add-apt-repository available.
    apt-get update && \
    apt-get -y install \
        python \
        python3 \
        python-pkg-resources \
        python3-pkg-resources \
        software-properties-common \
        unzip \
        git \
        openjdk-8-jdk

# install only docker client
COPY --from=docker:dind /usr/local/bin/docker /usr/local/bin/

WORKDIR /root
COPY parallelization_experiments.py /root/parallelization_experiments.py
COPY bazel_projects.csv /root/bazel_projects.csv
# run the container indefinitely as we will use it to run the experiments
CMD ["sleep", "infinity"]
