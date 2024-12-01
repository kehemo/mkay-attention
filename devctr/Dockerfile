FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
LABEL maintainer="Nikita Lazarev <niklz@csail.mit.edu"

ARG DEBIAN_FRONTEND=noninteractive

# Install needed packages.
RUN apt update && \
    apt -y install python3 python3-pip

RUN pip3 install torch --no-cache-dir
RUN pip3 install numpy
RUN pip3 install packaging ninja
RUN pip3 install setuptools
