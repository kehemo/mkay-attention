FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
LABEL maintainer="Nikita Lazarev <niklz@csail.mit.edu"

ARG DEBIAN_FRONTEND=noninteractive

# Install needed packages.
RUN apt update && \
    apt -y install python3 pip git

# These are for the reference implementation. Not sure how Docker works lol
RUN pip install packaging torch
RUN pip install flash-attn --no-build-isolation