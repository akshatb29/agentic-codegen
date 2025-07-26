# docker/Dockerfile.go
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Go
RUN apt-get update && apt-get install -y \
    wget \
    && wget -O go.tar.gz https://go.dev/dl/go1.21.6.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go.tar.gz \
    && rm go.tar.gz \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/local/go/bin:${PATH}"

# Create projects directory inside container
RUN mkdir -p /projects
WORKDIR /projects

CMD ["/bin/bash"]
