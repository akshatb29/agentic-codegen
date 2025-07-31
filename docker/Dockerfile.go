# # docker/Dockerfile.go
# FROM ubuntu:22.04

# ENV DEBIAN_FRONTEND=noninteractive

# # Install Go
# RUN apt-get update && apt-get install -y \
#     wget \
#     && wget -O go.tar.gz https://go.dev/dl/go1.21.6.linux-amd64.tar.gz \
#     && tar -C /usr/local -xzf go.tar.gz \
#     && rm go.tar.gz \
#     && rm -rf /var/lib/apt/lists/*

# ENV PATH="/usr/local/go/bin:${PATH}"

# # Create projects directory inside container
# RUN mkdir -p /projects
# WORKDIR /projects

# CMD ["/bin/bash"]

# docker/Dockerfile.go (for Windows containers)

# Use a Windows Server Core or Nano Server base image.
# Server Core is generally more compatible with traditional Windows applications.
# Nano Server is much smaller but has a more limited API surface.
# You might need to adjust the tag based on your specific Windows host version
# for best compatibility (e.g., ltsc2022, 1809, etc.).
FROM mcr.microsoft.com/windows/servercore:ltsc2022

# Set the default shell to PowerShell
SHELL ["powershell", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]

# Install Go
# This assumes you want Go 1.21.6. You'll need to adjust the URL if you want a different version.
# For Windows, you'd typically download the .zip or .msi installer.
# We'll use the .zip and extract it.
# Check https://go.dev/dl/ for the correct Windows AMD64 URL.
RUN Invoke-WebRequest -Uri "https://go.dev/dl/go1.21.6.windows-amd64.zip" -OutFile "go.zip"; \
    Expand-Archive -Path "go.zip" -DestinationPath "C:\"; \
    Remove-Item "go.zip"

# Set Go environment variables
ENV GOROOT="C:\go"
ENV PATH="$env:GOROOT\bin;$env:PATH"

# Create projects directory inside container
WORKDIR C:\projects

# You might want a different default command for Windows containers,
# perhaps cmd.exe, powershell.exe, or the Go executable itself.
CMD ["powershell.exe"]
