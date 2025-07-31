# Use a slim Python image as the base
# FROM python:3.11-slim
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# The build argument for the script to run (just for initial build)
ARG SCRIPT_NAME

# Copy the build script (temporary, just for building the image)
COPY ${SCRIPT_NAME} /app/build_temp.py

# Install basic dependencies that are commonly used
RUN pip install --no-cache-dir pandas numpy requests matplotlib seaborn scikit-learn

# Default command - will be overridden during execution
CMD ["python", "-u", "/app/build_temp.py"]