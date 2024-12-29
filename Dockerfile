# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt into the container at /app
COPY requirements.txt /app/

# Install any needed Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create writable directories for caching and configuration
RUN mkdir -p /app/cache /app/config && chmod -R 777 /app/cache /app/config

# Set environment variables
ENV HF_HOME=/app/cache \
    MPLCONFIGDIR=/app/config/matplotlib \
    YOLO_CONFIG_DIR=/app/config/ultralytics
    
ENV HUGGINGFACE_HUB_ENABLE_HF_UPGRADE_CHECK=false
# Copy the current directory contents into the container at /app
COPY . /app/

# Expose port 8000 for the FastAPI app
EXPOSE 8000

# Run FastAPI app using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
