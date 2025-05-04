FROM python:3.10-slim

# Install ffmpeg (which includes ffprobe)
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all project files from current directory to container
COPY . .

# Run the external inference script
CMD ["python", "external_inference.py"]
