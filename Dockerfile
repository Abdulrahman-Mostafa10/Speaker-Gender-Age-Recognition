FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all project files from current directory to container
COPY . .

# Run the external inference script
CMD ["python", "external_inference.py"]
