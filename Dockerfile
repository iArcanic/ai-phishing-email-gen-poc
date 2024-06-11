# Lightweight Python image as the base image
FROM python:3.9-slim

# Copy Python files
COPY src/ /app/

# Set working directory
WORKDIR /app/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Start Python program
CMD ["python3", "main.py"]
