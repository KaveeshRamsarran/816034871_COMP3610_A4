# 1. Start from a slim Python base image to keep the container small (~150MB vs ~900MB for full)
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy dependency file first for Docker layer caching
#    If requirements.txt hasn't changed, Docker reuses the cached pip install layer
COPY requirements.txt .

# 4. Install Python dependencies without caching to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy application code and model artifacts
COPY app.py .
COPY models/ ./models/

# 6. Document which port the app exposes (does not actually publish it)
EXPOSE 8000

# 7. Start the FastAPI server using uvicorn
#    --host 0.0.0.0 is required so the container is reachable from outside
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
