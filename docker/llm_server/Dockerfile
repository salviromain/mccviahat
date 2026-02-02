FROM python:3.12-slim

# Install Python libraries needed by server.py
# Keeping it here makes it very visible/understandable.
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    transformers \
    torch

WORKDIR /app
COPY server.py /app/server.py

# HuggingFace cache directory (we'll mount /cache from the host later)
ENV HF_HOME=/cache
ENV TRANSFORMERS_CACHE=/cache

EXPOSE 8000

# Start the API server
CMD ["uvicorn", "server:app", "--host=0.0.0.0", "--port=8000"]
