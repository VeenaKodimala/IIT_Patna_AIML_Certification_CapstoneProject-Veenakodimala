# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and corpus
COPY . .

# Pre-populate the ChromaDB collection at build time so the container
# starts with embeddings already available
RUN python ingest.py

EXPOSE 7860

# MOCK_LLM defaults to "1" (graded baseline); override with
#   docker run -e MOCK_LLM=0 -e OPENAI_API_KEY=sk-... ...
# for the optional real-LLM extension.
ENV MOCK_LLM=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
