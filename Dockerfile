# Use a trusted, minimal base image
FROM python:3.11-slim-bullseye

WORKDIR /app

# Do not bake secrets into the image!
# Use environment variables or secret managers in production

ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/german_customer_suggestions_top2vec.model
ENV EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
ENV MODEL_VERSION_PATH=/app/models/version.txt

# Build arg for device type (cpu, cuda, mps)
ARG TORCH_DEVICE=cpu
ENV TORCH_DEVICE=${TORCH_DEVICE}

COPY requirements.txt .
RUN pip install uv
RUN uv pip install --no-cache-dir -r requirements.txt
RUN uv pip install python-dotenv==1.0.1

# Download spaCy German model
RUN python -m spacy download de_core_news_sm

COPY ./app ./app
COPY ./models ./models

# Optionally copy .env for local/dev (never for production secrets)
COPY .env* ./

# Add a non-root user for runtime security
RUN useradd -m appuser
USER appuser

EXPOSE 8000

# Document seccomp/AppArmor usage
# In production, run with:
#   docker run --security-opt seccomp=default --security-opt apparmor=unconfined ...
# Or use custom profiles for stricter isolation

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 