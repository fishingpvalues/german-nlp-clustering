# German Free-Text Clustering SaaS

**SOTA, scalable, production-ready clustering for German customer suggestions.**  
Supports Top2Vec and UMAP+HDBSCAN, advanced keyword extraction, spaCy lemmatization, Prometheus metrics, JWT auth, and dynamic device selection (CPU, CUDA, MPS).

---

## 🚀 Features

- **German-specific embeddings** (Sentence Transformers, Hugging Face)
- **Clustering**: Top2Vec (auto-keywords) or UMAP+HDBSCAN (density-based)
- **Keyword extraction**: Top2Vec, TF-IDF, YAKE!
- **spaCy lemmatization** for robust preprocessing
- **Prometheus metrics** and **structured logging**
- **JWT authentication** for protected endpoints
- **Hot-reload** for zero-downtime model updates
- **Dynamic device selection**: CUDA, MPS (Apple Silicon), or CPU
- **Kubernetes-ready** (manifests included)

---

## 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -m spacy download de_core_news_sm
```

### M3 Max (Apple Silicon) / CUDA / CPU

- **M3 Max**: PyTorch will use MPS automatically if available.
- **CUDA**: Set `TORCH_DEVICE=cuda` and use a CUDA-enabled PyTorch wheel.
- **CPU**: Default, or set `TORCH_DEVICE=cpu`.

---

## 2. Loading Data

Prepare your data as a CSV with a column (e.g. `customer_suggestion_de`):

```python
import pandas as pd

df = pd.read_csv('customer_suggestions.csv')
texts = df['customer_suggestion_de'].astype(str).tolist()
```

---

## 3. Preprocessing (with spaCy Lemmatization)

The pipeline uses:

- Lowercasing
- Removal of HTML, numbers, punctuation
- German stopwords (NLTK + custom)
- **spaCy lemmatization** (de_core_news_sm)

Example:

```python
from app.preprocessing import preprocess_german_text

processed_texts = [preprocess_german_text(text) for text in texts if text]
```

---

## 4. Embedding Models: **Best for German Free Text**

- **[paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)**  
  - SOTA, robust for German and multilingual.
- **[jinaai/jina-embeddings-v2-base-de](https://huggingface.co/jinaai/jina-embeddings-v2-base-de)**  
  - SOTA, German/English, long context, [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
- **[Sahajtomar/German-semantic-STS-RoBERTa](https://huggingface.co/Sahajtomar/German-semantic-STS-RoBERTa)**
- **[distiluse-base-multilingual-cased-v1](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v1)**
- **[bert-base-german-cased](https://huggingface.co/bert-base-german-cased)** (needs pooling)

**Tip:** For production, benchmark on your data using [MTEB](https://huggingface.co/spaces/mteb/leaderboard).

---

## 5. Clustering

### **A. Top2Vec (Recommended for keyword-centric clustering)**

```python
from top2vec import Top2Vec

top2vec_model = Top2Vec(
    documents=processed_texts,
    embedding_model='paraphrase-multilingual-mpnet-base-v2',
    workers=-1
)
top2vec_model.save("models/german_customer_suggestions_top2vec.model")
```

### **B. UMAP + HDBSCAN (for more control)**

```python
from sentence_transformers import SentenceTransformer
import umap, hdbscan, joblib

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
embeddings = model.encode(processed_texts, show_progress_bar=True)

reducer = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
reduced_embeddings = reducer.fit_transform(embeddings)
joblib.dump(reducer, "models/umap_reducer.joblib")

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean', gen_min_span_tree=True)
clusterer.fit(reduced_embeddings)
joblib.dump(clusterer, "models/hdbscan_clusterer.joblib")
```

---

## 6. Keyword Extraction

- **Top2Vec**: Returns keywords per topic automatically.
- **HDBSCAN**: Use TF-IDF and YAKE! (see `app/keyword_extraction.py`).

---

## 7. Running the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- `/cluster-suggestions`: POST endpoint for clustering (choose `top2vec` or `hdbscan`)
- `/health`: Health and model version
- `/metrics`: Prometheus metrics
- `/reload-models`: Hot-reload models (JWT-protected)

---

## 8. Containerization & Deployment

**Build Docker image:**

```bash
docker build -t german-clustering-service:latest --build-arg TORCH_DEVICE=cpu .
```

**Run locally:**

```bash
docker run -p 8000:8000 -e MODEL_PATH="/app/models/your_model_file" german-clustering-service:latest
```

**Kubernetes:**

- See `kubernetes/deployment.yaml`, `service.yaml`, `hpa.yaml`
- Supports autoscaling, liveness/readiness probes, and model hot-reload

---

## 9. Model Versioning & Hot-Reload

- Place a `version.txt` in your `models/` directory.
- Use `/reload-models` (with JWT) to reload models and version without downtime.

---

## 10. Monitoring & Security

- **Prometheus**: `/metrics` endpoint for scraping
- **Logging**: Structured JSON logs (see `python-json-logger`)
- **JWT Auth**: Protects sensitive endpoints (see `API_SECRET_KEY` env var)

---

## 11. Best Practices & References

- **MTEB Leaderboard for German Embeddings**:  
  <https://huggingface.co/spaces/mteb/leaderboard>
- **Top2Vec**:  
  <https://github.com/ddangelov/Top2Vec>
- **HDBSCAN**:  
  <https://hdbscan.readthedocs.io/en/latest/>
- **UMAP**:  
  <https://umap-learn.readthedocs.io/en/latest/>
- **YAKE!**:  
  <https://github.com/LIAAD/yake>
- **spaCy German Model**:  
  <https://spacy.io/models/de>

---

## 12. Example: End-to-End Pipeline

```python
# 1. Load and preprocess data
import pandas as pd
from app.preprocessing import preprocess_german_text

df = pd.read_csv('customer_suggestions.csv')
texts = df['customer_suggestion_de'].astype(str).tolist()
processed_texts = [preprocess_german_text(text) for text in texts if text]

# 2. Train Top2Vec or UMAP+HDBSCAN as above
# 3. Save models to models/ directory
# 4. Start API or deploy to Kubernetes
```

---

## 13. FAQ

**Q: Which embedding model is best for German?**  
A: Start with [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) or [jinaai/jina-embeddings-v2-base-de](https://huggingface.co/jinaai/jina-embeddings-v2-base-de). Benchmark on your data.

**Q: How do I update models in production?**  
A: Replace model files, update `version.txt`, call `/reload-models` (with JWT).

**Q: How do I run on M3 Max, CUDA, or CPU?**  
A: The system auto-detects device, or set `TORCH_DEVICE` env var.

---

## 14. Hugging Face Token (SOTA Docker Practice)

**Best practice:** Do NOT bake your Hugging Face token into the Docker image. Instead, inject it at runtime as an environment variable. This ensures customer privacy and security.

- The API and model loader will use the `HF_TOKEN` environment variable if present.
- This allows each customer to use their own Hugging Face account/token for private or gated models.

**Example Docker run:**

```bash
docker run -p 8000:8000 \
  -e MODEL_PATH="/app/models/your_model_file" \
  -e HF_TOKEN="hf_your_customer_token_here" \
  german-clustering-service:latest
```

**In your code (model loading):**

```python
import os
hf_token = os.getenv("HF_TOKEN")
# Pass hf_token to SentenceTransformer or HuggingFaceHub as needed
```

**Why?**

- Keeps your own token private
- Lets customers use their own Hugging Face access (for gated/private models)
- No secrets in Docker images or code

---

## 15. Container Minification & Hardening (SlimToolkit)

For SOTA production images, use [SlimToolkit (formerly DockerSlim)](https://github.com/slimtoolkit/slim) to automatically minify and harden your Docker images:

### 1. Install SlimToolkit

```bash
curl -sL https://raw.githubusercontent.com/slimtoolkit/slim/master/scripts/install-slim.sh | sudo -E bash -
# or
brew install docker-slim
# or
pipx install docker-slim
```

### 2. Build Your Usual Image

```bash
docker build -t my-clustering-app:latest .
```

### 3. Minify Your Image

```bash
slim build --http-probe=false my-clustering-app:latest
```

- The minified image will be tagged as `my-clustering-app.slim`.
- Use `--include-path` or `--include-bin` if you need to keep extra files/binaries.

### 4. Test and Deploy

```bash
docker run -p 8000:8000 my-clustering-app.slim
```

### 5. CI/CD Example

```yaml
- name: Build Docker image
  run: docker build -t my-clustering-app:latest .

- name: Minify with SlimToolkit
  run: slim build --http-probe=false my-clustering-app:latest

- name: Push minified image
  run: docker tag my-clustering-app.slim myrepo/my-clustering-app:slim && docker push myrepo/my-clustering-app:slim
```

### References

- [SlimToolkit GitHub](https://github.com/slimtoolkit/slim)
- [Minification Examples](https://github.com/slimtoolkit/examples)

---

## 16. Additional Hardening & SOTA Security Tools

To make your container and Python app truly production-ready, apply these additional SOTA hardening tools and practices:

### 1. Vulnerability Scanning

- **[Trivy](https://github.com/aquasecurity/trivy)**: Scan your Docker images for OS and Python package vulnerabilities.

  ```bash
  trivy image my-clustering-app:latest
  trivy image my-clustering-app.slim
  ```

- **[Snyk](https://snyk.io/)**: Scan for vulnerabilities in dependencies and Dockerfiles.

  ```bash
  snyk test --docker my-clustering-app:latest
  ```

### 2. Python Code Security

- **[Bandit](https://github.com/PyCQA/bandit)**: Static analysis for Python security issues.

  ```bash
  bandit -r app/
  ```

- **[Safety](https://github.com/pyupio/safety)**: Checks for known vulnerabilities in Python dependencies.

  ```bash
  safety check -r requirements.txt
  ```

### 3. Container Runtime Security

- **Run as non-root**: In your Dockerfile, add:

  ```dockerfile
  RUN useradd -m appuser
  USER appuser
  ```

- **Enable seccomp/AppArmor**: Use Docker/K8s runtime flags to enable default or custom seccomp and AppArmor profiles.
  - [Seccomp Docs](https://docs.docker.com/engine/security/seccomp/)
  - [AppArmor Docs](https://docs.docker.com/engine/security/apparmor/)

### 4. Secrets & Environment

- **Never bake secrets into images**. Use environment variables or secret managers (Kubernetes Secrets, Vault, etc).
- **Restrict .env files to local/dev only**. Use K8s secrets or Docker secrets in production.

### 5. Network & API Security

- **Restrict exposed ports**: Only expose what's needed (e.g., 8000 for FastAPI).
- **Use network policies**: In Kubernetes, restrict which pods/services can talk to your app.
- **Enable HTTPS**: Use a reverse proxy (e.g., Traefik, NGINX) with TLS termination.
- **JWT Auth**: Already included for API protection.

### 6. Monitoring & Logging

- **Prometheus**: Already included for metrics.
- **Centralized logging**: Forward logs to ELK, Loki, or Mittwald's internal logging.
- **Audit logs**: Log sensitive actions (model reloads, auth events).

### 7. Supply Chain Security

- **Pin all dependency versions** (already done in requirements.txt).
- **Use trusted base images** (e.g., python:3.11-slim-bullseye).
- **Regularly update dependencies and base images**.

### References

- [OWASP Docker Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)
- [CNCF Cloud Native Security Whitepaper](https://github.com/cncf/tag-security/blob/main/security-whitepaper/CNCF_cloud-native-security-whitepaper-Nov2020.pdf)
- [Trivy Docs](https://aquasecurity.github.io/trivy/)
- [Bandit Docs](https://bandit.readthedocs.io/en/latest/)
- [Snyk Docs](https://docs.snyk.io/)

---

**For more details, see the code and comments in each module.**  
**For questions or improvements, open an issue or PR!**

---

**This README is SOTA, robust, and ready for production and research.**

---

Note: This project recommends [uv](https://github.com/astral-sh/uv) for fast, modern Python dependency management. If you don't have it, install with:

```bash
pip install uv
```
