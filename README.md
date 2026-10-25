# german-nlp-clustering

A small FastAPI service that assigns German customer suggestions (short free-text
sentences) to clusters and returns keywords for each cluster. It offers two methods,
selectable per request:

- `top2vec`: each text is embedded and assigned to the nearest topic vector of a
  pre-trained Top2Vec model; the first five topic words are returned.
- `hdbscan`: texts are embedded with a sentence-transformers model, reduced with a
  pre-fitted UMAP reducer, and assigned with a pre-fitted HDBSCAN clusterer. Keywords
  per cluster come from TF-IDF and YAKE. Texts labelled as noise (-1) are dropped from
  the response.

The repository contains the serving code only. The trained models are not included
(see Models below).

## Layout

- `app/main.py`: FastAPI app, `/metrics` (Prometheus), JSON logging, CORS, JWT check
  for `/reload-models`
- `app/preprocessing.py`: lowercasing, HTML/number/punctuation removal, German
  stopwords (NLTK plus a few custom ones), spaCy lemmatization with `de_core_news_sm`
- `app/clustering_logic.py`: the two clustering paths
- `app/keyword_extraction.py`: TF-IDF and YAKE keywords
- `app/model_loader.py`: model loading, device selection (`TORCH_DEVICE`, otherwise
  CUDA, then MPS, then CPU), thread-safe reload
- `app/canvas_app.py`: a minimal UI (uses the `canvas` package) that posts to the API
  on `localhost:8000`; not exercised by the Dockerfile
- `kubernetes/`: Deployment, Service and HorizontalPodAutoscaler (2 to 5 replicas).
  The image name in `deployment.yaml` is a placeholder (`your-registry/...`) and must
  be replaced.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download de_core_news_sm
uvicorn app.main:app --port 8000
```

## Models

`load_models()` reads these paths (override with environment variables):

| Variable | Default |
|---|---|
| `MODEL_PATH` | `/app/models/german_customer_suggestions_top2vec.model` |
| `UMAP_PATH` | `/app/models/umap_reducer.joblib` |
| `HDBSCAN_PATH` | `/app/models/hdbscan_clusterer.joblib` |
| `MODEL_VERSION_PATH` | `/app/models/version.txt` |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |

The embedding model is downloaded from Hugging Face on first start. The Top2Vec, UMAP
and HDBSCAN artifacts have to be trained separately and placed at those paths; there is
no training script in this repository. A method whose artifact is missing returns a
500 error. The Dockerfile copies `./models` into the image, so that directory must
exist before `docker build`.

## API

```bash
curl -X POST localhost:8000/cluster-suggestions \
  -H 'Content-Type: application/json' \
  -d '{"suggestions": ["Der Support ist zu langsam", "Mehr Speicherplatz wäre gut"], "method": "top2vec"}'
```

- `POST /cluster-suggestions`: body `{"suggestions": [...], "method": "top2vec" | "hdbscan"}`
- `GET /health`: status and contents of `version.txt`
- `GET /metrics`: Prometheus metrics (a request counter)
- `POST /reload-models`: reloads the models; requires a Bearer JWT signed with
  `API_SECRET_KEY` (HS256)

## Status

Personal project, not deployed anywhere. There are no tests. `API_SECRET_KEY` falls
back to the placeholder `supersecret` when unset, and CORS allows all origins; change
both before exposing the service. Model reload only fills models that are not loaded
yet, so replacing a model file needs a restart.

## License

MIT, see `LICENSE`.
