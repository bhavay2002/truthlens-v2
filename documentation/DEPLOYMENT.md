# Deployment Guide

This document describes how to **deploy TruthLens AI** in both development and production environments.

---

## Development Server

The development server uses Uvicorn with hot-reload enabled, watching only the source directories (not the Python package library) to avoid reload loops.

**Start the development server:**

```bash
python -m uvicorn api.app:app \
  --host 0.0.0.0 \
  --port 5000 \
  --reload \
  --reload-dir api \
  --reload-dir src \
  --reload-dir config \
  --reload-dir models
```

This is configured as the default workflow in Replit and runs automatically when you press the Run button.

**Access:**
- API: `http://localhost:5000`
- Swagger docs: `http://localhost:5000/docs`
- Health check: `http://localhost:5000/health`

---

## Production Deployment on Replit

TruthLens is configured for **Replit Autoscale** deployment.

### Production Run Command

```bash
gunicorn \
  --bind=0.0.0.0:5000 \
  --reuse-port \
  -k uvicorn.workers.UvicornWorker \
  api.app:app
```

This is pre-configured in `.replit`:
```toml
[deployment]
deploymentTarget = "autoscale"
run = ["gunicorn", "--bind=0.0.0.0:5000", "--reuse-port", "-k", "uvicorn.workers.UvicornWorker", "api.app:app"]
```

### Deploying

1. Train the model first: `python main.py` (ensure `models/truthlens_model/` exists)
2. Verify the health check passes: `curl http://localhost:5000/health`
3. Click the **Deploy** button in Replit, or use the deployment workflow
4. Replit handles TLS, load balancing, health checks, and scaling automatically

The deployed app will be accessible at your `.replit.app` domain.

### Pre-Deployment Checklist

- [ ] `models/truthlens_model/` exists and contains `config.json`, `tokenizer.json`, `model.safetensors`
- [ ] `GET /health` returns `"status": "healthy"`
- [ ] `POST /predict` returns a valid response
- [ ] `config/config.yaml` has correct `model.path` and `paths.tfidf_vectorizer_path`
- [ ] All required Python packages are in `requirements.txt`

---

## Environment Variables

TruthLens does not require environment variables for basic operation. All configuration is file-based via `config/config.yaml`.

If you add external integrations (e.g., database logging, authentication, external APIs), store secrets as Replit environment variables — never hardcode them in source files.

Access environment variables in Python:
```python
import os
my_secret = os.environ.get("MY_SECRET_KEY")
```

---

## Port Configuration

| Port | Usage                   |
|------|-------------------------|
| 5000 | FastAPI / Gunicorn HTTP |

The port is mapped in `.replit`:
```toml
[[ports]]
localPort = 5000
externalPort = 80
```

Port 5000 maps to external port 80 (standard HTTP). Replit's proxy handles HTTPS termination.

---

## Gunicorn Configuration

For production, additional Gunicorn settings can improve performance:

```bash
gunicorn \
  --bind=0.0.0.0:5000 \
  --reuse-port \
  --workers=2 \
  --worker-connections=1000 \
  --timeout=120 \
  --keepalive=5 \
  -k uvicorn.workers.UvicornWorker \
  api.app:app
```

**Worker count guidance:**
- CPU-only: `2 × CPU_cores + 1` workers, but note that each worker loads the model independently — this multiplies memory usage
- GPU inference: 1–2 workers maximum (model must be loaded once per worker process)
- For memory-constrained environments: 1 worker with async request handling (UvicornWorker handles concurrency within a single process)

---

## Startup and Warmup

The model is loaded into memory on the **first prediction request**, not at startup. This means:
- First request after deployment is slower (model loading: 5–30 seconds depending on model size and hardware)
- Subsequent requests use the cached model and respond quickly

To pre-warm the model at startup, add a startup event to `api/app.py`:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    # Pre-load model at startup
    from models.inference.predictor import load_model_and_tokenizer
    try:
        load_model_and_tokenizer()
    except Exception:
        pass  # Allow startup to succeed even if model not trained yet
    yield

app = FastAPI(lifespan=lifespan, title=APP_TITLE, ...)
```

---

## Health Monitoring

Use the `/health` endpoint for deployment health checks and uptime monitoring:

```bash
curl https://your-app.replit.app/health
```

Expected response when healthy:
```json
{ "status": "healthy", "model_exists": true, "model_files_complete": true }
```

Configure your monitoring tool to alert if:
- The status field is `"degraded"` or `"unhealthy"`
- The endpoint returns a non-200 HTTP status
- Response time exceeds your SLA threshold

---

## Scaling Considerations

**Memory:**
- `roberta-base` model: ~500 MB RAM per worker process
- Full `analyze` endpoint (with LIME): adds ~200 MB peak usage during explanation generation
- Plan for at least 2–3 GB RAM in production

**Latency:**
- `/predict`: ~100–500ms per request (GPU) or ~2–10s (CPU)
- `/analyze` with LIME: ~2–15s per request depending on article length and hardware

**Throughput:**
- LIME is the main bottleneck in `/analyze` — it runs 256 perturbations per request
- `/predict` is suitable for high-throughput usage
- For batch processing, use the `models/inference/predictor.py` `predict_batch()` function directly

---

## Logs

**Development logs:** Printed to the workflow console in Replit.

**Production logs:** Available via the Replit deployment logs panel.

Log format:
```
2026-04-05 04:20:02 | INFO | src.utils.config_loader | Loading configuration from /workspace/config/config.yaml
INFO:     Started server process [891]
INFO:     Application startup complete.
INFO:     10.81.19.32:0 - "GET / HTTP/1.1" 200 OK
```

Configure log level via `config/config.yaml`:
```yaml
logging:
  log_level: INFO    # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

---

## Updating a Deployed Application

1. Make code changes in your Replit workspace
2. The development server hot-reloads automatically for `src/`, `api/`, `config/`, and `models/` changes
3. Verify changes work in development
4. Re-deploy by clicking **Deploy** again in Replit

**Important:** If you retrain the model, the new `models/truthlens_model/` files must be present before re-deploying. The production environment reads model files from the deployment snapshot.

---

## Rollback

If a deployment causes issues:
1. Use Replit's checkpoint system to revert to a previous working state
2. Or redeploy the previous version from a known-good commit

The Replit checkpoint system automatically creates snapshots before deployments. You can revert through the Replit interface.
