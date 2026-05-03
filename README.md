# TruthLens AI

**Multi-task misinformation detection and credibility analysis for news, narratives, and bias.**

TruthLens AI is a FastAPI + PyTorch platform for detecting misinformation, profiling bias, analyzing propaganda and narrative framing, and producing explainable credibility scores.

## Overview

TruthLens AI solves a common problem: fast, structured review of news text without turning analysis into a black box. It combines multi-task transformers, engineered linguistic features, graph reasoning, and explanation layers into one API.

## Features

### Core AI
- Fake news classification
- Bias, ideology, propaganda, emotion, and narrative analysis
- Multi-task RoBERTa encoder with task-specific heads
- Calibration and uncertainty-aware outputs
- Explainability via SHAP, LIME, and attention rollout

### Infra
- FastAPI REST service
- Health checks and model registry
- Cached model loading
- Batch inference support
- ONNX / TorchScript export support

### UX
- Structured JSON responses
- Full analysis and quick prediction modes
- Human-readable credibility summaries
- Clear error reporting

### DevOps
- Replit-friendly development workflow
- Deployment-ready server config
- YAML-based configuration
- Comprehensive tests and reports

## System Highlights

- Shared encoder, multiple heads
- Linguistic + graph + explainability stack
- One-pass analysis with rich output
- Production-oriented health and deployment paths

## Architecture Snapshot

TruthLens follows a layered pipeline: input text → preprocessing → feature engineering → multi-task model → analysis/explainability → scoring → API response.

See [documentation/ARCHITECTURE.md](documentation/ARCHITECTURE.md) and [documentation/SYSTEM_DESIGN.md](documentation/SYSTEM_DESIGN.md).

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Start the API:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 5000 --reload
```

Minimal example:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Scientists confirm a new species in the Amazon rainforest."}'
```

## Example Output

Input:
```text
Scientists confirm a new species in the Amazon rainforest.
```

Output:
```json
{
  "prediction": "REAL",
  "fake_probability": 0.09,
  "confidence": 0.91
}
```

## Project Structure

See [documentation/PROJECT_STRUCTURE.md](documentation/PROJECT_STRUCTURE.md) for the full folder map.

## Configuration

See [documentation/CONFIGURATION.md](documentation/CONFIGURATION.md).

## Deployment

See [documentation/DEPLOYMENT.md](documentation/DEPLOYMENT.md).

## Testing

```bash
pytest
```

## Documentation Index

- [documentation/API_REFERENCE.md](documentation/API_REFERENCE.md) — REST endpoint and schema reference
- [documentation/ARCHITECTURE.md](documentation/ARCHITECTURE.md) — system layer overview and data flow
- [documentation/CONFIGURATION.md](documentation/CONFIGURATION.md) — YAML settings and runtime config
- [documentation/CONTRIBUTING.md](documentation/CONTRIBUTING.md) — contribution workflow and standards
- [documentation/DEPLOYMENT.md](documentation/DEPLOYMENT.md) — local, cloud, and production deployment
- [documentation/FEATURE_ENGINEERING.md](documentation/FEATURE_ENGINEERING.md) — feature pipeline and transformations
- [documentation/MODEL_CARD.md](documentation/MODEL_CARD.md) — model purpose, data, metrics, and limits
- [documentation/PROJECT_STRUCTURE.md](documentation/PROJECT_STRUCTURE.md) — repository layout and module map
- [documentation/SYSTEM_DESIGN.md](documentation/SYSTEM_DESIGN.md) — deeper system trade-offs and design decisions
- [documentation/TRAINING_GUIDE.md](documentation/TRAINING_GUIDE.md) — dataset prep, training, and reproducibility
- [documentation/TROUBLESHOOTING.md](documentation/TROUBLESHOOTING.md) — common issues and fixes

## Contributing

See [documentation/CONTRIBUTING.md](documentation/CONTRIBUTING.md).

## License

MIT
