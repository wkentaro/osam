# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Osam (/oʊˈsɑm/) is a Python tool for running open-source promptable vision models locally, inspired by Ollama. It provides a unified interface for segmentation models (SAM, SAM2, SAM3, EfficientSAM) and detection models (YOLO-World).

## Development Commands

```bash
make setup        # Install dev dependencies (uv sync --dev)
make format       # Format code with ruff
make lint         # Check formatting with ruff
make mypy         # Type checking with mypy
make check        # Run lint + mypy
make test         # Run tests with pytest (parallel execution)
make build        # Build package with uv
```

To run a single test:
```bash
pytest osam/apis_test.py::test_generate -v
```

## Architecture

### Core API Flow

All model inference flows through `osam/apis.py`:
1. `generate(request: GenerateRequest)` is the main entry point
2. Models are loaded lazily and cached in `running_model` singleton
3. `get_model_type_by_name()` resolves model names like "sam", "sam:latest", "sam:100m"

### Model Inheritance Hierarchy

Models inherit from `types.Model` (defined in `osam/types/_model.py`):

```
types.Model (abstract base)
├── SamBase (common SAM encode/decode logic)
│   ├── Sam → Sam100m, Sam300m, Sam600m
│   └── EfficientSam → EfficientSam10m, EfficientSam30m
├── Sam2Base → Sam2Tiny, Sam2Small, Sam2BasePlus, Sam2Large
└── _YoloWorld → YoloWorldXL
```

Each model class defines:
- `name`: Model identifier (e.g., "sam:100m")
- `_blobs`: Dict mapping blob names to `Blob` objects (ONNX model files)
- `generate()`: Inference method returning `GenerateResponse`

### Blob System for Model Files

The `Blob` class (`osam/types/_blob.py`) handles model file management:
- Content-addressable storage at `~/.cache/osam/models/blobs/{sha256}`
- Automatic download from GitHub releases via gdown
- SHA256 verification on download

### Three Interfaces

1. **Python API**: `osam.apis.generate()` - typed request/response
2. **CLI**: `osam run|list|pull|rm|serve` (Click-based in `__main__.py`)
3. **HTTP Server**: FastAPI on port 11368 (`_server.py`)

### Type System

Pydantic models in `osam/types/`:
- `GenerateRequest`: model name, image (numpy or base64), prompt
- `GenerateResponse`: annotations list, image_embedding cache
- `Annotation`: mask, bounding_box, text, score
- `Prompt`: points/labels for SAM, texts for YOLO-World

### Adding a New Model

1. Create module in `osam/_models/` with class inheriting from `types.Model`
2. Define `name`, `_blobs` dict, and implement `generate()`
3. Export from `osam/_models/__init__.py`
4. Add to `registered_model_types` in `osam/apis.py`
