# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0/).

## Unreleased

<!-- towncrier release notes start -->

## 0.9.5 - 2026-10-31

### Changed

- SAM3 no longer keeps a rewritten copy of its models for CoreML, which shrinks its CoreML cache from 10.3 GB to 6.9 GB; each CoreML load rewrites them in memory instead. ([#124](https://github.com/wkentaro/osam/pull/124))

## 0.9.4 - 2026-10-31

### Changed

- SAM3 encodes an image 2x faster on WebGPU (741 ms instead of 1553 ms on an RTX 3080 Ti) with the sam3-onnx v0.6.0 models, which it downloads again; on CoreML, osam rewrites them once on the first load. ([#123](https://github.com/wkentaro/osam/pull/123))

## 0.9.3 - 2026-10-31

### Added

- A `webgpu` device runs models on AMD, Intel, NVIDIA and Apple GPUs through Vulkan, Direct3D 12 or Metal; install it with `osam[webgpu]` (Python 3.11+). ([#121](https://github.com/wkentaro/osam/pull/121))
- Without a device, osam runs on WebGPU when `osam[webgpu]` finds a GPU and CUDA is unavailable; `model.device` tells which device was picked. ([#122](https://github.com/wkentaro/osam/pull/122))

## 0.9.2 - 2026-10-30

No significant changes.

## 0.9.1 - 2026-10-30

### Fixed

- CoreML runs no longer fail after an onnxruntime upgrade; osam keeps compiled CoreML models per onnxruntime version under `~/.cache/osam/coreml`. ([#118](https://github.com/wkentaro/osam/pull/118))

## 0.9.0 - 2026-10-30

### Added

- Models take `device="cpu"`, `"cuda"`, or `"coreml"` to choose where they run, and raise when that device is unavailable. With CoreML, SAM3 runs its point decoder on CPU. ([#104](https://github.com/wkentaro/osam/pull/104))
- SAM3 accepts negative example boxes as point label pairs `(4, 5)`, mixed with positive `(2, 3)` boxes and text. ([#105](https://github.com/wkentaro/osam/pull/105))
- SAM3 accepts point prompts (labels `0` and `1`) through a new point decoder. ([#107](https://github.com/wkentaro/osam/pull/107))
- Models expose `default_prompt(height, width)`, the prompt `generate` uses when a request has none. ([#108](https://github.com/wkentaro/osam/pull/108))
- `osam run` accepts `--point`, `--box`, `--negative-box`, `--text`, `--score-threshold`, `--iou-threshold`, `--max-annotations` and `--device`; thresholds apply on top of any prompt, including the default one, and `--prompt` reads JSON from `@file` or `-` (stdin). ([#108](https://github.com/wkentaro/osam/pull/108))
- `examples/benchmark.py` measures inference time per model, device and prompt; the README lists the results and the prompts each model accepts. ([#109](https://github.com/wkentaro/osam/pull/109))

### Changed

- SAM3 uses the v0.5.0 models, which are downloaded again on first use (about 3.6 GB), and box prompts take up to 16 example boxes as repeated `(2, 3)` point label pairs, with or without text. ([#103](https://github.com/wkentaro/osam/pull/103))
- SAM3 runs text and box prompts on CoreML (about 0.4 s instead of 2 s on an M5) and caches the compiled CoreML models under `~/.cache/osam/coreml`, which `osam rm` deletes. ([#109](https://github.com/wkentaro/osam/pull/109))

### Removed

- **Breaking:** `onnxruntime` is no longer a dependency; install `osam[cpu]` or `osam[gpu]`. ([#102](https://github.com/wkentaro/osam/pull/102))

### Fixed

- SAM3 runs on CUDA with `osam[gpu]`, which brings onnxruntime-gpu 1.24.1+ with its CUDA and cuDNN libraries and shares one CUDA arena across sessions. ([#102](https://github.com/wkentaro/osam/pull/102))
- `osam.apis.generate` releases the previous model before loading another, lowering peak memory when switching models. ([#103](https://github.com/wkentaro/osam/pull/103))
- SAM3 resizes images with bilinear interpolation like its PyTorch processor, raising scores that were lower than PyTorch's. ([#103](https://github.com/wkentaro/osam/pull/103))
- SAM3 honors `max_annotations`, keeping the highest-scoring annotations, and applies `score_threshold` and `max_annotations` to point prompts too. ([#108](https://github.com/wkentaro/osam/pull/108))
- SAM3 returns masks for objects cut off by the image edge, which came back empty, and clips their boxes to the image. ([#115](https://github.com/wkentaro/osam/pull/115))

## 0.8.1 - 2026-09-17

### Fixed

- Cancelling a model download interrupts stalled response reads promptly instead of waiting for the network timeout. ([#101](https://github.com/wkentaro/osam/pull/101))

## 0.8.0 - 2026-09-16

### Added

- `Blob.pull()` and `Model.pull()` accept `cancel`, a `threading.Event` that aborts a download in flight and raises `osam.types.PullCancelledError`. ([#92](https://github.com/wkentaro/osam/pull/92))
- `Blob.pull()` and `Model.pull()` accept a requests-style `timeout`, defaulting to 30 seconds, so a stalled download server no longer blocks a pull forever; requires gdown 6.3.0 or later. ([#95](https://github.com/wkentaro/osam/pull/95))

### Changed

- `osam.apis.non_maximum_suppression()` no longer downloads an ONNX file; the graph now ships with the package, so it works offline and on first call. ([#91](https://github.com/wkentaro/osam/pull/91))

### Fixed

- Stop including development-only tests and their image fixture in wheels. ([#93](https://github.com/wkentaro/osam/pull/93))
- Cancelling a stalled model download raises `PullCancelledError` when the transfer times out, without trying another mirror or reporting a generic failure on the final retry. ([#97](https://github.com/wkentaro/osam/pull/97))

## 0.7.0 - 2026-09-06

### Added

- Added `OSAM_ONNX_PROVIDERS` to choose onnxruntime execution providers as a comma-separated list tried in order, for example CoreML or DirectML. Unset keeps the CUDA-or-CPU default. ([#82](https://github.com/wkentaro/osam/pull/82))

### Removed

- `osam.types.ModelBase` and `osam.types.ModelBlob`, deprecated since July 2024, were removed; use `Model` and `Blob`. ([#80](https://github.com/wkentaro/osam/pull/80))

### Fixed

- Fixed ImageNet normalization before SAM2 image encoding, improving segmentation quality. ([#87](https://github.com/wkentaro/osam/pull/87))

## 0.6.0 - 2026-08-21

### Added

- `osam.apis.get_model_metadata` and `types.ModelMetadata` exposing each model's license name, SPDX identifier, license URL, and export source URL, all pinned to immutable revisions (#77).

### Changed

- SAM3 blobs download from a pinned Hugging Face revision instead of `main` (#77).

## 0.5.0 - 2026-06-30

### Added

- `OSAM_BLOB_ENDPOINT` to download model blobs from content-addressed mirrors. It takes a comma-separated list of endpoints (plus `direct` for the canonical URL) tried in order (#63).

### Changed

- Support Python 3.14, and pin `onnxruntime` per Python version (`<1.24` on 3.10, where cp310 wheels were dropped) (#56).

### Fixed

- Retry blob downloads with exponential backoff to survive transient fetch failures, such as HuggingFace serving a CDN error page that fails the hash check (#65).
- Pass `quiet=True` to gdown when a `progress` callback is supplied, so the callback is the single source of progress. This removes duplicate output and avoids crashes in console-less windowed builds (e.g. PyInstaller `--windowed`); CLI pulls keep gdown's progress bar (#61, #66).
- Guard `None` values in `Prompt` point serializers and validators (#59).
- Format `ValueError` messages instead of passing args positionally (#57).

## 0.4.1 - 2026-06-04

### Fixed

- Make standalone model blob cache paths Windows-safe by sanitizing `sha256:<hash>` to `sha256-<hash>`.

## 0.4.0 - 2026-04-15

### Added

- `progress` callback for `Blob.pull()` and `Model.pull()` reporting `(filename, bytes_downloaded, total_bytes)` so UIs can show a progress bar during model pulls.

### Changed

- **Breaking:** Update SAM3 ONNX models to v0.3.0; masks are now stored cropped to their bounding box instead of full-image, reducing memory usage.
- Bump `gdown` to `>=6.0.0`.

### Fixed

- Use `os.path.join` for Windows path compatibility.

## 0.3.1 - 2026-01-29

### Changed

- Bump `imgviz` to `>=2.0.0`.

## 0.3.0 - 2026-01-18

### Added

- SAM3 model support with both point and box prompts.
- Model blobs with attachments (`.onnx` + `.onnx.data` file pairs) for large models.

### Changed

- **Breaking:** `non_maximum_suppression()` now returns a 4-tuple `(boxes, scores, labels, indices)` instead of a 3-tuple, so callers can track which detections were kept.
- Drop Python 3.9 support and add Python 3.13.
- Upgrade `onnxruntime` to `>=1.23.2` and `gdown` to `>=5.2.1`.

### Fixed

- Fix ONNX external data loading on Windows by avoiding symlinks.

## 0.2.5 - 2025-07-01

### Fixed

- Remove the hack that hid `onnxruntime` error messages, so real errors surface (#33).
- Fix a missing exception in an error message.

## 0.2.4 - 2025-07-01

### Changed

- Use the common license format for the MIT License.

## 0.2.3 - 2025-03-15

### Added

- SAM2 model support (#26).
- `extra_features` on `ImageEmbedding`, and support for deserializing `ImageEmbedding.embedding`.

### Changed

- Simplify the `sam` and `efficientsam` modules, and modernize type annotations (`List` to `list`, `Tuple` to `tuple`) (#27, #28).

### Fixed

- Fix a typo in the README CLI example (`text` to `texts`) (#24).

## 0.2.2 - 2024-08-01

### Added

- `non_maximum_suppression()` in `osam.apis` (#23).

## 0.2.1 - 2024-07-30

### Changed

- Move all modules back into the `osam` package (#21).
- Make `osam serve` optional, installed with `pip install osam[serve]` (#22).

## 0.2.0 - 2024-06-29

### Added

- YOLO-World model: `osam run yoloworld --image <image_file>`.
- Receive and return image embeddings in the generate API (#19).
- Support `point_labels` 2 and 3 (bounding-box top-left and bottom-right) in `Prompt` (#18).

### Changed

- Separate core functionality into `osam-core` (#20).

### Fixed

- Support an empty `bounding_box`.

## 0.1.1 - 2024-02-13

### Added

- Try all available inference providers (#12).
- Support `<model_name>[:latest]` syntax and rename model tags to `:latest` (#17).

### Changed

- Install `onnxruntime-gpu` on non-Darwin platforms, enabling only CUDA when available (ignoring untested TensorRT, Azure, and CoreML) (#15, #16).

### Fixed

- Fix the SAM 308M model size typo.
- Avoid in-place modification of the request object in the generate function.

## 0.1.0 - 2024-02-05

- Initial release: run promptable vision models (SAM, EfficientSAM) locally via a CLI and an HTTP API.
