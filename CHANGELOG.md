# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0/).

## Unreleased

<!-- towncrier release notes start -->

## 0.11.8 - 2026-11-14

### Fixed

- A private server exits as soon as its app does, instead of 5 seconds later, and no longer warns about an unclosed file. ([#298](https://github.com/wkentaro/osam/pull/298))

## 0.11.7 - 2026-11-14

### Fixed

- Removing a model unloads it first, so a server no longer keeps it loaded from deleted files. ([#297](https://github.com/wkentaro/osam/pull/297))

## 0.11.6 - 2026-11-13

### Added

- `Client.list()` and `/api/models` report each model's `download_size`: the bytes a pull downloads for the device, known before the model is pulled. ([#294](https://github.com/wkentaro/osam/pull/294))
- `Client.load()` and `/api/load` return the loaded model as `ps()` lists it, with the device it runs on, or `None` once `keep_alive=0` unloads it. ([#295](https://github.com/wkentaro/osam/pull/295))

## 0.11.5 - 2026-11-13

### Fixed

- A client given `device`, `max_loaded_models` or `auto_pull` starts its private server even with `OSAM_HOST` set, instead of raising `ValueError`. ([#292](https://github.com/wkentaro/osam/pull/292))

## 0.11.4 - 2026-11-13

### Fixed

- Closing a client with a private server no longer leaves a pipe to the server open. ([#291](https://github.com/wkentaro/osam/pull/291))

## 0.11.3 - 2026-11-13

### Added

- `Client(auto_pull=False)` makes `load` and the calls that load raise `NotPulledError` for missing or damaged model files, with the device whose files to pull, instead of downloading them, such as the CPU files after an automatic device falls back to the CPU. A server reports it as a 404 `model_not_pulled` with the device in the body. ([#288](https://github.com/wkentaro/osam/pull/288))

## 0.11.2 - 2026-11-13

### Changed

- SAM3 tracking on CoreML keeps memory attention on CoreML for every frame, instead of falling back to WebGPU after the first few. ([#285](https://github.com/wkentaro/osam/pull/285))
- `osam[cpu]` installs the WebGPU plugin on Apple silicon with macOS 14 or later and Python 3.11 or later, so SAM3 tracks a frame on an M5 in 1.8 s instead of 7.4 s. ([#286](https://github.com/wkentaro/osam/pull/286))

## 0.11.1 - 2026-11-13

### Added

- `Client.pull(cancel=event)` stops a download once the event is set, even while no data arrives, deletes its partial files, and raises `PullCancelledError`, both in process and through a server. ([#284](https://github.com/wkentaro/osam/pull/284))

## 0.11.0 - 2026-11-13

### Added

- `osam serve --host` and `--port` choose the address it listens on. ([#148](https://github.com/wkentaro/osam/pull/148))
- `osam serve` adds `GET /api/models`, `POST /api/pull` and `DELETE /api/models/{name}`, matching `osam list`, `pull` and `rm`. `/api/pull` streams its progress as NDJSON. ([#149](https://github.com/wkentaro/osam/pull/149))
- `osam serve --device` runs every request on one device, as `osam run --device` does. ([#150](https://github.com/wkentaro/osam/pull/150))
- `osam serve` keeps the embeddings of recent images, so further prompts on the same image skip the image encoder. ([#151](https://github.com/wkentaro/osam/pull/151))
- `osam serve` adds `GET /api/version`, `GET /api/ps` and a `keep_alive` field on `/api/segment` and `/api/load`, which unloads the model after five minutes idle by default, as in Ollama. `"keep_alive": 0` on `/api/load` unloads the model now. ([#156](https://github.com/wkentaro/osam/pull/156))
- `osam.Client` runs models on a private server it starts for the calling process, or on a running `osam serve` given by `host=` or `OSAM_HOST`. ([#157](https://github.com/wkentaro/osam/pull/157))
- `osam list`, `pull`, `rm` and `run` run on the `osam serve` that `OSAM_HOST` names, when set. ([#158](https://github.com/wkentaro/osam/pull/158))
- `osam serve --max-loaded-models` and `osam.Client(max_loaded_models=...)` keep more than one model loaded; a full server unloads the least recently used one first. ([#162](https://github.com/wkentaro/osam/pull/162))
- `osam.Client` sends a repeated image by its reference, and encodes images at PNG level 1, which is 4 times faster; `/api/segment` returns the image's reference, `"sha256:<hex>"`, in `image`, which a later request can send as `image` in place of the file, and answers 412 with `"code": "missing_images"` and the references in `images` when it no longer has that image. ([#163](https://github.com/wkentaro/osam/pull/163))
- `osam.Client(timeout=...)` raises `TimeoutError` when a response takes longer, and restarts a private server so a hung model stops. ([#164](https://github.com/wkentaro/osam/pull/164))
- SAM3 finds objects for every text in `prompt.texts`, not only the first, and `max_annotations` caps them together. ([#167](https://github.com/wkentaro/osam/pull/167))
- `Client.segment` takes the prompt as keyword arguments in pixels: `points`, `point_labels` and one `box` for one object, or `texts`, `example_boxes` and `example_box_labels` for every match. `Client.load` loads a model and sets its `keep_alive`. ([#168](https://github.com/wkentaro/osam/pull/168))
- `osam.non_maximum_suppression` filters boxes by their per-class scores on CPU, as `osam.apis.non_maximum_suppression` did. ([#169](https://github.com/wkentaro/osam/pull/169))
- `osam.Client(in_process=True)` runs models in the calling process, with no server, and takes the same calls as a client of a private server. ([#177](https://github.com/wkentaro/osam/pull/177))
- `osam serve --help` lists the `OSAM_SERVE_DEVICE` and `OSAM_MAX_LOADED_MODELS` environment variables, which set `--device` and `--max-loaded-models`. ([#200](https://github.com/wkentaro/osam/pull/200))
- Every error response carries a `code`, such as `"model_not_found"` or `"missing_images"`, that stays the same across releases; `osam.ResponseError.code` holds it. ([#203](https://github.com/wkentaro/osam/pull/203))
- `/api/segment` and `osam.Client.segment` take `include_masks`; `false` leaves out each annotation's mask. ([#206](https://github.com/wkentaro/osam/pull/206))
- `/api/models` and `osam.Client.list` give each model's `capabilities` and `prompt_kinds`. ([#207](https://github.com/wkentaro/osam/pull/207))
- Add `POST /api/track`, which tracks objects through a list of video frames with SAM3 and streams one NDJSON line per frame, and `Annotation.object_id`. ([#211](https://github.com/wkentaro/osam/pull/211))
- Add `Client.track`, which tracks objects through video frames on `/api/track` and sends only the frames the server lacks. `ResponseError.images` lists the image references a `missing_images` error asks for. ([#212](https://github.com/wkentaro/osam/pull/212))
- Add `osam run --video DIR`, which tracks the objects of the prompt flags from the first of a directory of frames and writes one PNG per frame with `-o`, or the NDJSON of `/api/track` with `--json`. ([#213](https://github.com/wkentaro/osam/pull/213))
- `/api/track` streams current action and frame positions while SAM3 runs, followed by finalized frames and a done or error record. `Client.track(progress=callback)` exposes progress in Python. Disconnecting or raising from the callback cancels between frames and discards a half-applied timeline. Inference errors carry their status code and conflict or missing-image metadata inside the HTTP 200 stream; the Python client preserves these as exceptions. OpenAPI documents the NDJSON records and request validation errors. The video CLI shows progress on stderr while keeping JSON stdout parseable. ([#276](https://github.com/wkentaro/osam/pull/276))

### Changed

- **Breaking:** `osam serve` returns errors as `{"error": "..."}` instead of `{"detail": ...}`. ([#156](https://github.com/wkentaro/osam/pull/156))
- `fastapi` and `uvicorn` are dependencies of osam, so the `serve` extra is removed; `pip install osam[serve]` warns that osam has no such extra and installs osam without it. ([#157](https://github.com/wkentaro/osam/pull/157))
- **Breaking:** `osam run --prompt` takes the keyword names of `Client.segment`, and `--json` no longer prints `image_embedding`. ([#168](https://github.com/wkentaro/osam/pull/168))
- An unknown model raises `LookupError` from every `Client` method on both backends; from a server it is still an `osam.ResponseError` with status 404. ([#186](https://github.com/wkentaro/osam/pull/186))
- A failed `Client.pull` raises `RuntimeError` on both backends; from a server it is still an `osam.ResponseError`. ([#187](https://github.com/wkentaro/osam/pull/187))
- YOLO-World reuses an image sent before, so `client.segment` with it sends only the image hash, and a SAM, SAM2, or EfficientSAM prompt whose mask comes out empty returns no annotation. ([#195](https://github.com/wkentaro/osam/pull/195))
- The server keeps recent image embeddings up to 1 GiB instead of the last 4, so it holds about 10 for SAM3 and over 200 for EfficientSAM. ([#208](https://github.com/wkentaro/osam/pull/208))
- Loading a model checks the SHA-256 of its files once instead of twice, which takes about 3 s off a SAM3 load. ([#229](https://github.com/wkentaro/osam/pull/229))
- The server answers 400 for a `ValueError` on every route, not only `/api/segment`. ([#235](https://github.com/wkentaro/osam/pull/235))
- `osam.non_maximum_suppression` returns the boxes it trims to `max_num_detections` by class and then from the highest score, as it returns boxes that fit, instead of by ascending score. YOLO-World annotations past `max_annotations` follow the same order. ([#242](https://github.com/wkentaro/osam/pull/242))
- SAM3 avoids copying two unused outputs of the image encoder from the GPU on CUDA. ([#254](https://github.com/wkentaro/osam/pull/254))
- SAM3 runs the sam3-onnx v0.7.0 models, in float16 on GPUs, and keeps image features on the GPU on CUDA. Image encoding takes 197 ms on an RTX 3080 Ti with CUDA and 1.2 s on an Apple M5 with CoreML, excluding text decoding, transport and output serialization. The measured first CoreML load takes 5 s with no existing compilation cache. ([#256](https://github.com/wkentaro/osam/pull/256))
- On macOS with onnxruntime 1.24 or later, SAM3 runs on CoreML unless a device or `OSAM_ONNX_PROVIDERS` is given. On an Apple M5, image encoding takes 1.2 s on CoreML and 2.4 s on WebGPU, excluding text decoding, transport and output serialization. CoreML writes a 1.8 GB compilation cache on the first load. ([#259](https://github.com/wkentaro/osam/pull/259))
- Tracking a video with SAM3 keeps 45% less memory per object per frame at 1280x800 and 59% less at 1080p, by packing the masks it keeps 8 pixels to a byte. ([#260](https://github.com/wkentaro/osam/pull/260))
- SAM3 point and box prompts now give the same mask as SAM3's own predictor: one mask instead of the best of three for two or more points or a box, with small background holes filled. ([#266](https://github.com/wkentaro/osam/pull/266))
- **Breaking:** API request bodies and prompt actions reject unknown fields with 422 instead of silently ignoring them. ([#272](https://github.com/wkentaro/osam/pull/272))
- SAM, SAM2, SAM3 and EfficientSAM point and box prompts, and SAM3 tracking for each object in each frame, find the bounding box of a mask in 0.2 ms instead of 10 ms at 2560x1600. ([#278](https://github.com/wkentaro/osam/pull/278))
- `osam run` writes its PNG files at compression level 1, so `osam run --video` writes a 960x540 frame in 27 ms instead of 160 ms, for files about 7% larger. ([#279](https://github.com/wkentaro/osam/pull/279))

### Removed

- **Breaking:** `osam.apis` is removed. Use `osam.Client`, with `in_process=True` to run models in this process, and `osam.non_maximum_suppression`. ([#159](https://github.com/wkentaro/osam/pull/159))
- **Breaking:** `osam serve` no longer serves `POST /api/generate`. Use `POST /api/segment`, and `POST /api/load` for a request with only `keep_alive`. ([#168](https://github.com/wkentaro/osam/pull/168))
- **Breaking:** `Prompt`, `GenerateRequest`, `GenerateResponse`, `ImageEmbedding` and `Model` are removed from `osam.types`. `osam.Client.segment` takes the prompt as keyword arguments. ([#168](https://github.com/wkentaro/osam/pull/168))
- **Breaking:** Remove `osam serve --reload`. ([#201](https://github.com/wkentaro/osam/pull/201))

### Fixed

- `osam list` no longer raises when a model is removed while it reads the model's size or download time. ([#144](https://github.com/wkentaro/osam/pull/144))
- `Annotation` reads back the base64 PNG mask it writes. ([#145](https://github.com/wkentaro/osam/pull/145))
- `osam serve` serves its OpenAPI schema and `/docs` page under pydantic v2. ([#146](https://github.com/wkentaro/osam/pull/146))
- `osam serve` runs inference off the event loop, and an inference error returns its message instead of a bare 500. ([#147](https://github.com/wkentaro/osam/pull/147))
- A model whose later session falls back to CPU, such as out of GPU memory, runs every session on CPU, so `Model.device` reports `cpu`. ([#162](https://github.com/wkentaro/osam/pull/162))
- YOLO-World rejects points and boxes, and SAM, SAM2 and EfficientSAM reject texts, instead of ignoring them. YOLO-World keeps its bounding boxes inside the image. ([#168](https://github.com/wkentaro/osam/pull/168))
- Raising from the `progress` callback of a pull stops the download and raises that exception, instead of retrying the download and raising `RuntimeError`. ([#177](https://github.com/wkentaro/osam/pull/177))
- `osam run --json` reports the model name the server lists, instead of resolving it from the local package, so it works with `OSAM_HOST` set to a server whose models differ. ([#188](https://github.com/wkentaro/osam/pull/188))
- Palette, grayscale, and RGBA images are converted to RGB before segmenting, so a palette PNG is segmented by its colors instead of its palette indices, and SAM models accept grayscale and RGBA images instead of raising `ValueError`. ([#194](https://github.com/wkentaro/osam/pull/194))
- `osam run` converts the image to RGB, so a palette PNG is segmented by its colors instead of its palette indices, and an RGBA image no longer fails when the result is drawn. ([#219](https://github.com/wkentaro/osam/pull/219))
- `client.segment` raises `ValueError` for an image array that is not uint8 RGB, RGBA or grayscale, such as a float or uint16 array, instead of failing later or returning a wrong result. ([#220](https://github.com/wkentaro/osam/pull/220))
- A text prompt too long for the model raises `ValueError`, a 400 from the server, instead of a 500 that restarted a private server, and YOLO-World no longer fails with `IndexError` at a `score_threshold` near 0. ([#221](https://github.com/wkentaro/osam/pull/221))
- osam declares numpy as a dependency instead of relying on other packages to install it. ([#222](https://github.com/wkentaro/osam/pull/222))
- Declare the minimum pydantic (2), FastAPI (0.100) and Pillow (9.1) versions, so pip upgrades or rejects older ones instead of installing an osam that fails to import. ([#231](https://github.com/wkentaro/osam/pull/231))
- `~/.cache/osam/osam.log` no longer has terminal color codes in the lines the CLI writes. ([#233](https://github.com/wkentaro/osam/pull/233))
- An unavailable device, such as `cuda` without CUDA, raises `RuntimeError` when `osam.Client(device=...)` or `osam serve --device` starts, instead of on the first request, where a private server restarted and answered 500. ([#234](https://github.com/wkentaro/osam/pull/234))
- A `segment` or `load` whose model download fails raises `RuntimeError` from a server too, as status 502 with the code `pull_failed`, and no longer restarts a private server and downloads again. ([#236](https://github.com/wkentaro/osam/pull/236))
- `osam list` says "1 day ago" instead of "1 days ago", and a file dated ahead of the clock reads "0 seconds ago" instead of "23 hours ago". ([#240](https://github.com/wkentaro/osam/pull/240))
- `osam list` shows a size of 0 as "0.00 B", like every other size, instead of "0B". ([#253](https://github.com/wkentaro/osam/pull/253))
- SAM box prompts segment the object in the box: the decoder gets `has_mask_input` 0 without a mask, as in segment-anything, where it got -1 and returned a mask of about 1% of the box. A box prompt also no longer gets a padding point. ([#262](https://github.com/wkentaro/osam/pull/262))
- SAM3 text and example-box masks line up with their bounding boxes, where they sat up to a native mask pixel off. ([#263](https://github.com/wkentaro/osam/pull/263))
- `osam.Client()` starts its private server from a process without a stderr, such as one under `pythonw.exe`, where it failed with "osam server exited with code 1 on start". ([#264](https://github.com/wkentaro/osam/pull/264))
- Require Pydantic 2.1 or later, as an installation with Pydantic 2.0 failed while importing osam. ([#270](https://github.com/wkentaro/osam/pull/270))
- **Breaking:** `osam run --video` refuses to replace existing output files unless `--overwrite` is given, and protects input frames even through symlinks and hard links. ([#271](https://github.com/wkentaro/osam/pull/271))
- `osam.ResponseError.result` preserves the current tracking digest returned by a 409 `base_changed` conflict. ([#272](https://github.com/wkentaro/osam/pull/272))
- Closing an in-process client releases cached video frames and tracking results, even after its model was already unloaded. ([#274](https://github.com/wkentaro/osam/pull/274))
- SAM3 runs on the CPU with onnxruntime-gpu before 1.24, which Python 3.10 installs, where on CUDA its point and box prompts failed and some text prompts found nothing; `device="cuda"` raises `RuntimeError` there. ([#277](https://github.com/wkentaro/osam/pull/277))

## 0.10.3 - 2026-11-12

### Fixed

- SAM, SAM2, EfficientSAM and SAM3 read prompt points and boxes at the prompted pixel's center, where they sat up to half a resized pixel off. SAM3 text boxes end at the object's last pixel, where they reached one past it. ([#265](https://github.com/wkentaro/osam/pull/265))

## 0.10.2 - 2026-11-06

### Fixed

- Automatically use CPU when a Windows adapter lacks Direct3D 12 support, avoiding slow WebGPU software rendering in virtual machines such as VMware Fusion. ([#179](https://github.com/wkentaro/osam/pull/179))

## 0.10.1 - 2026-11-02

### Added

- SAM3 without text prompts can now combine points with one box. ([#127](https://github.com/wkentaro/osam/pull/127))

## 0.10.0 - 2026-11-01

### Changed

- **Breaking:** SAM3 segments one object from a single box without text, as SAM and SAM2 do; several boxes or a negative box without text raise, and `texts=["visual"]` keeps finding every object like the boxes. ([#126](https://github.com/wkentaro/osam/pull/126))

## 0.9.6 - 2026-11-01

### Fixed

- Automatic device selection skips software WebGPU adapters, such as Windows' Microsoft Basic Render Driver, and runs on CPU instead of a renderer that encodes images about 25 times slower. ([#125](https://github.com/wkentaro/osam/pull/125))

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
