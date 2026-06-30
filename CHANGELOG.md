# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0/).

## [Unreleased]

## [0.5.0] - 2026-06-30

### Added

- `OSAM_BLOB_ENDPOINT` to download model blobs from content-addressed mirrors. It takes a comma-separated list of endpoints (plus `direct` for the canonical URL) tried in order (#63).

### Changed

- Support Python 3.14, and pin `onnxruntime` per Python version (`<1.24` on 3.10, where cp310 wheels were dropped) (#56).

### Fixed

- Retry blob downloads with exponential backoff to survive transient fetch failures, such as HuggingFace serving a CDN error page that fails the hash check (#65).
- Pass `quiet=True` to gdown when a `progress` callback is supplied, so the callback is the single source of progress. This removes duplicate output and avoids crashes in console-less windowed builds (e.g. PyInstaller `--windowed`); CLI pulls keep gdown's progress bar (#61, #66).
- Guard `None` values in `Prompt` point serializers and validators (#59).
- Format `ValueError` messages instead of passing args positionally (#57).

## [0.4.1] - 2026-06-04

### Fixed

- Make standalone model blob cache paths Windows-safe by sanitizing `sha256:<hash>` to `sha256-<hash>`.

## [0.4.0] - 2026-04-15

### Added

- `progress` callback for `Blob.pull()` and `Model.pull()` reporting `(filename, bytes_downloaded, total_bytes)` so UIs can show a progress bar during model pulls.

### Changed

- **Breaking:** Update SAM3 ONNX models to v0.3.0; masks are now stored cropped to their bounding box instead of full-image, reducing memory usage.
- Bump `gdown` to `>=6.0.0`.

### Fixed

- Use `os.path.join` for Windows path compatibility.

## [0.3.1] - 2026-01-29

### Changed

- Bump `imgviz` to `>=2.0.0`.

## [0.3.0] - 2026-01-18

### Added

- SAM3 model support with both point and box prompts.
- Model blobs with attachments (`.onnx` + `.onnx.data` file pairs) for large models.

### Changed

- **Breaking:** `non_maximum_suppression()` now returns a 4-tuple `(boxes, scores, labels, indices)` instead of a 3-tuple, so callers can track which detections were kept.
- Drop Python 3.9 support and add Python 3.13.
- Upgrade `onnxruntime` to `>=1.23.2` and `gdown` to `>=5.2.1`.

### Fixed

- Fix ONNX external data loading on Windows by avoiding symlinks.

## [0.2.5] - 2025-07-01

### Fixed

- Remove the hack that hid `onnxruntime` error messages, so real errors surface (#33).
- Fix a missing exception in an error message.

## [0.2.4] - 2025-07-01

### Changed

- Use the common license format for the MIT License.

## [0.2.3] - 2025-03-15

### Added

- SAM2 model support (#26).
- `extra_features` on `ImageEmbedding`, and support for deserializing `ImageEmbedding.embedding`.

### Changed

- Simplify the `sam` and `efficientsam` modules, and modernize type annotations (`List` to `list`, `Tuple` to `tuple`) (#27, #28).

### Fixed

- Fix a typo in the README CLI example (`text` to `texts`) (#24).

## [0.2.2] - 2024-08-01

### Added

- `non_maximum_suppression()` in `osam.apis` (#23).

## [0.2.1] - 2024-07-30

### Changed

- Move all modules back into the `osam` package (#21).
- Make `osam serve` optional, installed with `pip install osam[serve]` (#22).

## [0.2.0] - 2024-06-29

### Added

- YOLO-World model: `osam run yoloworld --image <image_file>`.
- Receive and return image embeddings in the generate API (#19).
- Support `point_labels` 2 and 3 (bounding-box top-left and bottom-right) in `Prompt` (#18).

### Changed

- Separate core functionality into `osam-core` (#20).

### Fixed

- Support an empty `bounding_box`.

## [0.1.1] - 2024-02-13

### Added

- Try all available inference providers (#12).
- Support `<model_name>[:latest]` syntax and rename model tags to `:latest` (#17).

### Changed

- Install `onnxruntime-gpu` on non-Darwin platforms, enabling only CUDA when available (ignoring untested TensorRT, Azure, and CoreML) (#15, #16).

### Fixed

- Fix the SAM 308M model size typo.
- Avoid in-place modification of the request object in the generate function.

## [0.1.0] - 2024-02-05

- Initial release: run promptable vision models (SAM, EfficientSAM) locally via a CLI and an HTTP API.

[0.1.0]: https://github.com/wkentaro/osam/releases/tag/v0.1.0
[0.1.1]: https://github.com/wkentaro/osam/compare/v0.1.0...v0.1.1
[0.2.0]: https://github.com/wkentaro/osam/compare/v0.1.1...v0.2.0
[0.2.1]: https://github.com/wkentaro/osam/compare/v0.2.0...v0.2.1
[0.2.2]: https://github.com/wkentaro/osam/compare/v0.2.1...v0.2.2
[0.2.3]: https://github.com/wkentaro/osam/compare/v0.2.2...v0.2.3
[0.2.4]: https://github.com/wkentaro/osam/compare/v0.2.3...v0.2.4
[0.2.5]: https://github.com/wkentaro/osam/compare/v0.2.4...v0.2.5
[0.3.0]: https://github.com/wkentaro/osam/compare/v0.2.5...v0.3.0
[0.3.1]: https://github.com/wkentaro/osam/compare/v0.3.0...v0.3.1
[0.4.0]: https://github.com/wkentaro/osam/compare/v0.3.1...v0.4.0
[0.4.1]: https://github.com/wkentaro/osam/compare/v0.4.0...v0.4.1
[0.5.0]: https://github.com/wkentaro/osam/compare/v0.4.1...v0.5.0
[unreleased]: https://github.com/wkentaro/osam/compare/v0.5.0...HEAD
