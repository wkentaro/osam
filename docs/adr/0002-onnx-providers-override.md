# ONNX execution provider override

osam picks the onnxruntime execution providers for every model session in
one place. We added `OSAM_ONNX_PROVIDERS`, a comma-separated ordered list, as
the only way to change that choice. Unset keeps the existing behaviour: CUDA
when available, else CPU.

## Context

Users on Apple Silicon, DirectML-capable Windows machines, and CUDA builds
that reject one model's ops (#43, #76, #30) had no way to influence provider
selection short of patching a private function.

## Considered options

- **Constructor or `generate` parameter.** Rejected: labelme and the CLI
  both instantiate models with no arguments, so a parameter would not reach
  the places that need it without widening every call site.
- **Auto-probe platform accelerators (CoreML on macOS, DirectML on
  Windows).** Deferred: onnxruntime falls back per node, which can be slower
  than pure CPU, so a default needs per-model measurement first.
- **Ordered env var (chosen).** Same idiom as `OSAM_BLOB_ENDPOINT`; one knob
  expresses "prefer X, then Y" and is settable by an end user of any
  downstream app.

## Consequences

- Session creation still falls back to CPU with a logged error when the
  requested list fails, so a typo degrades performance rather than crashing.
- Auto-probing can be added later behind the same resolver without changing
  the interface.
