# onnxruntime build as an extra

osam no longer depends on `onnxruntime`. Users pick one onnxruntime build
through the `cpu` or `gpu` extra, or install another build (for example
`onnxruntime-directml`) next to a bare `osam`.

## Context

`onnxruntime`, `onnxruntime-gpu`, and `onnxruntime-directml` all install into
the same `onnxruntime/` package directory. With two of them installed, the
one written last wins, and uninstalling either breaks `import onnxruntime`.
A hard dependency on `onnxruntime` therefore stopped every downstream project
from offering a GPU option without risking a CPU-only or broken install.

## Considered options

- **Keep `onnxruntime` and document installing `onnxruntime-gpu` on top.**
  Rejected: any later reinstall of `onnxruntime` silently brings back the CPU
  build, and lockfile tools cannot express the swap.
- **Default to `onnxruntime-gpu` on Linux and Windows.** Rejected: it adds
  about 1 GB of CUDA libraries for users without an NVIDIA GPU.
- **`cpu` and `gpu` extras with no default (chosen).**

## Consequences

- **Breaking:** `pip install osam` alone no longer runs. Importing osam
  without an onnxruntime build raises an `ImportError` that names the extras.
- `osam[gpu]` brings the CUDA and cuDNN wheels, and osam preloads them, so no
  system CUDA install is needed.
- `osam[gpu]` requires onnxruntime-gpu 1.24.1 or later on Python 3.11+.
  1.23.2 cannot create CUDA sessions for SAM3 (#43). It is the last release
  with Python 3.10 wheels, so on 3.10 SAM3 falls back to CPU.
- CUDA sessions share one arena. With an arena per session, SAM3 runs out of
  memory on a 12 GB GPU.
