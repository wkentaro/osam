# SAM3 benchmark

SAM3 runs offline through onnxruntime on CPU, CUDA, WebGPU and CoreML. These are the
numbers behind the speed table in the [README](../README.md#speed), measured on
2026-10-31 with the sam3-onnx v0.6.0 models.

## Inference time

Milliseconds, median of 5 runs after a warm-up, excluding model loading. Each
prompt reuses the image embedding, so a prompt column is the time per prompt
after the image is encoded once.

| Device | Encode | Point | Box | Negative box | Text |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU, Intel i9-12900K | 6563 | 83 | 1152 | 1162 | 1162 |
| CUDA, NVIDIA RTX 3080 Ti | 347 | 27 | 72 | 72 | 74 |
| WebGPU, NVIDIA RTX 3080 Ti | 741 | 57 | 199 | 196 | 203 |
| CPU, Apple M5 | 15341 | 64 | 1475 | 1489 | 1519 |
| CoreML, Apple M5 | 2466 | 61 | 375 | 376 | 376 |
| WebGPU, Apple M5 | 3691 | 46 | 610 | 621 | 604 |

On the M5, CoreML encodes the image 6.2x faster than the CPU and runs a text or
box prompt 4.0x faster. The point decoder runs on CPU on both, because CoreML
can't load its unbounded point input.

CoreML and CPU returned the same result on the M5 with the v0.5.0 models: for the text prompt "dog" and
a box prompt, all 3 masks match with an IoU of 1.0 and the scores are equal.

## Load time on the M5

| Device | Load |
| --- | ---: |
| CPU | 2.9 s |
| WebGPU | 3.2 s |
| CoreML, first load | 40 s |
| CoreML, later loads | 15 s |

On every CoreML load, osam rewrites each model's Gemm weights in memory to the
layout CoreML loads in seconds. The first load compiles the result and caches
it under `~/.cache/osam/coreml` (6.9 GB for SAM3). `osam rm sam3` deletes it.

## Setup

- Image: `examples/_images/dogs.jpg` (2560x1600). The prompts are in
  [`examples/benchmark.py`](../examples/benchmark.py).
- Linux 7.2: Python 3.10 and onnxruntime 1.23.2 for CPU, Python 3.12 and
  onnxruntime-gpu 1.31.1 for CUDA.
- MacBook Pro (Mac17,2), Apple M5, 24 GB, macOS 26.6: Python 3.10 and
  onnxruntime 1.23.2 for CPU and CoreML.
- WebGPU on both: Python 3.12, onnxruntime 1.30.0 and onnxruntime-ep-webgpu
  0.5.0, on Vulkan and Metal.

Reproduce with:

```bash
python examples/benchmark.py --model sam3 --device cpu --device coreml --device webgpu
```
