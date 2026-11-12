# SAM3 benchmark

SAM3 runs offline through onnxruntime on CPU, CUDA, WebGPU and CoreML. These are the
numbers behind the speed table in the [README](../README.md#speed), measured on
2026-11-12 with the sam3-onnx v0.7.0 models.

## Inference time

Milliseconds, median of 5 runs after a warm-up, excluding model loading. Each
prompt reuses the image embedding, so a prompt column is the time per prompt
after the image is encoded once.

| Device | Encode | Point | Box | Negative box | Text |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU, Intel i9-12900K | 8431 | 100 | 55 | 1062 | 1085 |
| CUDA, NVIDIA RTX 3080 Ti | 197 | 30 | 28 | 31 | 33 |
| WebGPU, NVIDIA RTX 3080 Ti | 616 | 50 | 49 | 95 | 99 |
| CPU, Apple M5 | 16425 | 67 | 66 | 1319 | 1320 |
| CoreML, Apple M5 | 1163 | 44 | 46 | 330 | 333 |
| WebGPU, Apple M5 | 2390 | 42 | 43 | 319 | 320 |

GPUs run the float16 models, and CUDA and WebGPU the image encoder with fused
attention; CPUs run the float32 ones. On CUDA, the image features stay on the
GPU between the encoder and the decoders. On CoreML, only the image encoder
runs on CoreML, and the rest on WebGPU.

The v0.6.0 and v0.7.0 float32 image encoders take the same 7.8 s on the
i9-12900K, so the CPU rows don't compare across the two dates.

On CUDA, float16 finds the same 4 people on `bus.jpg` as float32 on the CPU,
with equal scores to 3 decimals and mask IoUs of 0.95 to 1.0.

## Load time on the M5

| Device | Load |
| --- | ---: |
| CPU | 2.7 s |
| WebGPU | 1.3 s |
| CoreML, first load | 5.0 s |
| CoreML, later loads | 2.1 s |

On every CoreML load, osam rewrites the image encoder's Gemm weights in memory
to the layout CoreML loads in seconds. The first load compiles the result and
caches it under `~/.cache/osam/coreml` (1.8 GB for SAM3). `osam rm sam3`
deletes it.

onnxruntime before 1.24 runs the float16 encoder on CoreML 10x slower than the
float32 one, so on those versions osam loads the float32 encoder on CoreML.

## Setup

- Image: `examples/_images/dogs.jpg` (2560x1600). The prompts are in
  [`examples/benchmark.py`](../examples/benchmark.py).
- Linux 7.2: Python 3.10 and onnxruntime 1.23.2 for CPU, Python 3.12 and
  onnxruntime-gpu 1.31.1 for CUDA.
- MacBook Pro (Mac17,2), Apple M5, 24 GB, macOS 26.6: Python 3.10 and
  onnxruntime 1.23.2 for CPU.
- WebGPU on both, and CoreML: Python 3.12, onnxruntime 1.30.0 and
  onnxruntime-ep-webgpu 0.5.0, on Vulkan and Metal.

Reproduce with:

```bash
python examples/benchmark.py --model sam3 --device cpu --device coreml --device webgpu
```

The script prints New image in place of Encode: encoding plus SAM3's default
prompt, the text `visual`.
