# SAM3 benchmark

SAM3 runs offline through onnxruntime on CPU, CUDA and CoreML. These are the
numbers behind the speed table in the [README](../README.md#speed), measured on
2026-10-30.

## Inference time

Milliseconds, median of 5 runs after a warm-up, excluding model loading. Each
prompt reuses the image embedding, so a prompt column is the time per prompt
after the image is encoded once.

| Device | Encode | Point | Box | Negative box | Text |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU, Intel i9-12900K | 6512 | 89 | 1172 | 1157 | 1174 |
| CUDA, NVIDIA RTX 3080 Ti | 352 | 28 | 74 | 76 | 77 |
| CPU, Apple M5 | 15233 | 64 | 1427 | 1425 | 1426 |
| CoreML, Apple M5 | 2425 | 59 | 372 | 371 | 373 |

On the M5, CoreML encodes the image 6.3x faster than the CPU and runs a text or
box prompt 3.8x faster. The point decoder runs on CPU on both, because CoreML
can't load its unbounded point input.

CoreML and CPU return the same result on the M5: for the text prompt "dog" and
a box prompt, all 3 masks match with an IoU of 1.0 and the scores are equal.

## Load time on the M5

| Device | Load |
| --- | ---: |
| CPU | 2.6 s |
| CoreML, first load | 26 to 54 s |
| CoreML, later loads | 5 to 16 s |

CoreML compiles each model on its first load and osam caches the result under
`~/.cache/osam/coreml` (6.9 GB for SAM3). `osam rm sam3` deletes it. The ranges
come from runs with the Mac idle and in use.

## Setup

- Image: `examples/_images/dogs.jpg` (2560x1600). The prompts are in
  [`examples/benchmark.py`](../examples/benchmark.py).
- Linux 7.2, Python 3.10, onnxruntime 1.23.2 for CPU and onnxruntime-gpu 1.31.1
  for CUDA.
- MacBook Pro (Mac17,2), Apple M5, 24 GB, macOS 26.6, Python 3.10,
  onnxruntime 1.23.2.

Reproduce with:

```bash
python examples/benchmark.py --model sam3 --device cpu --device coreml
```
