# SAM3 benchmark

SAM3 runs offline through onnxruntime on CPU, CUDA, WebGPU and CoreML. These are the
numbers behind the speed table in the [README](../README.md#speed), measured on
2026-11-12 with the sam3-onnx v0.7.0 models.

## Inference time

Milliseconds, median of 5 runs after a warm-up, excluding model loading. Each
prompt reuses the image embedding, so a prompt column is the time per prompt
after the image is encoded once.

`Encode` measures the image encoder alone. A new image with a text prompt also
runs the text decoder. Adding the published stage medians gives 230 ms on CUDA
and 1496 ms on M5 CoreML. These sums estimate inference work; they are not
measured end-to-end medians. HTTP requests also hash and encode the input,
transfer it, and serialize output masks. The table does not measure server
latency or video frame rate.

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
prompt, the text `visual`. Its cached Text column uses `dog`. Do not subtract
that column from New image to infer an encoder measurement, since the prompts
differ. The table above records stage measurements from the dated benchmark;
running the script produces its own measurements with these operation names.

## Against Meta's PyTorch release

osam runs SAM3's video tracker as ONNX models on onnxruntime. Meta's release,
[facebookresearch/sam3](https://github.com/facebookresearch/sam3) at
`660a5e9`, runs it on PyTorch. Both tracked SAM3's
[`bedroom.mp4`](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/assets/videos/bedroom.mp4)
(960x540, 200 frames) from the text prompt `person`, which finds the same 2
people. Measured on 2026-11-13 with osam 0.11.2.

| | osam | Meta |
| --- | --- | --- |
| Masks | IoU 0.99 to Meta's | Reference |
| Time per tracked frame, RTX 3080 Ti | 266 ms | 232 ms |
| Runs on macOS | Yes, on CoreML or WebGPU | No |
| Installed size | 2.6 GB | 7.3 GB |

### Masks

Mean IoU of each person's mask to Meta's over the 200 frames, with the lowest
single frame in parentheses:

| Device | Person 0 | Person 1 |
| --- | --- | --- |
| CUDA, RTX 3080 Ti | 0.991 (0.82) | 0.993 (0.97) |
| CoreML, Apple M5 | 0.990 (0.81) | 0.993 (0.97) |
| WebGPU, Apple M5 | 0.988 (0.80) | 0.991 (0.97) |

### Time per tracked frame

Each time is over the first 40 frames, the median of 5 runs after a warm-up,
without model loading. Meta's time includes reading the JPEG files, which its
`start_session` does. osam's starts from decoded frames. The README's
[video tracking table](../README.md#video-tracking-ms-per-frame) has osam on
the other devices: 890 ms on WebGPU on the RTX 3080 Ti, and 1775 ms on CoreML
and 3407 ms on WebGPU on the M5.

### macOS

Meta's video predictor moves the model to the GPU with
[`.cuda()`](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model/sam3_video_predictor.py#L53),
so it needs an NVIDIA GPU. On macOS, osam runs SAM3's image encoder and memory
attention on CoreML and the rest on WebGPU, without a flag.

### Installed size

The files of every installed package in a fresh Python 3.12 environment on
Linux, without model weights:

| | Packages | NVIDIA CUDA packages | Total |
| --- | ---: | ---: | ---: |
| `osam[gpu]` | 0.46 GB | 2.17 GB (CUDA 13) | 2.63 GB |
| Meta's install steps | 2.62 GB | 4.70 GB (CUDA 12) | 7.32 GB |

Meta's README installs `torch==2.10.0` from the `cu128` index, then the
repository with `pip install -e .`. Most of the gap is PyTorch. torch and
triton take 2.4 GB, and its CUDA packages take 2.5 GB more than osam's,
including NCCL, cuSOLVER, cuSPARSE and NVSHMEM, which onnxruntime doesn't
install.

### Reproduce

Extract the frames, then time each tracker on the first 40 and compare masks
on all 200. Run `track_meta.py` in Meta's environment.

```bash
mkdir frames frames40
ffmpeg -i bedroom.mp4 -q:v 2 -start_number 0 frames/%05d.jpg
cp frames/000[0-3]?.jpg frames40/

python track_osam.py frames40 cuda osam40.npz  # or webgpu, coreml
python track_meta.py frames40 meta40.npz

python track_osam.py frames cuda osam.npz
python track_meta.py frames meta.npz
python iou.py osam.npz meta.npz
```

`track_osam.py`:

```python
# python track_osam.py FRAMES_DIR DEVICE OUT.npz
import glob, statistics, sys, time
import numpy as np, PIL.Image, osam

frames = [np.asarray(PIL.Image.open(p).convert("RGB")) for p in sorted(glob.glob(sys.argv[1] + "/*.jpg"))]
actions = [{"type": "prompt", "frame": 0, "texts": ["person"]}]
times = []
for run in range(6):  # run 0 warms up
    client = osam.Client(in_process=True, device=sys.argv[2])
    client.load("sam3")
    t = time.perf_counter()
    result = client.track("sam3", frames, actions)
    times.append((time.perf_counter() - t) / len(frames))
    client.close()
print(f"{statistics.median(times[1:]) * 1000:.0f} ms per frame")

h, w = frames[0].shape[:2]
masks = {}
for frame in result.frames:
    for a in frame.annotations:
        m = np.zeros((h, w), bool)
        b = a.bounding_box
        m[b.ymin : b.ymin + a.mask.shape[0], b.xmin : b.xmin + a.mask.shape[1]] = a.mask
        masks[f"{frame.frame}_{a.object_id}"] = m
np.savez_compressed(sys.argv[3], **masks)
```

`track_meta.py`:

```python
# python track_meta.py FRAMES_DIR OUT.npz
import statistics, sys, time
import numpy as np, torch
from sam3.model_builder import build_sam3_video_predictor

predictor = build_sam3_video_predictor(gpus_to_use=[0])
times = []
for run in range(6):  # run 0 warms up
    t = time.perf_counter()
    session = predictor.handle_request(dict(type="start_session", resource_path=sys.argv[1]))["session_id"]
    predictor.handle_request(dict(type="add_prompt", session_id=session, frame_index=0, text="person"))
    masks, n = {}, 0
    for r in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=session)):
        n += 1
        for i, m in zip(r["outputs"]["out_obj_ids"], r["outputs"]["out_binary_masks"]):
            masks[f"{r['frame_index']}_{i}"] = np.asarray(m, bool)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t) / n)
    predictor.handle_request(dict(type="close_session", session_id=session))
print(f"{statistics.median(times[1:]) * 1000:.0f} ms per frame")
np.savez_compressed(sys.argv[2], **masks)
```

`iou.py`:

```python
# python iou.py A.npz B.npz
import sys
import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
assert set(a.files) == set(b.files), "different frames or objects"
ious = {}
for k in a.files:
    union = (a[k] | b[k]).sum()
    ious.setdefault(k.split("_")[1], []).append((a[k] & b[k]).sum() / union if union else 1.0)
for obj, v in sorted(ious.items()):
    print(f"object {obj}: mean IoU {np.mean(v):.3f}, min {np.min(v):.3f}, {len(v)} frames")
```

`size.py`, run with each environment's Python:

```python
import importlib.metadata as md
import os

cuda = other = 0
for dist in md.distributions():
    size = sum(os.path.getsize(f.locate()) for f in dist.files or [] if os.path.isfile(f.locate()))
    if dist.metadata["Name"].lower().startswith(("nvidia-", "cuda-")):
        cuda += size
    else:
        other += size
print(f"{other / 1e9:.2f} GB + {cuda / 1e9:.2f} GB CUDA = {(other + cuda) / 1e9:.2f} GB")
```
