#!/usr/bin/env python
"""Time inference per model, device and prompt, skipping model loading.

    python examples/benchmark.py --device cpu --device cuda
    python examples/benchmark.py --model sam3 --model efficientsam:10m

Each cell is the median of --repeat runs after one warm-up run. Image
encoding is timed once per model; prompts reuse the image embedding.
"""

import argparse
import pathlib
import statistics
import sys
import time
import typing

import numpy as np
import PIL.Image

import osam

IMAGE_PATH = pathlib.Path(__file__).parent / "_images" / "dogs.jpg"

PROMPTS: dict[str, osam.types.Prompt] = {
    "point": osam.types.Prompt(
        points=np.array([[1439, 504]]), point_labels=np.array([1])
    ),
    "box": osam.types.Prompt(
        points=np.array([[1233, 376], [1649, 691]]), point_labels=np.array([2, 3])
    ),
    "negative box": osam.types.Prompt(
        points=np.array([[1233, 376], [1649, 691], [1400, 500], [1500, 600]]),
        point_labels=np.array([2, 3, 4, 5]),
    ),
    "text": osam.types.Prompt(texts=["dog"]),
}


def _median_ms(fn: typing.Callable[[], object], repeat: int) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000


def _benchmark(
    model_name: str, device: str, image: np.ndarray, repeat: int
) -> list[str]:
    try:
        model_type = osam.apis.get_model_type_by_name(model_name)
        model = model_type(device=typing.cast(osam.types.Device, device))
    except Exception as e:
        print(f"{model_name} on {device}: {e}", file=sys.stderr)
        return [model_name, device, "fails to load"] + [""] * len(PROMPTS)

    try:
        embedding = model.encode_image(image)  # also the warm-up
        encode_ms = _median_ms(lambda: model.encode_image(image), repeat)
    except NotImplementedError:
        encode_ms = None
        embedding = None

    row = [model.name, device, "-" if encode_ms is None else f"{encode_ms:.0f}"]
    for prompt in PROMPTS.values():
        request = osam.types.GenerateRequest(
            model=model.name,
            image=image if embedding is None else None,
            image_embedding=embedding,
            prompt=prompt,
        )
        try:
            model.generate(request)  # also the warm-up
        except ValueError:
            row.append("✗")
            continue
        row.append(f"{_median_ms(lambda: model.generate(request), repeat):.0f}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        action="append",
        help="model name (default: all)",
    )
    parser.add_argument(
        "--device",
        action="append",
        choices=typing.get_args(osam.types.Device),
        help="device (default: cpu)",
    )
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    model_names = args.model or [m.name for m in osam.apis.registered_model_types]
    image = np.asarray(PIL.Image.open(IMAGE_PATH).convert("RGB"))

    print(f"| Model | Device | Encode (ms) | {' | '.join(PROMPTS)} |")
    print("|" + " --- |" * (3 + len(PROMPTS)))
    for model_name in model_names:
        for device in args.device or ["cpu"]:
            row = _benchmark(model_name, device, image, repeat=args.repeat)
            print("| " + " | ".join(row) + " |", flush=True)


if __name__ == "__main__":
    main()
