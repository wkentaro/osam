#!/usr/bin/env python
"""Time osam.Client.segment per model, device and prompt, skipping model loading.

    python examples/benchmark.py --device cpu --device cuda
    python examples/benchmark.py --model sam3 --model efficientsam:10m

Each cell is the median of --repeat runs after one warm-up run. New image
segments an image the client hasn't seen with the model's default prompt, so
it includes image encoding; the prompts reuse the cached embedding of one
image.
"""

import argparse
import pathlib
import statistics
import sys
import time
import typing
from typing import Any
from typing import Final

import numpy as np
import PIL.Image

import osam

IMAGE_PATH: Final = pathlib.Path(__file__).parent / "_images" / "dogs.jpg"

PROMPTS: Final[dict[str, dict[str, Any]]] = {
    "point": {"points": [[1439, 504]], "point_labels": [1]},
    "box": {"box": [1233, 376, 1649, 691]},
    "negative box": {
        "example_boxes": [[1233, 376, 1649, 691], [1400, 500, 1500, 600]],
        "example_box_labels": [1, 0],
    },
    "text": {"texts": ["dog"]},
}


def _median_ms(*, fn: typing.Callable[[int], object], repeat: int) -> float:
    times = []
    for i in range(repeat):
        t0 = time.perf_counter()
        fn(i)
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000


def _new_image(*, image: np.ndarray, i: int) -> np.ndarray:
    # A changed pixel changes the image hash, so the client encodes it again.
    image = image.copy()
    image[0, 0, 0] = (int(image[0, 0, 0]) + 1 + i) % 256
    return image


def _benchmark(
    *, client: osam.Client, model: str, device: str, image: np.ndarray, repeat: int
) -> list[str]:
    try:
        client.load(model)
    except Exception as e:
        print(f"{model} on {device}: {e}", file=sys.stderr)
        return [model, device, "fails to load"] + [""] * len(PROMPTS)

    client.segment(model, image)  # warm-up; caches the embedding
    new_image_ms = _median_ms(
        fn=lambda i: client.segment(model, _new_image(image=image, i=i)),
        repeat=repeat,
    )

    row = [model, device, f"{new_image_ms:.0f}"]
    for prompt in PROMPTS.values():
        try:
            client.segment(model, image, **prompt)  # also the warm-up
        except ValueError:
            row.append("✗")
            continue
        segment_ms = _median_ms(
            fn=lambda i: client.segment(model, image, **prompt), repeat=repeat
        )
        row.append(f"{segment_ms:.0f}")
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

    image = np.asarray(PIL.Image.open(IMAGE_PATH).convert("RGB"))

    print(f"| Model | Device | New image (ms) | {' | '.join(PROMPTS)} |")
    print("|" + " --- |" * (3 + len(PROMPTS)))
    for device in args.device or ["cpu"]:
        with osam.Client(
            in_process=True, device=typing.cast(osam.types.Device, device)
        ) as client:
            models = args.model or [m.name for m in client.list()]
            for model in models:
                row = _benchmark(
                    client=client,
                    model=model,
                    device=device,
                    image=image,
                    repeat=args.repeat,
                )
                print("| " + " | ".join(row) + " |", flush=True)


if __name__ == "__main__":
    main()
