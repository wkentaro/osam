#!/usr/bin/env python

import time

import numpy as np
import PIL.Image
from loguru import logger

import osam.apis
import osam.types


def benchmark(n_times: int):
    request = osam.types.GenerateRequest(
        model="efficient-sam",
        image=np.asarray(PIL.Image.open("../_images/dogs.jpg")),
        prompt={"points": [[1280, 800]], "point_labels": [1]},
    )

    logger.info("Warming up")
    for _ in range(3):
        osam.apis.generate(request)
    logger.info("Finished warming up")

    logger.info(f"Benchmarking {n_times} times")
    elapsed_times = []
    for _ in range(n_times):
        t_start = time.time()
        osam.apis.generate(request)
        elapsed_time = time.time() - t_start
        elapsed_times.append(elapsed_time)
    logger.info(f"Average elapsed time: {np.mean(elapsed_times)} [s]")


def main():
    benchmark(n_times=10)


if __name__ == "__main__":
    main()
