#!/usr/bin/env python3

import argparse

import onnx
from loguru import logger


def count_parameters(onnx_file):
    model = onnx.load(onnx_file)
    total_params = 0

    for initializer in model.graph.initializer:
        param_count = 1
        for dim in initializer.dims:
            param_count *= dim
        total_params += param_count

    return total_params


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("onnx_file")
    args = parser.parse_args()

    logger.debug(f"ONNX file: {args.onnx_file!r}")
    num_params = count_parameters(onnx_file=args.onnx_file)
    logger.debug(
        "Number of parameters: "
        f"{num_params} = {num_params / 1e6:.2f}M = {num_params / 1e9:.2f}B"
    )


if __name__ == "__main__":
    main()
