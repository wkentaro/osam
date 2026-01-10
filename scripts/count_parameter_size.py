#!/usr/bin/env python3

import argparse
from typing import Type

import onnx
from loguru import logger

import osam


def count_parameters(onnx_file) -> int:
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
    parser.add_argument(
        "model",
        type=str,
        choices=[model_type.name for model_type in osam.apis.registered_model_types],
    )
    args = parser.parse_args()

    model_type: Type[osam.types.Model] = osam.apis.get_model_type_by_name(
        name=args.model
    )
    model_type.pull()

    blob: osam.types.Blob
    num_params: int = 0
    for blob in model_type._blobs.values():
        logger.debug(f"Counting parameters in {blob.path}")
        num_params += count_parameters(onnx_file=blob.path)

    logger.debug(
        "Number of parameters: "
        f"{num_params} = {num_params / 1e6:.2f}M = {num_params / 1e9:.2f}B"
    )


if __name__ == "__main__":
    main()
