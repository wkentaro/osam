import datetime
import json
import sys

import click
import humanize
import imgviz
import numpy as np

from osam import _jsondata
from osam import _tabulate
from osam import logger
from osam import models
from osam.prompt import Prompt


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def cli():
    pass


@cli.command(help="list available models")
@click.option("--all", "-a", "show_all", is_flag=True, help="show all models")
def list(show_all):
    rows = []
    for model in models.MODELS:
        size = model.get_size()
        modified_at = model.get_modified_at()

        if not show_all and (size is None or modified_at is None):
            continue

        rows.append(
            [
                model.name,
                model.get_id(),
                "<not pulled>" if size is None else humanize.naturalsize(size),
                "<not pulled>"
                if modified_at is None
                else humanize.naturaltime(datetime.datetime.fromtimestamp(modified_at)),
            ]
        )
    print(_tabulate.tabulate(rows, headers=["NAME", "ID", "SIZE", "MODIFIED"]))


@cli.command(help="run model")
@click.argument("model_name", metavar="model", type=str)
@click.option(
    "--image",
    "image_path",
    type=click.Path(exists=True),
    help="image path",
    required=True,
)
@click.option("--prompt", type=json.loads, help="prompt", required=True)
def run(model_name, image_path, prompt):
    for cls in models.MODELS:
        if cls.name == model_name:
            break
    else:
        logger.error(f"Model {model_name} not found.")
        sys.exit(1)

    if not ("points" in prompt and "point_labels" in prompt):
        logger.error("'points' and 'point_labels' must be specified in prompt.")
        sys.exit(1)
    if len(prompt["points"]) != len(prompt["point_labels"]):
        logger.error("Length of 'points' and 'point_labels' must be same.")
        sys.exit(1)

    model = cls()
    logger.debug(f"Loaded {model_name!r}: {model}")

    image = imgviz.io.imread(image_path)
    logger.debug(f"Loaded {image_path!r}: {image.shape}, {image.dtype}")

    image_embedding = model.encode_image(image)
    logger.debug(
        f"Encoded image: {image_embedding.embedding.shape}, "
        f"{image_embedding.embedding.dtype}"
    )

    mask = model.generate_mask(
        image_embedding=image_embedding,
        prompt=Prompt(points=prompt["points"], point_labels=prompt["point_labels"]),
    )
    print(_jsondata.ndarray_to_b64data(mask.astype(np.uint8) * 255), end="")


if __name__ == "__main__":
    cli()
