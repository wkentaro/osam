import datetime
import json
import sys

import click
import humanize
import imgviz
import imshow
import numpy as np
import PIL.Image
import tabulate
from loguru import logger

from osam import models
from osam.prompt import Prompt


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def cli():
    pass


@cli.command(help="list models")
def list():
    rows = []
    for model in models.MODELS:
        size = model.get_size()
        modified_at = model.get_modified_at()
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
    print(
        tabulate.tabulate(
            rows, headers=["NAME", "ID", "SIZE", "MODIFIED"], tablefmt="plain"
        )
    )


@cli.command(help="run model (`osam list` to see available models)")
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

    image_gray = imgviz.gray2rgb(imgviz.rgb2gray(image))
    prompt_visualized = PIL.Image.fromarray(image_gray)
    for point, point_label in zip(prompt["points"], prompt["point_labels"]):
        imgviz.draw.circle_(
            img=prompt_visualized,
            center=point[::-1],
            diameter=max(1, min(image.shape[:2]) // 100),
            fill=(0, 255, 0) if point_label == 1 else (255, 0, 0),
        )
    prompt_visualized = np.asarray(prompt_visualized)
    mask_visualized = imgviz.label2rgb(
        label=mask.astype(np.int32) * 2, image=image_gray, alpha=0.7
    )

    visualized = imgviz.tile(
        [
            image,
            prompt_visualized,
            mask_visualized,
        ],
        shape=(1, 3),
        border=(255, 255, 255),
        border_width=10,
    )
    imshow.imshow(
        [visualized],
        get_title_from_item=lambda x: f"{model_name} - {image_path}",
    )


if __name__ == "__main__":
    cli()
