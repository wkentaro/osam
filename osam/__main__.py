import datetime
import io
import json
import os
import sys

import click
import imgviz
import numpy as np
import PIL.Image
from loguru import logger

from . import __version__
from . import _humanize
from . import _tabulate
from . import apis
from . import types


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option(__version__)
def cli():
    logger.remove(0)
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        format="<level>{message}</level>",
        backtrace=False,
        diagnose=False,
    )
    os.makedirs(os.path.expanduser("~/.cache/osam"), exist_ok=True)
    logger.add(
        os.path.expanduser("~/.cache/osam/osam.log"), colorize=True, level="DEBUG"
    )


@cli.command(help="Help about any command")
@click.argument("subcommand", required=False, type=str)
@click.pass_context
def help(ctx, subcommand):
    if subcommand is None:
        click.echo(cli.get_help(ctx))
        return

    subcommand_obj = cli.get_command(ctx, subcommand)
    if subcommand_obj is None:
        logger.warning("Unknown subcommand {subcommand!r}", subcommand=subcommand)
        click.echo(cli.get_help(ctx))
    else:
        click.echo(subcommand_obj.get_help(ctx))


@cli.command(help="List models")
@click.option("--all", "-a", "show_all", is_flag=True, help="show all models")
def list(show_all):
    rows = []
    for model_type in apis.registered_model_types:
        size = model_type.get_size()
        modified_at = model_type.get_modified_at()

        if size is None or modified_at is None:
            if show_all:
                size = "<not pulled>"
                modified_at = "<not pulled>"
            else:
                continue
        else:
            size = _humanize.naturalsize(size)
            modified_at = _humanize.naturaltime(
                datetime.datetime.fromtimestamp(modified_at)
            )

        rows.append([model_type.name, model_type.get_id(), size, modified_at])
    click.echo(_tabulate.tabulate(rows, headers=["NAME", "ID", "SIZE", "MODIFIED"]))


@cli.command(help="Pull a model")
@click.argument("model_name", metavar="model", type=str)
def pull(model_name):
    cls = apis.get_model_type_by_name(model_name)
    logger.info("Pulling {model_name!r}...", model_name=model_name)
    cls.pull()
    logger.info("Pulled {model_name!r}", model_name=model_name)


@cli.command(help="Remove a model")
@click.argument("model_name", metavar="model", type=str)
def rm(model_name):
    cls = apis.get_model_type_by_name(model_name)
    logger.info("Removing {model_name!r}...", model_name=model_name)
    cls.remove()
    logger.info("Removed {model_name!r}", model_name=model_name)


@cli.command(help="Start server")
@click.option("--reload", is_flag=True, help="reload server on file changes")
def serve(reload):
    try:
        import uvicorn

        import osam._server  # noqa: F401

        logger.info("Starting server...")
        uvicorn.run("osam._server:app", host="127.0.0.1", port=11368, reload=reload)
    except ImportError:
        logger.error("Run `pip install osam[serve]` to use `osam serve`")
        sys.exit(1)


@cli.command(help="Run a model")
@click.argument("model_name", metavar="model", type=str)
@click.option(
    "--image",
    "image_path",
    type=click.Path(exists=True),
    help="image path",
    required=True,
)
@click.option("--prompt", type=json.loads, help="prompt")
@click.option("--json", is_flag=True, help="json output")
def run(model_name: str, image_path: str, prompt, json: bool) -> None:
    image: np.ndarray = np.asarray(PIL.Image.open(image_path))

    try:
        request: types.GenerateRequest = types.GenerateRequest(
            model=model_name, image=image, prompt=prompt
        )
        response: types.GenerateResponse = apis.generate(request=request)
    except Exception:
        logger.exception("Failed to run model")
        sys.exit(1)

    if json:
        click.echo(response.model_dump_json())
    else:
        visualization: np.ndarray

        if request.prompt and request.prompt.texts is not None:
            labels = [
                0
                if annotation.text is None
                else 1 + request.prompt.texts.index(annotation.text)
                for annotation in response.annotations
            ]
        else:
            labels = [1] * len(response.annotations)

        captions = []
        for annotation in response.annotations:
            if annotation.text is not None and annotation.score is not None:
                caption = f"{annotation.text}: {annotation.score:.2f}"
            elif annotation.text is not None:
                caption = f"{annotation.text}"
            elif annotation.score is not None:
                caption = f"{annotation.score:.2f}"
            else:
                caption = None
            captions.append(caption)

        if all(
            annotation.bounding_box is not None for annotation in response.annotations
        ):
            bboxes = [
                [
                    getattr(annotation.bounding_box, key)
                    for key in ["ymin", "xmin", "ymax", "xmax"]
                ]
                for annotation in response.annotations
            ]
        else:
            bboxes = None

        visualization = imgviz.instances2rgb(
            image=image,
            labels=labels,
            bboxes=bboxes,
            masks=[annotation.mask for annotation in response.annotations],
            captions=captions,
            alpha=0.5,
        )

        sys.stdout.buffer.write(_image_ndarray_to_data(visualization))


def _image_ndarray_to_data(ndarray: np.ndarray) -> bytes:
    pil = PIL.Image.fromarray(ndarray)
    with io.BytesIO() as f:
        pil.save(f, format="PNG")
        return f.getvalue()


if __name__ == "__main__":
    cli()
