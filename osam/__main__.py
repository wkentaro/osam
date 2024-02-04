import datetime
import io
import json
import os
import sys

import click
import numpy as np
import PIL.Image
import uvicorn
from loguru import logger

from osam import __version__
from osam import _humanize
from osam import _models
from osam import _tabulate
from osam import apis
from osam import types


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option(__version__)
def cli():
    logger.remove(0)
    logger.add(
        sys.stderr, level="INFO", colorize=True, format="<level>{message}</level>"
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
    for model in _models.MODELS:
        size = model.get_size()
        modified_at = model.get_modified_at()

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

        rows.append([model.name, model.get_id(), size, modified_at])
    click.echo(_tabulate.tabulate(rows, headers=["NAME", "ID", "SIZE", "MODIFIED"]))


@cli.command(help="Pull a model")
@click.argument("model_name", metavar="model", type=str)
def pull(model_name):
    for cls in _models.MODELS:
        if cls.name == model_name:
            break
    else:
        logger.warning("Model {model_name!r} not found.", model_name=model_name)
        sys.exit(1)

    logger.info("Pulling {model_name!r}...", model_name=model_name)
    cls.pull()
    logger.info("Pulled {model_name!r}", model_name=model_name)


@cli.command(help="Remove a model")
@click.argument("model_name", metavar="model", type=str)
def rm(model_name):
    for cls in _models.MODELS:
        if cls.name == model_name:
            break
    else:
        logger.warning("Model {model_name} not found.", model_name=model_name)
        sys.exit(1)

    logger.info("Removing {model_name!r}...", model_name=model_name)
    cls.remove()
    logger.info("Removed {model_name!r}", model_name=model_name)


@cli.command(help="Start server")
@click.option("--reload", is_flag=True, help="reload server on file changes")
def serve(reload):
    logger.info("Starting server...")
    uvicorn.run("osam._server:app", host="127.0.0.1", port=11368, reload=reload)


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
    try:
        request: types.GenerateRequest = types.GenerateRequest(
            model=model_name,
            image=np.asarray(PIL.Image.open(image_path)),
            prompt=prompt,
        )
        response: types.GenerateResponse = apis.generate(request=request)
    except ValueError as e:
        logger.error("{e}", e=e)
        sys.exit(1)

    if json:
        click.echo(response.model_dump_json())
    else:
        visualization: np.ndarray = (
            0.5 * request.image
            + 0.5
            * np.array([0, 255, 0])[None, None, :]
            * (response.mask > 0)[:, :, None]
        ).astype(np.uint8)
        sys.stdout.buffer.write(_image_ndarray_to_data(visualization))


def _image_ndarray_to_data(ndarray: np.ndarray) -> bytes:
    pil = PIL.Image.fromarray(ndarray)
    with io.BytesIO() as f:
        pil.save(f, format="PNG")
        return f.getvalue()


if __name__ == "__main__":
    cli()
