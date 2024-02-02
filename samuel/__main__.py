import datetime
import io
import json
import sys

import click
import numpy as np
import PIL.Image
import uvicorn

from samuel import _humanize
from samuel import _models
from samuel import _tabulate
from samuel import apis
from samuel import types


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def cli():
    pass


@cli.command(help="Help about any command")
@click.argument("subcommand", required=False, type=str)
@click.pass_context
def help(ctx, subcommand):
    if subcommand is None:
        click.echo(cli.get_help(ctx))
        return

    subcommand_obj = cli.get_command(ctx, subcommand)
    if subcommand_obj is None:
        click.echo(f"Unknown subcommand {subcommand!r}", err=True)
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
    print(_tabulate.tabulate(rows, headers=["NAME", "ID", "SIZE", "MODIFIED"]))


@cli.command(help="Pull a model")
@click.argument("model_name", metavar="model", type=str)
def pull(model_name):
    for cls in _models.MODELS:
        if cls.name == model_name:
            break
    else:
        click.echo(f"Model {model_name} not found.", err=True)
        sys.exit(1)

    click.echo(f"Pulling {model_name!r}...", err=True)
    cls.pull()
    click.echo(f"Pulled {model_name!r}", err=True)


@cli.command(help="Remove a model")
@click.argument("model_name", metavar="model", type=str)
def rm(model_name):
    for cls in _models.MODELS:
        if cls.name == model_name:
            break
    else:
        click.echo(f"Model {model_name} not found.", err=True)
        sys.exit(1)

    click.echo(f"Removing {model_name!r}...", err=True)
    cls.remove()
    click.echo(f"Removed {model_name!r}", err=True)


@cli.command(help="Start server")
@click.option("--reload", is_flag=True, help="reload server on file changes")
def serve(reload):
    click.echo("Starting server...", err=True)
    uvicorn.run("samuel._server:app", host="127.0.0.1", port=11368, reload=reload)


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
        request: types.GenerateMaskRequest = types.GenerateMaskRequest(
            model=model_name,
            image=np.asarray(PIL.Image.open(image_path)),
            prompt=prompt,
        )
        response: types.GenerateMaskResponse = apis.generate_mask(request=request)
    except ValueError as e:
        click.echo(e, err=True)
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
