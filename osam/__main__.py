import datetime
import io
import json
import os
import pathlib
import sys
import typing

import click
import imgviz
import numpy as np
import PIL.Image
from loguru import logger
from numpy.typing import NDArray

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


@cli.command(name="list", help="List models")
@click.option("--all", "-a", "show_all", is_flag=True, help="show all models")
def list_(show_all):
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
        import uvicorn  # type: ignore[unresolved-import]

        import osam._server  # noqa: F401

        logger.info("Starting server...")
        uvicorn.run("osam._server:app", host="127.0.0.1", port=11368, reload=reload)
    except ImportError:
        logger.error("Run `pip install osam[serve]` to use `osam serve`")
        sys.exit(1)


def _load_json(ctx, param, value: str | None) -> dict | None:
    if value is None:
        return None
    try:
        if value == "-":
            value = sys.stdin.read()
        elif value.startswith("@"):
            value = pathlib.Path(value[1:]).read_text()
        return json.loads(value)
    except (OSError, ValueError) as e:
        raise click.BadParameter(str(e))


@cli.command(help="Run a model")
@click.argument("model_name", metavar="model", type=str)
@click.option(
    "--image",
    "image_path",
    type=click.Path(exists=True),
    help="image path",
    required=True,
)
@click.option(
    "--prompt",
    callback=_load_json,
    help="prompt as JSON, @file to read it from a file, or - for stdin",
)
@click.option(
    "--point",
    "points",
    multiple=True,
    callback=lambda ctx, param, value: [_floats(v, (2, 3)) for v in value],
    help="x,y[,label]: label 1 (default) is foreground, 0 is background",
)
@click.option(
    "--box",
    "boxes",
    multiple=True,
    callback=lambda ctx, param, value: [_floats(v, (4,)) for v in value],
    help="x1,y1,x2,y2",
)
@click.option(
    "--negative-box",
    "negative_boxes",
    multiple=True,
    callback=lambda ctx, param, value: [_floats(v, (4,)) for v in value],
    help="x1,y1,x2,y2 to exclude (sam3)",
)
@click.option("--text", "texts", multiple=True, help="text prompt")
@click.option("--score-threshold", type=float, help="minimum score (sam3, yoloworld)")
@click.option("--iou-threshold", type=float, help="NMS IoU threshold (yoloworld)")
@click.option(
    "--max-annotations",
    type=click.IntRange(min=0),
    help="maximum number of annotations (sam3, yoloworld)",
)
@click.option(
    "--device",
    type=click.Choice(typing.get_args(types.Device)),
    help="execution device (default: the onnxruntime providers available)",
)
@click.option("--json", is_flag=True, help="json output")
def run(
    model_name: str,
    image_path: str,
    prompt,
    points: list[list[float]],
    boxes: list[list[float]],
    negative_boxes: list[list[float]],
    texts: tuple[str, ...],
    score_threshold: float | None,
    iou_threshold: float | None,
    max_annotations: int | None,
    device: types.Device | None,
    json: bool,
) -> None:
    image: np.ndarray = np.asarray(PIL.Image.open(image_path))

    thresholds: dict = {
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "max_annotations": max_annotations,
    }
    thresholds = {k: v for k, v in thresholds.items() if v is not None}
    prompt_flags: dict = {"texts": list(texts)} if texts else {}
    if points or boxes or negative_boxes:
        xy: list[list[float]] = []
        labels: list[int] = []
        for point in points:
            xy.append(point[:2])
            label = point[2] if len(point) == 3 else 1
            if label not in (0, 1):
                raise click.BadParameter("label must be 0 or 1", param_hint="--point")
            labels.append(int(label))
        for box_labels, box_list in [((2, 3), boxes), ((4, 5), negative_boxes)]:
            for box in box_list:
                xy += [box[:2], box[2:]]
                labels += box_labels
        prompt_flags.update(points=xy, point_labels=labels)
    if prompt is not None and prompt_flags:
        raise click.UsageError(
            "--prompt can't be combined with --point, --box, --negative-box or --text"
        )

    try:
        base: types.Prompt | None = None
        if prompt_flags:
            base = types.Prompt(**prompt_flags)
        elif prompt is not None:
            base = types.Prompt.model_validate(prompt)
        model = apis.get_model_type_by_name(model_name)(device=device)
        if base is None and thresholds:
            base = model.default_prompt(height=image.shape[0], width=image.shape[1])
        request: types.GenerateRequest = types.GenerateRequest(
            model=model_name,
            image=image,
            prompt=base and base.model_copy(update=thresholds),
        )
        response: types.GenerateResponse = model.generate(request=request)
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

        bboxes: NDArray[np.float32] | None = None
        if all(
            annotation.bounding_box is not None for annotation in response.annotations
        ):
            bboxes = np.array(
                [
                    [
                        getattr(annotation.bounding_box, key)
                        for key in ["ymin", "xmin", "ymax", "xmax"]
                    ]
                    for annotation in response.annotations
                ],
                dtype=np.float32,
            )

        masks: NDArray[np.bool_] | None = None
        if all(
            annotation.mask is not None and annotation.bounding_box is not None
            for annotation in response.annotations
        ):
            h, w = image.shape[:2]
            mask_list = []
            for annotation in response.annotations:
                full_mask = np.zeros((h, w), dtype=bool)
                assert annotation.bounding_box is not None
                bb = annotation.bounding_box
                full_mask[
                    bb.ymin : bb.ymax + 1, bb.xmin : bb.xmax + 1
                ] = annotation.mask
                mask_list.append(full_mask)
            masks = np.array(mask_list)

        visualization = imgviz.instances2rgb(
            image=image,
            labels=labels,
            bboxes=bboxes,
            masks=masks,
            captions=captions,
            alpha=0.5,
        )

        sys.stdout.buffer.write(_image_ndarray_to_data(visualization))


def _floats(value: str, lengths: tuple[int, ...]) -> list[float]:
    try:
        floats = [float(v) for v in value.split(",")]
    except ValueError:
        floats = []
    if len(floats) not in lengths:
        raise click.BadParameter(f"expected {' or '.join(map(str, lengths))} numbers")
    return floats


def _image_ndarray_to_data(ndarray: np.ndarray) -> bytes:
    pil = PIL.Image.fromarray(ndarray)
    with io.BytesIO() as f:
        pil.save(f, format="PNG")
        return f.getvalue()


if __name__ == "__main__":
    cli()
