import contextlib
import datetime
import http.client
import inspect
import json
import os
import pathlib
import re
import sys
import typing
from collections.abc import Iterator
from typing import Final

import click
import imgviz
import numpy as np
import PIL.Image
import pydantic
from loguru import logger
from numpy.typing import NDArray

from . import __version__
from . import _humanize
from . import _json
from . import _tabulate
from . import types
from ._client import Client
from ._client import ResponseError

# The keyword names of Client.segment that --prompt takes.
_PROMPT_KEYS: Final = frozenset(
    name
    for name in inspect.signature(Client.segment).parameters
    if name not in ("self", "model", "image", "keep_alive")
)


class _RunOutput(pydantic.BaseModel):
    model: str
    annotations: list[types.Annotation]


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option(__version__)
def cli():
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        format="<level>{message}</level>",
        backtrace=False,
        diagnose=False,
    )
    os.makedirs(os.path.expanduser("~/.cache/osam"), exist_ok=True)
    logger.add(os.path.expanduser("~/.cache/osam/osam.log"), level="DEBUG")


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
    with _client() as client:
        models = client.list()
    rows = []
    for model in models:
        if model.size is None or model.modified_at is None:
            if not show_all:
                continue
            size = modified_at = "<not pulled>"
        else:
            size = _humanize.naturalsize(size=model.size)
            modified_at = _humanize.naturaltime(
                dt=datetime.datetime.fromtimestamp(model.modified_at)
            )
        rows.append([model.name, model.id, size, modified_at])
    click.echo(
        _tabulate.tabulate(rows=rows, headers=["NAME", "ID", "SIZE", "MODIFIED"])
    )


@cli.command(help="Pull a model")
@click.argument("model_name", metavar="model", type=str)
def pull(model_name):
    status: str | None = None

    def progress(line: dict[str, typing.Any]) -> None:
        nonlocal status
        if status is not None and line["status"] != status:
            click.echo(err=True)
        status = line["status"]
        total: int | None = line.get("total")
        percent: str = f" {100 * line['completed'] // total}%" if total else ""
        click.echo(f"\r{status}{percent}", nl=False, err=True)

    with _client() as client:
        client.pull(model_name, progress=progress)
    if status is not None:
        click.echo(err=True)
    logger.info("Pulled {model_name!r}", model_name=model_name)


@cli.command(help="Remove a model")
@click.argument("model_name", metavar="model", type=str)
def rm(model_name):
    with _client() as client:
        client.remove(model_name)
    logger.info("Removed {model_name!r}", model_name=model_name)


@cli.command(help="Start server")
@click.option("--host", default="127.0.0.1", show_default=True, help="address to bind")
@click.option("--port", default=11368, show_default=True, help="port to bind")
@click.option(
    "--device",
    type=click.Choice(typing.get_args(types.Device)),
    envvar="OSAM_SERVE_DEVICE",
    show_envvar=True,
    help="execution device (default: the onnxruntime providers available)",
)
@click.option(
    "--max-loaded-models",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    envvar="OSAM_MAX_LOADED_MODELS",
    show_envvar=True,
    help="models to keep loaded at once",
)
def serve(
    host: str,
    port: int,
    device: types.Device | None,
    max_loaded_models: int,
) -> None:
    from . import _runtime
    from . import _server

    logger.info("Starting server at http://{host}:{port}/docs", host=host, port=port)
    with contextlib.closing(
        _runtime.Runtime(device=device, max_loaded_models=max_loaded_models)
    ) as runtime:
        _server.uvicorn_server(
            app=_server.create_app(runtime=runtime), host=host, port=port
        ).run()


@contextlib.contextmanager
def _client(*, device: types.Device | None = None) -> Iterator[Client]:
    # OSAM_HOST points every command at a running `osam serve`. Expected
    # failures, such as an unknown model or an unreachable host, print one line.
    try:
        with Client(device=device) as client:
            yield client
    except (OSError, http.client.HTTPException) as e:
        server: str = os.environ.get("OSAM_HOST") or "the osam server"
        raise click.ClickException(f"can't reach {server}: {e}")
    except (ResponseError, ValueError, RuntimeError) as e:
        raise click.ClickException(str(e))


def _full_name(*, client: Client, name: str) -> str:
    # The server's name for it, such as "sam3:latest" for "sam3".
    names: set[str] = {model.name for model in client.list()}
    for full_name in (name, f"{name}:latest"):
        if full_name in names:
            return full_name
    return name


def _floats_callback(*, lengths: tuple[int, ...]) -> typing.Callable:
    def callback(ctx, param, value: tuple[str, ...]) -> list[list[float]]:
        parsed = []
        for item in value:
            try:
                floats = [float(v) for v in item.split(",")]
            except ValueError:
                floats = []
            if len(floats) not in lengths:
                raise click.BadParameter(
                    f"expected {' or '.join(map(str, lengths))} numbers"
                )
            parsed.append(floats)
        return parsed

    return callback


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
@click.option("--image", "image_path", type=click.Path(exists=True), help="image path")
@click.option(
    "--video",
    "video_dir",
    type=click.Path(exists=True, file_okay=False),
    help="directory of video frames as images, in name order; the prompt is "
    "on the first (sam3)",
)
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(file_okay=False),
    help="directory to write a PNG per frame to, for --video",
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
    callback=_floats_callback(lengths=(2, 3)),
    help="x,y[,label]: label 1 (default) is foreground, 0 is background",
)
@click.option(
    "--box",
    "boxes",
    multiple=True,
    callback=_floats_callback(lengths=(4,)),
    help="x1,y1,x2,y2",
)
@click.option(
    "--negative-box",
    "negative_boxes",
    multiple=True,
    callback=_floats_callback(lengths=(4,)),
    help="x1,y1,x2,y2 to exclude (sam3, with --text)",
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
    image_path: str | None,
    video_dir: str | None,
    output_dir: str | None,
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
    if (image_path is None) == (video_dir is None):
        raise click.UsageError("give one of --image or --video")
    if video_dir is not None and output_dir is None and not json:
        raise click.UsageError("--video needs -o, or --json")

    thresholds: dict = {
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "max_annotations": max_annotations,
    }
    thresholds = {k: v for k, v in thresholds.items() if v is not None}
    prompt_flags: dict = {}
    if texts:
        # Boxes with text are examples of the objects to find.
        prompt_flags["texts"] = list(texts)
        if boxes or negative_boxes:
            prompt_flags["example_boxes"] = boxes + negative_boxes
            prompt_flags["example_box_labels"] = [1] * len(boxes) + [0] * len(
                negative_boxes
            )
        if points:
            raise click.UsageError("--point can't be combined with --text")
    else:
        if negative_boxes:
            raise click.UsageError("--negative-box needs --text")
        if len(boxes) > 1:
            raise click.UsageError("give one --box, or add --text to find every match")
        if boxes:
            prompt_flags["box"] = boxes[0]
        if points:
            labels: list[int] = []
            for point in points:
                label = point[2] if len(point) == 3 else 1
                if label not in (0, 1):
                    raise click.BadParameter(
                        "label must be 0 or 1", param_hint="--point"
                    )
                labels.append(int(label))
            prompt_flags["points"] = [point[:2] for point in points]
            prompt_flags["point_labels"] = labels
    if prompt is not None and prompt_flags:
        raise click.UsageError(
            "--prompt can't be combined with --point, --box, --negative-box or --text"
        )
    if prompt is not None:
        if not isinstance(prompt, dict) or not set(prompt) <= _PROMPT_KEYS:
            raise click.BadParameter(
                f"give an object with keys among {sorted(_PROMPT_KEYS)}",
                param_hint="--prompt",
            )
        prompt_flags = prompt

    if video_dir is not None:
        if thresholds:
            raise click.UsageError("--video takes no thresholds")
        _track(
            model_name=model_name,
            video_dir=video_dir,
            output_dir=output_dir,
            prompt=prompt_flags,
            device=device,
            json=json,
        )
        return

    assert image_path is not None
    # RGB, as a palette image's array holds palette indices, and the
    # visualization draws on three channels.
    image: np.ndarray = np.asarray(PIL.Image.open(image_path).convert("RGB"))
    try:
        with _client(device=device) as client:
            annotations: list[types.Annotation] = client.segment(
                model_name, image, **(prompt_flags | thresholds)
            )
            if json:
                model_name = _full_name(client=client, name=model_name)
    except click.ClickException:
        raise
    except Exception:
        logger.exception("Failed to run model")
        sys.exit(1)

    if json:
        click.echo(
            _RunOutput(
                model=model_name,
                annotations=annotations,
            ).model_dump_json()
        )
    else:
        run_texts: list[str] | None = prompt_flags.get("texts")
        if run_texts:
            instance_labels = [
                0 if annotation.text is None else 1 + run_texts.index(annotation.text)
                for annotation in annotations
            ]
        else:
            instance_labels = [1] * len(annotations)

        captions = []
        for annotation in annotations:
            if annotation.text is not None and annotation.score is not None:
                caption = f"{annotation.text}: {annotation.score:.2f}"
            elif annotation.text is not None:
                caption = f"{annotation.text}"
            elif annotation.score is not None:
                caption = f"{annotation.score:.2f}"
            else:
                caption = None
            captions.append(caption)

        visualization: np.ndarray = _visualize(
            image=image,
            annotations=annotations,
            labels=instance_labels,
            captions=captions,
        )
        sys.stdout.buffer.write(_json.image_ndarray_to_png(ndarray=visualization))


def _visualize(
    *,
    image: np.ndarray,
    annotations: list[types.Annotation],
    labels: list[int],
    captions: list[str | None],
    colormap: NDArray[np.uint8] | None = None,
) -> np.ndarray:
    bboxes: NDArray[np.float32] | None = None
    if all(annotation.bounding_box is not None for annotation in annotations):
        bboxes = np.array(
            [
                [
                    getattr(annotation.bounding_box, key)
                    for key in ["ymin", "xmin", "ymax", "xmax"]
                ]
                for annotation in annotations
            ],
            dtype=np.float32,
        )

    masks: NDArray[np.bool_] | None = None
    if all(
        annotation.mask is not None and annotation.bounding_box is not None
        for annotation in annotations
    ):
        h, w = image.shape[:2]
        mask_list = []
        for annotation in annotations:
            full_mask = np.zeros((h, w), dtype=bool)
            assert annotation.bounding_box is not None
            bb = annotation.bounding_box
            full_mask[bb.ymin : bb.ymax + 1, bb.xmin : bb.xmax + 1] = annotation.mask
            mask_list.append(full_mask)
        masks = np.array(mask_list)

    return imgviz.instances2rgb(
        image=image,
        labels=labels,
        bboxes=bboxes,
        masks=masks,
        captions=captions,
        alpha=0.5,
        colormap=colormap,
    )


def _visualize_objects(
    *, image: np.ndarray, annotations: list[types.Annotation]
) -> np.ndarray:
    # instances2rgb fills each mask with the color of its position, so the
    # colormap puts each object's own color there, the same on every frame.
    ids: list[int] = [typing.cast(int, a.object_id) for a in annotations]
    colors: NDArray[np.uint8] = imgviz.label_colormap()
    return _visualize(
        image=image,
        annotations=annotations,
        labels=list(range(1, 1 + len(ids))),
        captions=[str(i) for i in ids],
        colormap=colors[[0, *(1 + i % (len(colors) - 1) for i in ids)]],
    )


def _track(
    *,
    model_name: str,
    video_dir: str,
    output_dir: str | None,
    prompt: dict,
    device: types.Device | None,
    json: bool,
) -> None:
    # Frames in name order, numbers by value, so frame_2 comes before frame_10.
    paths: list[pathlib.Path] = sorted(
        (p for p in pathlib.Path(video_dir).iterdir() if p.is_file()),
        key=lambda p: [
            int(part) if part.isdigit() else part for part in re.split(r"(\d+)", p.name)
        ],
    )
    if not paths:
        raise click.UsageError(f"{video_dir} has no frames")
    if output_dir is not None and len({p.stem for p in paths}) < len(paths):
        raise click.UsageError(
            f"{video_dir} has frames of the same name, which -o would write to "
            "one file"
        )
    try:
        frames: list[np.ndarray] = [
            np.asarray(PIL.Image.open(path).convert("RGB")) for path in paths
        ]
    except PIL.UnidentifiedImageError as e:
        raise click.BadParameter(str(e), param_hint="--video")
    # --prompt takes Client.segment's keywords: those Client.track takes too,
    # such as include_masks, are options of the request, the rest the action's.
    track_keys: set[str] = set(inspect.signature(Client.track).parameters)
    options: dict = {k: v for k, v in prompt.items() if k in track_keys}
    action: dict = {k: v for k, v in prompt.items() if k not in track_keys}
    try:
        with _client(device=device) as client:
            response: types.TrackResponse = client.track(
                model_name,
                frames,
                [{"type": "prompt", "frame": 0} | action],
                **options,
            )
    except click.ClickException:
        raise
    except Exception:
        logger.exception("Failed to run model")
        sys.exit(1)

    if json:
        for frame in response.frames:
            click.echo(frame.model_dump_json())
        click.echo(response.done.model_dump_json())
    if output_dir is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    for path, frame in zip(paths, response.frames):
        visualization: np.ndarray = _visualize_objects(
            image=frames[frame.frame], annotations=frame.annotations
        )
        with open(os.path.join(output_dir, f"{path.stem}.png"), "wb") as f:
            f.write(_json.image_ndarray_to_png(ndarray=visualization))


if __name__ == "__main__":
    cli()
