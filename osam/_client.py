import builtins
import threading
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from . import _http
from . import _requests
from . import _runtime
from . import types
from ._errors import ResponseError

__all__ = ["Client", "ResponseError"]


class _Backend(Protocol):
    # Where Client's calls run, given as request models.

    def close(self) -> None: ...

    def version(self) -> str: ...

    def list(self) -> list[types.ModelInfo]: ...

    def ps(self) -> builtins.list[types.RunningModel]: ...

    def pull(
        self,
        *,
        model: str,
        progress: Callable[[dict[str, Any]], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> None: ...

    def remove(self, *, model: str) -> None: ...

    def load(self, *, request: _requests.LoadRequest) -> types.RunningModel | None: ...

    def segment(
        self, *, request: _requests.SegmentRequest
    ) -> _requests.SegmentResponse: ...

    def track(
        self,
        *,
        request: _requests.TrackRequest,
        progress: Callable[[types.TrackProgress], None] | None = None,
    ) -> types.TrackResponse: ...


# Client.list shadows list inside the class, hence builtins.list there.
class Client:
    """Run models on an osam server.

    With no host, it starts a private server for this process, which exits
    with it; a frozen application must call multiprocessing.freeze_support()
    first. host, or OSAM_HOST when host is None and no private-server
    setting below is given, connects to a running `osam serve` instead,
    such as "127.0.0.1:11368". in_process=True runs
    models in this process, with no server; it takes no host or timeout.

    device, max_loaded_models: as in `osam serve --device` and
    `--max-loaded-models`, for the private server or this process.
    auto_pull: False makes load, and calls that load, such as segment, raise
    osam.types.NotPulledError for a model whose files aren't on disk, such as
    the CPU's after an automatic device falls back to it, instead of pulling
    them; a server reports it as a ResponseError with code
    "model_not_pulled".
    timeout: seconds to wait for each response, such as 600, or None to wait
    forever. A request that times out raises TimeoutError, and a private
    server restarts, so a hung model stops.
    A model name osam doesn't know raises LookupError; from a server, it is
    also a ResponseError with status_code 404.
    """

    _backend: _Backend

    def __init__(
        self,
        host: str | None = None,
        *,
        device: types.Device | None = None,
        max_loaded_models: int | None = None,
        timeout: float | None = None,
        in_process: bool = False,
        auto_pull: bool = True,
    ):
        if not in_process:
            self._backend = _http.HttpBackend(
                host,
                device=device,
                max_loaded_models=max_loaded_models,
                timeout=timeout,
                auto_pull=auto_pull,
            )
            return
        if (host, timeout) != (None, None):
            raise ValueError(
                "in_process runs no server, so it takes no host or timeout"
            )
        self._backend = _runtime.Runtime(
            device=device,
            max_loaded_models=1 if max_loaded_models is None else max_loaded_models,
            auto_pull=auto_pull,
        )

    def close(self) -> None:
        """Stop the private server, or release in-process models and caches."""
        self._backend.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def version(self) -> str:
        return self._backend.version()

    def list(self) -> list[types.ModelInfo]:
        """List every model osam knows, pulled or not."""
        return self._backend.list()

    def ps(self) -> builtins.list[types.RunningModel]:
        """List the loaded models."""
        return self._backend.ps()

    def pull(
        self,
        model: str,
        *,
        progress: Callable[[dict[str, Any]], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> None:
        """Download a model, calling progress with each line the server sends.

        Lines look like {"status": "pulling encoder.onnx", "completed": 524288,
        "total": 24799761}. An exception raised from progress stops the
        download on the server too.
        Setting cancel, such as from another thread, stops the download even
        while no data arrives, deletes its partial files, and raises
        osam.types.PullCancelledError.
        A download that fails raises RuntimeError.
        """
        self._backend.pull(model=model, progress=progress, cancel=cancel)

    def remove(self, model: str) -> None:
        self._backend.remove(model=model)

    def load(
        self, model: str, *, keep_alive: float | str | None = None
    ) -> types.RunningModel | None:
        """Load model, pulling it if needed with auto_pull, and keep it loaded
        for keep_alive. Returns it as ps lists it, with the device it runs on,
        or None once keep_alive 0 unloads it.

        keep_alive: in seconds or as "5m"; -1 keeps it loaded, 0 unloads it
        now. None leaves the server's default of five minutes.
        """
        return self._backend.load(
            request=_requests.LoadRequest.model_validate(
                _requests.to_json_fields(
                    fields={"model": model, "keep_alive": keep_alive}
                )
            )
        )

    def segment(
        self,
        model: str,
        image: np.ndarray,
        *,
        points: ArrayLike | None = None,
        point_labels: ArrayLike | None = None,
        box: ArrayLike | None = None,
        texts: builtins.list[str] | None = None,
        example_boxes: ArrayLike | None = None,
        example_box_labels: ArrayLike | None = None,
        iou_threshold: float | None = None,
        score_threshold: float | None = None,
        max_annotations: int | None = None,
        include_masks: bool | None = None,
        keep_alive: float | str | None = None,
    ) -> builtins.list[types.Annotation]:
        """Segment image, an HxWx3 uint8 RGB array, with model; uint8 grayscale
        (HxW) and RGBA (HxWx4) arrays are converted to RGB.

        Give one object, or ask for every match:

        - points (N, 2) as x, y in pixels, with point_labels (N,), 1 for
          foreground and 0 for background, and at most one box as xmin, ymin,
          xmax, ymax, segment one object.
        - texts, example_boxes (M, 4), or both find every match.
          example_box_labels (M,) is 1 for a positive example and 0 for a
          negative one, all 1 if unset.

        A prompt the model doesn't take raises ValueError. With no prompt, the
        model uses its default one. The thresholds apply to models that find
        every match. Prompts on an image the server saw recently reuse its
        embedding.
        include_masks: False leaves out each annotation's mask, for boxes and
        scores only.
        keep_alive: as in load.
        """
        prompt: dict[str, Any] = {
            "points": points,
            "point_labels": point_labels,
            "box": box,
            "texts": texts,
            "example_boxes": example_boxes,
            "example_box_labels": example_box_labels,
            "iou_threshold": iou_threshold,
            "score_threshold": score_threshold,
            "max_annotations": max_annotations,
            "include_masks": include_masks,
            "keep_alive": keep_alive,
        }
        request = _requests.SegmentRequest.model_validate(
            {"model": model, "image": image} | _requests.to_json_fields(fields=prompt)
        )
        return self._backend.segment(request=request).annotations

    def track(
        self,
        model: str,
        frames: Sequence[np.ndarray],
        actions: builtins.list[dict[str, Any]],
        *,
        first_frame: int | None = None,
        last_frame: int | None = None,
        include_masks: bool | None = None,
        keep_alive: float | str | None = None,
        progress: Callable[[types.TrackProgress], None] | None = None,
    ) -> types.TrackResponse:
        """Track objects through frames, RGB arrays of one size, with model.

        actions: every edit so far, in order, with coordinates in pixels:

        - {"type": "prompt", "frame", ...} with the prompt of segment, first
          and only first: points or a box track one object, texts every match.
        - {"type": "add", "frame", "points", "point_labels", "box", "base"}
        - {"type": "refine", "frame", "object_id", "points", "point_labels",
          "base"}
        - {"type": "remove", "object_id", "base"}

        base is the done.result of the response the edit was made on; another
        result raises ResponseError with code "base_changed". The response has
        the annotations of first_frame to last_frame, each with its object_id,
        and done with each object's frames, result, and whether it hit the
        server's cache. Frames the server saw before go by reference.
        include_masks, keep_alive: as in segment.
        progress receives the current action and frame, starting with a
        preparing event. Positions can go backwards during propagation.
        Raising from the callback cancels between frames, after any native
        inference call in flight finishes.
        """
        request = _requests.TrackRequest.model_validate(
            {"model": model, "frames": builtins.list(frames), "actions": actions}
            | _requests.to_json_fields(
                fields={
                    "first_frame": first_frame,
                    "last_frame": last_frame,
                    "include_masks": include_masks,
                    "keep_alive": keep_alive,
                }
            )
        )
        return self._backend.track(request=request, progress=progress)
