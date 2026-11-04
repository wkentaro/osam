import builtins
from collections.abc import Callable
from typing import Any
from typing import Optional
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from . import _http
from . import _requests
from . import types
from ._http import ResponseError

__all__ = ["Client", "ResponseError"]


class _Backend(Protocol):
    # Where Client's calls run, given as request models.

    def close(self) -> None:
        ...

    def version(self) -> str:
        ...

    def list(self) -> list[types.ModelInfo]:
        ...

    def ps(self) -> builtins.list[types.RunningModel]:
        ...

    def pull(
        self,
        *,
        model: str,
        progress: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        ...

    def remove(self, *, model: str) -> None:
        ...

    def load(self, *, request: _requests.LoadRequest) -> None:
        ...

    def segment(
        self, *, request: _requests.SegmentRequest
    ) -> _requests.SegmentResponse:
        ...


# Client.list shadows list inside the class, hence builtins.list there.
class Client:
    """Run models on an osam server.

    With no host, it starts a private server for this process, which exits
    with it; a frozen application must call multiprocessing.freeze_support()
    first. host, or OSAM_HOST when host is None, connects to a running
    `osam serve` instead, such as "127.0.0.1:11368".

    device, max_loaded_models: as in `osam serve --device` and
    `--max-loaded-models`, for the private server only.
    timeout: seconds to wait for each response, such as 600, or None to wait
    forever. A request that times out raises TimeoutError, and a private
    server restarts, so a hung model stops.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        *,
        device: Optional[types.Device] = None,
        max_loaded_models: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        self._backend: _Backend = _http.HttpBackend(
            host, device=device, max_loaded_models=max_loaded_models, timeout=timeout
        )

    def close(self) -> None:
        """Stop the private server, if this client started one."""
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
        progress: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        """Download a model, calling progress with each line the server sends.

        Lines look like {"status": "pulling encoder.onnx", "completed": 524288,
        "total": 24799761}. An exception raised from progress, such as on a
        cancel, stops the download on the server too.
        """
        self._backend.pull(model=model, progress=progress)

    def remove(self, model: str) -> None:
        self._backend.remove(model=model)

    def load(self, model: str, *, keep_alive: Optional[float | str] = None) -> None:
        """Load model, pulling it if needed, and keep it loaded for keep_alive.

        keep_alive: in seconds or as "5m"; -1 keeps it loaded, 0 unloads it
        now. None leaves the server's default of five minutes.
        """
        self._backend.load(
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
        points: Optional[ArrayLike] = None,
        point_labels: Optional[ArrayLike] = None,
        box: Optional[ArrayLike] = None,
        texts: Optional[builtins.list[str]] = None,
        example_boxes: Optional[ArrayLike] = None,
        example_box_labels: Optional[ArrayLike] = None,
        iou_threshold: Optional[float] = None,
        score_threshold: Optional[float] = None,
        max_annotations: Optional[int] = None,
        keep_alive: Optional[float | str] = None,
    ) -> builtins.list[types.Annotation]:
        """Segment image, an HxWx3 uint8 array, with model.

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
            "keep_alive": keep_alive,
        }
        request = _requests.SegmentRequest.model_validate(
            {"model": model, "image": image} | _requests.to_json_fields(fields=prompt)
        )
        return self._backend.segment(request=request).annotations
