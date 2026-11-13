from __future__ import annotations

import abc
import hashlib
import threading
import typing
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import onnxruntime
from loguru import logger

from .. import _onnx
from .._errors import NotPulledError
from ..types import Annotation
from ..types import Blob
from ..types import BoundingBox
from ..types import Device
from ..types import ModelMetadata
from ._image_embedding import ImageEmbedding
from ._prompt import THRESHOLDS
from ._prompt import Prompt
from ._prompt import PromptKind

if typing.TYPE_CHECKING:
    from .._models.sam3._timeline import Timeline


class Model(abc.ABC):
    name: str
    metadata: ModelMetadata

    # The prompt kinds decode takes; others raise ValueError.
    prompt_kinds: frozenset[PromptKind]

    # The file of each session, by session name, which remove deletes; a model
    # with files for some devices lists them all, and _device_blobs picks.
    _blobs: dict[str, Blob]
    # Sessions that run on CoreML with device="coreml", all when None; the
    # others run on WebGPU when it finds a GPU, and on the CPU otherwise.
    _coreml_blobs: frozenset[str] | None = None
    _inference_sessions: dict[str, onnxruntime.InferenceSession]

    def __init__(self, device: Device | None = None, *, auto_pull: bool = True):
        """Load the model.

        device: "cpu", "cuda", "coreml", or "webgpu". None picks OSAM_ONNX_PROVIDERS,
        or the first available of CUDA and a hardware WebGPU adapter, falling
        back to CPU; CoreML before WebGPU for a model that runs faster there.
        auto_pull: False raises osam.types.NotPulledError for files not on
        disk, such as the CPU's after a fallback, instead of pulling them.
        """
        logger.debug("Initializing model {name}", name=self.name)
        if device is None and self._picks_coreml():
            try:
                self.__init__(device="coreml", auto_pull=auto_pull)
                return
            except Exception as e:
                logger.warning(
                    "Failed to load {name} on CoreML, picking another device: {e}",
                    name=self.name,
                    e=e,
                )
        if device is None and self._picks_cpu():
            device = "cpu"
        if device is not None:
            _onnx.device_providers(device=device)  # fail before downloading
            if not self._runs_on(device=device):
                raise RuntimeError(
                    f"{self.name} fails on {device} with onnxruntime "
                    f"{onnxruntime.__version__}"
                )
        # Without a device, past CoreML, the files of the one picked otherwise.
        files_device: str = device or _onnx.default_device()
        self._pull(device=files_device, auto_pull=auto_pull, cancel=None)
        blobs: dict[str, Blob] = self._device_blobs(device=files_device)
        if device is None:
            self._inference_sessions = _onnx.load_inference_sessions(blobs=blobs)
            if self.device != "cpu" or blobs == self._device_blobs(device="cpu"):
                return
            # A fallback to the CPU, which runs its own files faster, such as
            # float32 instead of a GPU's float16.
            del self._inference_sessions  # free them before loading the others
            device = "cpu"
            self._pull(device=device, auto_pull=auto_pull, cancel=None)
            blobs = self._device_blobs(device=device)
        self._inference_sessions = {
            key: _onnx.create_device_session(
                path=blob.path,
                providers=_onnx.device_providers(
                    device=device,
                    coreml=self._coreml_blobs is None or key in self._coreml_blobs,
                    coreml_cache_path=blob.coreml_cache_path,
                ),
            )
            for key, blob in blobs.items()
        }
        logger.info(
            "Initialized inference sessions on {device!r}: {providers!r}",
            device=device,
            providers={
                k: s.get_providers()[0] for k, s in self._inference_sessions.items()
            },
        )

    @classmethod
    def _pull(
        cls,
        *,
        device: str | None,
        auto_pull: bool,
        cancel: threading.Event | None,
    ) -> None:
        # Pulls the files of device, or with auto_pull off, raises for missing or
        # damaged ones instead of downloading them again.
        if auto_pull:
            cls.pull(device=device, cancel=cancel)
        elif not all(blob.verified() for blob in cls._blobs_of(device=device).values()):
            raise NotPulledError(model=cls.name, device=device)

    @property
    def device(self) -> str:
        # The device the model runs on, e.g. "webgpu", or "cpu" after a fallback.
        session = next(iter(self._inference_sessions.values()))
        provider: str = session.get_providers()[0]
        return _onnx.PROVIDER_DEVICES.get(provider, provider)

    @classmethod
    def _device_blobs(cls, *, device: str) -> dict[str, Blob]:
        # The file of each session on device, for a model with files that run
        # faster on some devices, such as float16 copies for GPUs.
        return cls._blobs

    @classmethod
    def _runs_on(cls, *, device: str) -> bool:
        # Whether the installed onnxruntime runs the model right on device;
        # picking a device automatically picks the CPU instead of one it doesn't.
        return True

    @classmethod
    def _runs_faster_on_coreml(cls) -> bool:
        # Than on WebGPU, so that picking a device automatically picks CoreML.
        return False

    @classmethod
    def _picks_coreml(cls) -> bool:
        # Whether picking a device automatically tries CoreML first.
        return cls._runs_faster_on_coreml() and _onnx.picks_coreml()

    @classmethod
    def _picks_cpu(cls) -> bool:
        # Whether picking a device automatically picks the CPU, as the model
        # fails on the device picked otherwise.
        return not cls._runs_on(device=_onnx.default_device())

    @classmethod
    def _blobs_of(cls, *, device: str | None) -> dict[str, Blob]:
        # The files a model loads on device, or with the device it picks itself.
        if device is not None:
            return cls._device_blobs(device=device)
        blobs: dict[str, Blob] = cls._device_blobs(
            device="cpu" if cls._picks_cpu() else _onnx.default_device()
        )
        if not cls._picks_coreml():
            return blobs
        # CoreML's, unless only the others are on disk, such as offline after
        # an upgrade; loading falls back to them when it can't pull CoreML's.
        coreml: dict[str, Blob] = cls._device_blobs(device="coreml")
        if any(b.size is None for b in coreml.values()) and all(
            b.size is not None for b in blobs.values()
        ):
            return blobs
        return coreml

    @classmethod
    def pull(
        cls,
        *,
        device: str | None = None,
        progress: Callable[[str, int, int | None], None] | None = None,
        cancel: threading.Event | None = None,
        timeout: float | tuple[float, float] | None = 30,
    ) -> None:
        # The files loading tries first, even with others on disk.
        if device is None and cls._picks_coreml():
            device = "coreml"
        for blob in cls._blobs_of(device=device).values():
            blob.pull(progress=progress, cancel=cancel, timeout=timeout)

    @classmethod
    def remove(cls):
        for blob in cls._blobs.values():
            blob.remove()

    @classmethod
    def get_id(cls, *, device: str | None = None) -> str:
        return hashlib.md5(
            "+".join(
                blob.hash for blob in cls._blobs_of(device=device).values()
            ).encode()
        ).hexdigest()[:12]

    @classmethod
    def get_size(cls, *, device: str | None = None) -> int | None:
        size = 0
        for blob in cls._blobs_of(device=device).values():
            # Read once: a concurrent remove can turn a second read into None.
            blob_size: int | None = blob.size
            if blob_size is None:
                return None
            size += blob_size
        return size

    @classmethod
    def get_download_size(cls, *, device: str | None = None) -> int | None:
        size = 0
        for top in cls._blobs_of(device=device).values():
            for blob, _ in top._files:  # with its attachments
                if blob.download_size is None:
                    return None
                size += blob.download_size
        return size

    @classmethod
    def get_modified_at(cls, *, device: str | None = None) -> float | None:
        modified_at: float = 0
        for blob in cls._blobs_of(device=device).values():
            blob_modified_at: float | None = blob.modified_at
            if blob_modified_at is None:
                return None
            modified_at = max(modified_at, blob_modified_at)
        return modified_at

    @abc.abstractmethod
    def encode_image(self, *, image: np.ndarray) -> ImageEmbedding:
        # image: an RGB (H, W, 3) uint8 array.
        pass

    @classmethod
    @abc.abstractmethod
    def default_prompt(cls, *, height: int, width: int) -> Prompt:
        # The prompt used when a request gives none.
        pass

    def decode(self, *, embedding: ImageEmbedding, prompt: Prompt) -> list[Annotation]:
        if not prompt.kinds:
            prompt = self.default_prompt(
                height=embedding.original_height, width=embedding.original_width
            ).model_copy(
                update=prompt.model_dump(include=set(THRESHOLDS), exclude_none=True)
            )
            logger.debug("Using the default prompt: {prompt!r}", prompt=prompt)
        if unsupported := prompt.kinds - self.prompt_kinds:
            raise ValueError(
                f"{self.name} takes {', '.join(sorted(self.prompt_kinds))}, "
                f"not {', '.join(sorted(unsupported))}"
            )
        return self._decode(embedding=embedding, prompt=prompt)

    @abc.abstractmethod
    def _decode(self, *, embedding: ImageEmbedding, prompt: Prompt) -> list[Annotation]:
        pass

    @classmethod
    def tracks(cls) -> bool:
        # Whether new_timeline tracks objects through video frames.
        return False

    def new_timeline(
        self,
        *,
        get_embedding: Callable[[int], ImageEmbedding],
        num_frames: int,
        height: int,
        width: int,
    ) -> Timeline:
        # A video of num_frames frames, whose embeddings get_embedding returns
        # by frame index, for /api/track's actions.
        raise NotImplementedError(f"{self.name} doesn't track")


def annotations_from_mask(
    *, mask: npt.NDArray[np.bool_], score: float | None = None
) -> list[Annotation]:
    # The mask cropped to its bounding box, or none for an empty mask.
    # From the rows and columns with a pixel: 0.2 ms at 2560x1600, where
    # imgviz.masks_to_bboxes took 10 ms.
    ys = np.flatnonzero(mask.any(axis=1))
    if ys.size == 0:
        return []
    xs = np.flatnonzero(mask.any(axis=0))
    ymin, ymax, xmin, xmax = int(ys[0]), int(ys[-1]), int(xs[0]), int(xs[-1])
    return [
        Annotation(
            # A copy, so a cached annotation doesn't keep the whole mask.
            mask=mask[ymin : ymax + 1, xmin : xmax + 1].copy(),
            bounding_box=BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax),
            score=score,
        )
    ]
