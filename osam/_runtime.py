# Runs models: loads and unloads them, keeps recent image embeddings, and
# pulls and removes their files. osam serve runs one, and its routes call it;
# so does Client(in_process=True).
import builtins
import collections
import dataclasses
import hashlib
import threading
import time
import typing
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import Final

import numpy as np
from loguru import logger

from . import __version__
from . import _errors
from . import _json
from . import _onnx
from . import _registry
from . import _types as types
from ._requests import LoadRequest
from ._requests import SegmentRequest
from ._requests import SegmentResponse
from ._requests import TrackRequest

if typing.TYPE_CHECKING:
    from ._models.sam3._timeline import Timeline


@dataclasses.dataclass
class _Track:
    # The annotations on every frame after a list of actions, and its digest.
    frames: list[list[types.Annotation]]
    result: str

    @property
    def nbytes(self) -> int:
        # With about 128 bytes a frame for the lists and its reference in the
        # key, which holds the actions by digest, so long histories stay small,
        # and about 1200 bytes an annotation besides its mask's pixels, so many
        # small masks count too.
        return 128 * len(self.frames) + sum(
            1200 + (0 if a.mask is None else a.mask.nbytes)
            for f in self.frames
            for a in f
        )


@dataclasses.dataclass
class _LiveTimeline:
    # The video, by model, device and frames, and the actions applied to it.
    video: tuple[str, str, tuple[str, ...]]
    timeline: "Timeline"
    actions: list[str] = dataclasses.field(default_factory=list)
    results: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class _Loaded:
    model: types.Model
    # When it unloads, as time.time(); None keeps it loaded.
    expires_at: float | None = None
    timer: threading.Timer | None = None


# Runtime.list shadows list inside the class, hence builtins.list there.
class Runtime:
    # The embedding cache's budget. One SAM3 embedding takes about 100 MB and
    # one EfficientSAM embedding 4 MB; the newest one stays even when it alone
    # is larger.
    # ponytail: fixed budget, make it configurable if memory or hit rate matters.
    _embedding_cache_bytes: int = 1 << 30
    # The frames /api/track was sent, which it encodes again when their
    # embeddings were evicted: about 6 MB each at 1080p.
    _frame_cache_bytes: int = 2 << 30
    # Track results by video and actions, for undo and retries.
    _track_cache_bytes: int = 1 << 30

    # One call at a time, under _lock: loading and running models, and pull and
    # remove, which must not change the files under a model being loaded.
    # device: every loaded model runs on it; None picks one automatically.
    # max_loaded_models: when full, the least recently used model unloads first.
    # model_types: the models it runs, by name; tests pass fakes.

    _device: Final[types.Device | None]
    _max_loaded_models: Final[int]
    _model_types: Final[Sequence[type[types.Model]]]
    _lock: threading.Lock
    # By model name, least recently used first. Written under _lock and
    # _loaded_lock; ps takes only _loaded_lock, so it doesn't wait for a
    # running model.
    _loaded: collections.OrderedDict[str, _Loaded]
    _loaded_lock: threading.Lock
    # By model name and image reference, least recently used first; under _lock.
    _embeddings: collections.OrderedDict[tuple[str, str], types.ImageEmbedding]
    # By image reference, least recently used first; under _lock.
    _frames: collections.OrderedDict[str, np.ndarray]
    _tracks: collections.OrderedDict[
        tuple[tuple[str, str, tuple[str, ...]], str], _Track
    ]
    # ponytail: one timeline, so an edit after an undo replays every
    # action; keep one per prefix if those replays take too long.
    _timeline: _LiveTimeline | None
    # The frames of the request running, by index.
    _track_images: builtins.list[np.ndarray]

    def __init__(
        self,
        *,
        device: types.Device | None = None,
        max_loaded_models: int = 1,
        model_types: Sequence[type[types.Model]] = _registry.registered_model_types,
    ):
        if device not in (None, *typing.get_args(types.Device)):
            raise ValueError(f"unknown device {device!r}")
        if device is not None:
            # At start, not on the first load: a private server then fails to
            # start with one error, instead of answering 500.
            _onnx.device_providers(device=device)
        if max_loaded_models < 1:
            raise ValueError(
                f"max_loaded_models must be 1 or more: {max_loaded_models}"
            )
        self._device = device
        self._max_loaded_models = max_loaded_models
        self._model_types = model_types
        self._lock = threading.Lock()
        self._loaded = collections.OrderedDict()
        self._loaded_lock = threading.Lock()
        self._embeddings = collections.OrderedDict()
        self._frames = collections.OrderedDict()
        self._tracks = collections.OrderedDict()
        self._timeline = None
        self._track_images = []

    def version(self) -> str:
        return __version__

    def list(self) -> list[types.ModelInfo]:
        return [
            types.ModelInfo(
                name=model_type.name,
                id=model_type.get_id(device=self._device),
                size=model_type.get_size(device=self._device),
                modified_at=model_type.get_modified_at(device=self._device),
                metadata=model_type.metadata,
                capabilities=["segment", "track"]
                if model_type.tracks()
                else ["segment"],
                prompt_kinds=sorted(model_type.prompt_kinds),
            )
            for model_type in self._model_types
        ]

    def ps(self) -> builtins.list[types.RunningModel]:
        # Most recently used first, as Ollama lists them.
        with self._loaded_lock:
            loaded: tuple[_Loaded, ...] = tuple(self._loaded.values())
        return [
            types.RunningModel(
                name=entry.model.name,
                device=entry.model.device,
                expires_at=entry.expires_at,
            )
            for entry in reversed(loaded)
        ]

    def pull(
        self,
        *,
        model: str,
        progress: Callable[[dict[str, Any]], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> None:
        # progress gets a line for each chunk, as /api/pull streams them; cancel
        # stops the download.
        model_type: type[types.Model] = self.model_type(name=model)

        def report(filename: str, completed: int, total: int | None) -> None:
            line = {"status": f"pulling {filename}", "completed": completed}
            typing.cast(Callable, progress)(line | {"total": total})

        with self._lock:
            model_type.pull(
                device=self._device,
                progress=report if progress else None,
                cancel=cancel,
            )

    def remove(self, *, model: str) -> None:
        model_type: type[types.Model] = self.model_type(name=model)
        with self._lock:
            model_type.remove()

    def load(self, *, request: LoadRequest) -> None:
        # Loads the model, pulling it if needed, and keeps it loaded for
        # keep_alive seconds; negative keeps it loaded, 0 unloads it, as in Ollama.
        name: str = self.model_type(name=request.model).name
        with self._lock:
            if request.keep_alive != 0:
                self._load(model_name=name)
            self._keep_alive(model_name=name, seconds=request.keep_alive)

    def segment(self, *, request: SegmentRequest) -> SegmentResponse:
        # Raises UnknownImageError for an image reference it has no embedding for,
        # and ValueError for a prompt the model doesn't take.
        model_type: type[types.Model] = self.model_type(name=request.model)
        with self._lock:
            try:
                return self._segment(request=request, model_type=model_type)
            finally:
                self._keep_alive(model_name=model_type.name, seconds=request.keep_alive)

    def track(self, *, request: TrackRequest) -> types.TrackResponse:
        # Raises UnknownImageError for frame references it doesn't have,
        # _errors.BaseChangedError for an action on another result, and ValueError for
        # actions the model or the result doesn't take.
        model_type: type[types.Model] = self.model_type(name=request.model)
        if not model_type.tracks():
            raise ValueError(f"{model_type.name} doesn't track")
        with self._lock:
            try:
                return self._track(request=request, model_type=model_type)
            finally:
                self._keep_alive(model_name=model_type.name, seconds=request.keep_alive)

    def close(self) -> None:
        # Unloads every model and stops their timers, which hold this runtime.
        with self._lock:
            while self._loaded:
                self._unload(model_name=next(iter(self._loaded)))

    def model_type(self, *, name: str) -> type[types.Model]:
        # Raises UnknownModelError for a model it doesn't run.
        return _registry.get_model_type_by_name(
            name=name, model_types=self._model_types
        )

    def _segment(
        self, *, request: SegmentRequest, model_type: type[types.Model]
    ) -> SegmentResponse:
        # Recent image embeddings stay cached, by model name and image reference, so
        # several prompts on one image encode it once.
        image_ref: str = (
            request.image
            if isinstance(request.image, str)
            else _json.image_ref(image=request.image)
        )
        key: tuple[str, str] = (model_type.name, image_ref)
        embedding: types.ImageEmbedding | None = self._embeddings.get(key)
        if isinstance(request.image, str) and embedding is None:
            # Before loading the model, which a retry with the image would
            # otherwise load a second time after keep_alive 0 unloads it.
            raise _errors.UnknownImageError(refs=[image_ref])
        model: types.Model = self._load(model_name=model_type.name)
        if embedding is None:
            assert not isinstance(request.image, str)  # raised UnknownImageError
            embedding = model.encode_image(image=request.image)
        self._cache_embedding(key=key, embedding=embedding)
        annotations: list[types.Annotation] = model.decode(
            embedding=embedding, prompt=request
        )
        if not request.include_masks:
            annotations = [a.model_copy(update={"mask": None}) for a in annotations]
        return SegmentResponse(annotations=annotations, image=image_ref)

    def _cache_embedding(
        self, *, key: tuple[str, str], embedding: types.ImageEmbedding
    ) -> None:
        self._embeddings[key] = embedding
        self._embeddings.move_to_end(key)
        _evict(cache=self._embeddings, max_bytes=self._embedding_cache_bytes)

    def _track(
        self, *, request: TrackRequest, model_type: type[types.Model]
    ) -> types.TrackResponse:
        refs: builtins.list[str] = []
        images: builtins.list[np.ndarray] = []
        missing: builtins.list[str] = []
        for frame in request.frames:
            if isinstance(frame, str):
                refs.append(frame)
                if frame in self._frames:
                    images.append(self._frames[frame])
                else:
                    missing.append(frame)
            else:
                ref: str = _json.image_ref(image=frame)
                refs.append(ref)
                # A copy, so that a caller in this process can't change the
                # frame its reference names.
                images.append(
                    self._frames[ref] if ref in self._frames else frame.copy()
                )
        if missing:
            raise _errors.UnknownImageError(refs=missing)
        for ref, image in zip(refs, images):
            self._frames[ref] = image
            self._frames.move_to_end(ref)
        _evict(cache=self._frames, max_bytes=self._frame_cache_bytes)
        if len({image.shape for image in images}) > 1:
            raise ValueError("frames must all have the same size")
        num_frames: int = len(images)
        last_frame: int = (
            num_frames - 1 if request.last_frame is None else request.last_frame
        )
        if not request.first_frame <= last_frame < num_frames:
            raise ValueError(
                f"first_frame and last_frame must be in order and under {num_frames}"
            )

        model: types.Model = self._load(model_name=model_type.name)
        video: tuple[str, str, tuple[str, ...]] = (
            model_type.name,
            model.device,
            tuple(refs),
        )
        actions: tuple[str, ...] = tuple(a.model_dump_json() for a in request.actions)
        key: tuple[tuple[str, str, tuple[str, ...]], str] = (
            video,
            hashlib.sha256("\n".join(actions).encode()).hexdigest(),
        )
        track: _Track | None = self._tracks.get(key)
        cache: typing.Literal["hit", "partial", "miss"] = "hit"
        if track is None:
            # For the live timeline to encode frames again, during this request
            # only, so it keeps no frames between requests.
            self._track_images = images
            try:
                track, cache = self._apply(
                    request=request,
                    model=model,
                    video=video,
                    actions=actions,
                    num_frames=num_frames,
                    height=images[0].shape[0],
                    width=images[0].shape[1],
                )
            finally:
                self._track_images = []
            self._tracks[key] = track
        self._tracks.move_to_end(key)
        _evict(cache=self._tracks, max_bytes=self._track_cache_bytes)

        objects: dict[int, types.TrackObject] = {}
        for i, annotations in enumerate(track.frames):
            for a in annotations:
                assert a.object_id is not None
                objects.setdefault(
                    a.object_id,
                    types.TrackObject(
                        object_id=a.object_id, first_frame=i, last_frame=i
                    ),
                ).last_frame = i
        return types.TrackResponse(
            frames=[
                types.TrackFrame(
                    frame=i,
                    # Copies, so that a caller in this process can't change
                    # the cached results.
                    annotations=[
                        a.model_copy(
                            update={} if request.include_masks else {"mask": None},
                            deep=True,
                        )
                        for a in track.frames[i]
                    ],
                )
                for i in range(request.first_frame, last_frame + 1)
            ],
            done=types.TrackDone(
                objects=sorted(objects.values(), key=lambda o: o.object_id),
                result=track.result,
                cache=cache,
                images=refs,
            ),
        )

    def _apply(
        self,
        *,
        request: TrackRequest,
        model: types.Model,
        video: tuple[str, str, tuple[str, ...]],
        actions: tuple[str, ...],
        num_frames: int,
        height: int,
        width: int,
    ) -> tuple[_Track, typing.Literal["hit", "partial", "miss"]]:
        # Applies the actions the live timeline hasn't, or replays them all
        # on a new one, checking each action's base before applying it.
        live: _LiveTimeline | None = self._timeline
        if (
            live is None
            or live.video != video
            or tuple(live.actions) != actions[: len(live.actions)]
        ):

            def get_embedding(frame: int) -> types.ImageEmbedding:
                key: tuple[str, str] = (video[0], video[2][frame])
                embedding: types.ImageEmbedding | None = self._embeddings.get(key)
                if embedding is None:
                    embedding = model.encode_image(image=self._track_images[frame])
                self._cache_embedding(key=key, embedding=embedding)
                return embedding

            self._timeline = None  # free the old one first
            live = _LiveTimeline(
                video=video,
                timeline=model.new_timeline(
                    get_embedding=get_embedding,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                ),
            )
        self._timeline = live
        applied: int = len(live.actions)
        try:
            for i in range(applied, len(actions)):
                action: types.Action = request.actions[i]
                if not isinstance(action, types.PromptAction) and (
                    action.base != live.results[i - 1]
                ):
                    raise _errors.BaseChangedError(index=i, result=live.results[i - 1])
                live.timeline.apply(action=action)
                live.actions.append(actions[i])
                live.results.append(live.timeline.result())
        except _errors.BaseChangedError:
            raise
        except Exception:
            # An action may have half applied.
            self._timeline = None
            raise
        track = _Track(
            frames=[live.timeline.annotations(frame=i) for i in range(num_frames)],
            result=live.results[-1],
        )
        cache: typing.Literal["hit", "partial", "miss"] = (
            "hit" if applied == len(actions) else "partial" if applied else "miss"
        )
        return track, cache

    def _load(self, *, model_name: str) -> types.Model:
        # Under _lock. When full, the least recently used model unloads first;
        # one call runs at a time, so that model is idle.
        if model_name in self._loaded:
            with self._loaded_lock:
                self._loaded.move_to_end(model_name)
            return self._loaded[model_name].model
        model_type: type[types.Model] = self.model_type(name=model_name)
        # Before unloading anything, so a failed download leaves the others
        # loaded. Files already on disk are verified once, when the model loads.
        if model_type.get_size(device=self._device) is None:
            model_type.pull(device=self._device)
        while len(self._loaded) >= self._max_loaded_models:
            self._unload(model_name=next(iter(self._loaded)))
        model: types.Model | None = None
        try:
            model = model_type(device=self._device)
        except Exception:
            if not self._loaded:
                raise
            logger.exception(
                "Failed to load {}; unloading the other models", model_name
            )
        if (
            model is not None
            and self._device is None
            and model.device == "cpu"
            and any(entry.model.device != "cpu" for entry in self._loaded.values())
        ):
            # An automatic device falls back to CPU when the GPU, which the others
            # use, is out of memory.
            logger.warning(
                "{} fell back to CPU; unloading the other models", model_name
            )
            model = None
        if model is None:
            # Likely out of memory, so unload the others and try once more, as
            # Ollama does. Outside the except block, whose traceback would keep
            # the failed load's sessions, and their memory, alive.
            while self._loaded:
                self._unload(model_name=next(iter(self._loaded)))
            model = model_type(device=self._device)
        with self._loaded_lock:
            self._loaded[model_name] = _Loaded(model=model)
        return model

    def _keep_alive(self, *, model_name: str, seconds: float) -> None:
        # Under _lock. A timer unloads the model, unless a later call replaced it.
        loaded: _Loaded | None = self._loaded.get(model_name)
        if loaded is None:  # it failed to load, or it isn't loaded
            return
        if loaded.timer is not None:
            loaded.timer.cancel()
            loaded.timer = None
        if seconds < 0:
            loaded.expires_at = None
            return
        if seconds == 0:
            self._unload(model_name=model_name)
            return
        loaded.expires_at = time.time() + seconds
        timer = threading.Timer(
            seconds,
            lambda: self._unload_if_current(model_name=model_name, timer=timer),
        )
        timer.daemon = True
        timer.start()
        loaded.timer = timer

    def _unload_if_current(self, *, model_name: str, timer: threading.Timer) -> None:
        # No local reference to the model, so it is freed, and its memory with
        # it, before the lock lets another load start.
        with self._lock:
            if getattr(self._loaded.get(model_name), "timer", None) is timer:
                self._unload(model_name=model_name)

    def _unload(self, *, model_name: str) -> None:
        with self._loaded_lock:
            loaded: _Loaded = self._loaded.pop(model_name)
        if loaded.timer is not None:
            loaded.timer.cancel()
        for key in [key for key in self._embeddings if key[0] == model_name]:
            del self._embeddings[key]
        if self._timeline is not None and self._timeline.video[0] == model_name:
            self._timeline = None


def _evict(*, cache: collections.OrderedDict, max_bytes: int) -> None:
    # Least recently used first, keeping the newest even when it alone is larger.
    while len(cache) > 1 and sum(v.nbytes for v in cache.values()) > max_bytes:
        cache.popitem(last=False)
