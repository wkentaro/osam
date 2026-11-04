# Runs models: loads and unloads them, keeps recent image embeddings, and
# pulls and removes their files. osam serve runs one; its routes call it.
import builtins
import collections
import dataclasses
import threading
import time
import typing
from collections.abc import Callable
from typing import Final
from typing import Optional

import numpy as np
from loguru import logger

from . import _json
from . import _registry
from . import _types as types
from ._requests import SegmentRequest
from ._requests import SegmentResponse


class UnknownImageError(LookupError):
    # An image_hash whose embedding the runtime doesn't have.
    pass


@dataclasses.dataclass
class _Loaded:
    model: types.Model
    # When it unloads, as time.time(); None keeps it loaded.
    expires_at: Optional[float] = None
    timer: Optional[threading.Timer] = None


# Runtime.list shadows list inside the class, hence builtins.list there.
class Runtime:
    # One call at a time, under _lock: loading and running models, and pull and
    # remove, which must not change the files under a model being loaded.
    # device: every loaded model runs on it; None picks one automatically.
    # max_loaded_models: when full, the least recently used model unloads first.

    def __init__(
        self, *, device: Optional[types.Device] = None, max_loaded_models: int = 1
    ):
        if device not in (None, *typing.get_args(types.Device)):
            raise ValueError(f"unknown device {device!r}")
        if max_loaded_models < 1:
            raise ValueError(
                f"max_loaded_models must be 1 or more: {max_loaded_models}"
            )
        self._device: Final = device
        self._max_loaded_models: Final = max_loaded_models
        self._lock = threading.Lock()
        # By model name, least recently used first. Written under _lock and
        # _loaded_lock; ps takes only _loaded_lock, so it doesn't wait for a
        # running model.
        self._loaded: collections.OrderedDict[str, _Loaded] = collections.OrderedDict()
        self._loaded_lock = threading.Lock()
        # By model name and image hash, least recently used first; under _lock.
        self._embeddings: collections.OrderedDict[
            tuple[str, str], types.ImageEmbedding
        ] = collections.OrderedDict()

    def list(self) -> list[types.ModelInfo]:
        return [
            types.ModelInfo(
                name=model_type.name,
                id=model_type.get_id(),
                size=model_type.get_size(),
                modified_at=model_type.get_modified_at(),
                metadata=model_type.metadata,
            )
            for model_type in _registry.registered_model_types
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
        progress: Optional[Callable[[str, int, Optional[int]], None]] = None,
        cancel: Optional[threading.Event] = None,
    ) -> None:
        # progress(filename, completed, total) for each chunk; cancel stops it.
        model_type: type[types.Model] = _registry.get_model_type_by_name(name=model)
        with self._lock:
            model_type.pull(progress=progress, cancel=cancel)

    def remove(self, *, model: str) -> None:
        model_type: type[types.Model] = _registry.get_model_type_by_name(name=model)
        with self._lock:
            model_type.remove()

    def load(self, *, model: str, keep_alive: float) -> None:
        # Loads model, pulling it if needed, and keeps it loaded for keep_alive
        # seconds; negative keeps it loaded, 0 unloads it, as in Ollama.
        name: str = _registry.get_model_type_by_name(name=model).name
        with self._lock:
            if keep_alive != 0:
                self._load(model_name=name)
            self._keep_alive(model_name=name, seconds=keep_alive)

    def segment(self, *, request: SegmentRequest) -> SegmentResponse:
        # Raises UnknownImageError for an image_hash it has no embedding for,
        # and ValueError for a prompt the model doesn't take.
        model_type: type[types.Model] = _registry.get_model_type_by_name(
            name=request.model
        )
        with self._lock:
            try:
                return self._segment(request=request, model_type=model_type)
            finally:
                self._keep_alive(model_name=model_type.name, seconds=request.keep_alive)

    def generate(
        self,
        *,
        model: str,
        image: np.ndarray,
        prompt: Optional[types.Prompt],
        keep_alive: float,
    ) -> types.GenerateResponse:
        # For osam 0.10's /api/generate: prompt in the models' point labels.
        name: str = _registry.get_model_type_by_name(name=model).name
        with self._lock:
            try:
                return self._generate(
                    key=(name, _json.hash_image(image=image)),
                    image=image,
                    prompt=prompt,
                )
            finally:
                self._keep_alive(model_name=name, seconds=keep_alive)

    def close(self) -> None:
        # Unloads every model and stops their timers, which hold this runtime.
        with self._lock:
            while self._loaded:
                self._unload(model_name=next(iter(self._loaded)))

    def _segment(
        self, *, request: SegmentRequest, model_type: type[types.Model]
    ) -> SegmentResponse:
        image_hash: str = (
            typing.cast(str, request.image_hash)
            if request.image is None
            else _json.hash_image(image=request.image)
        )
        key: tuple[str, str] = (model_type.name, image_hash)
        embedding: Optional[types.ImageEmbedding] = self._embeddings.get(key)
        if request.image is None:
            if embedding is None:
                # Before loading the model, which a retry with the image would
                # otherwise load a second time after keep_alive 0 unloads it.
                raise UnknownImageError(
                    f"no embedding for image {image_hash}; send the image"
                )
            height, width = embedding.original_height, embedding.original_width
        else:
            height, width = request.image.shape[:2]
        prompt: Optional[types.Prompt] = request.to_prompt(
            model_type=model_type, height=height, width=width
        )
        response: types.GenerateResponse = self._generate(
            key=key, image=request.image, prompt=prompt
        )
        return SegmentResponse(annotations=response.annotations, image_hash=image_hash)

    def _generate(
        self,
        *,
        key: tuple[str, str],
        image: Optional[np.ndarray],
        prompt: Optional[types.Prompt],
    ) -> types.GenerateResponse:
        # Runs prompt on the embedding cached under key, the model's name and the
        # image's hash, or on image. Recent image embeddings stay cached, so
        # several prompts on one image encode it once.
        # A SAM3 embedding takes about 100 MB.
        # ponytail: fixed size, make it configurable if memory or hit rate matters.
        EMBEDDING_CACHE_SIZE: Final = 4

        model: types.Model = self._load(model_name=key[0])
        embedding: Optional[types.ImageEmbedding] = self._embeddings.get(key)
        # Models use the embedding when given one, and return the one they used.
        response: types.GenerateResponse = model.generate(
            request=types.GenerateRequest(
                model=model.name, image=image, image_embedding=embedding, prompt=prompt
            )
        )
        if response.image_embedding is not None:
            self._embeddings[key] = response.image_embedding
            self._embeddings.move_to_end(key)
            while len(self._embeddings) > EMBEDDING_CACHE_SIZE:
                self._embeddings.popitem(last=False)
        return response

    def _load(self, *, model_name: str) -> types.Model:
        # Under _lock. When full, the least recently used model unloads first;
        # one call runs at a time, so that model is idle.
        if model_name in self._loaded:
            with self._loaded_lock:
                self._loaded.move_to_end(model_name)
            return self._loaded[model_name].model
        model_type: type[types.Model] = _registry.get_model_type_by_name(
            name=model_name
        )
        # Before unloading anything, so a failed download leaves the others loaded.
        model_type.pull()
        while len(self._loaded) >= self._max_loaded_models:
            self._unload(model_name=next(iter(self._loaded)))
        model: Optional[types.Model] = None
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
        loaded: Optional[_Loaded] = self._loaded.get(model_name)
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
