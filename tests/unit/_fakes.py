# Models with no weights, for Runtime(model_types=...): they load, encode, and
# decode at once.
import numpy as np

from osam import _types as types


class FakeModel(types.Model):
    name = "fake:latest"
    metadata = types.ModelMetadata(license_name="MIT", license_url="", source_url="")
    prompt_kinds = frozenset({"points"})
    _blobs = {}
    # The device it lands on when Runtime picks one.
    device: str = "cpu"

    def __init__(self, device: types.Device | None = None, *, auto_pull: bool = True):
        if device is not None:
            self.device = device

    def encode_image(self, *, image: np.ndarray) -> types.ImageEmbedding:
        return types.ImageEmbedding(
            original_height=image.shape[0],
            original_width=image.shape[1],
            embedding=np.zeros((1, 1, 1), dtype=np.float32),
        )

    @classmethod
    def default_prompt(cls, *, height: int, width: int) -> types.Prompt:
        return types.Prompt()

    def _decode(
        self, *, embedding: types.ImageEmbedding, prompt: types.Prompt
    ) -> list[types.Annotation]:
        return []


def model_type(*, name: str, device: str = "cpu") -> type[FakeModel]:
    # A new class each call, so a test can patch it alone.
    class Fake(FakeModel):
        pass

    Fake.name, Fake.device = name, device
    return Fake
