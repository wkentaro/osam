import numpy as np

from osam.types import Annotation
from osam.types import BoundingBox


def test_json_roundtrip_decodes_mask() -> None:
    mask = np.array([[True, False, True], [False, True, False]])
    original = Annotation(
        bounding_box=BoundingBox(xmin=0, ymin=0, xmax=2, ymax=1), mask=mask
    )
    restored = Annotation.model_validate_json(original.model_dump_json())
    np.testing.assert_array_equal(restored.mask, mask)
