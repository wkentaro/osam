import pytest

from osam._models import sam


@pytest.mark.parametrize(
    "height, width, expected",
    [
        (3000, 4000, (768, 1024)),
        (4000, 3000, (1024, 768)),
        (1, 10000, (1, 1024)),
        (10000, 1, (1024, 1)),
    ],
)
def test_compute_resized_hw_keeps_every_side_at_least_one_pixel(
    height: int, width: int, expected: tuple[int, int]
) -> None:
    resized = sam._models._compute_resized_hw(
        height=height, width=width, target_size=1024
    )
    assert resized == expected
