import numpy as np
import numpy.typing as npt
import PIL.Image


def compute_scale_to_resize_image(
    height: int, width: int, image_size: int
) -> tuple[float, int, int]:
    scale: float
    new_height: int
    new_width: int
    if width > height:
        scale = image_size / width
        new_height = int(round(height * scale))
        new_width = image_size
    else:
        scale = image_size / height
        new_height = image_size
        new_width = int(round(width * scale))
    return scale, new_height, new_width


def resize_image(image: npt.NDArray, image_size: int) -> tuple[float, npt.NDArray]:
    scale: float
    new_height: int
    new_width: int
    scale, new_height, new_width = compute_scale_to_resize_image(
        height=image.shape[0],
        width=image.shape[1],
        image_size=image_size,
    )
    scaled_image: npt.NDArray = np.asarray(
        PIL.Image.fromarray(image).resize(
            (new_width, new_height),
            resample=PIL.Image.BILINEAR,  # type: ignore[attr-defined]
        )
    )
    return scale, scaled_image
