import io
import pathlib
import subprocess

import PIL.Image

image_path = (
    pathlib.Path(__file__).resolve().parents[2] / "examples" / "_images" / "dogs.jpg"
)


def test_run():
    cmd = ["osam", "run", "efficientsam:10m", "--image", str(image_path)]
    output = subprocess.check_output(cmd)
    image = PIL.Image.open(io.BytesIO(output))
    assert image.size == (2560, 1600)
