import io
import os
import shlex
import subprocess

import PIL.Image

here = os.path.dirname(os.path.abspath(__file__))


def test_run():
    cmd = f"osam run efficientsam:10m --image {here}/../examples/_images/dogs.jpg"
    output = subprocess.check_output(shlex.split(cmd))
    image = PIL.Image.open(io.BytesIO(output))
    assert image.size == (2560, 1600)
