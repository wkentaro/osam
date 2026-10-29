import io
import json
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


def test_run_point_flags_json():
    cmd = [
        "osam",
        "run",
        "efficientsam:10m",
        "--image",
        str(image_path),
        "--point",
        "1439,504",
        "--point",
        "1439,1289,0",
        "--device",
        "cpu",
        "--json",
    ]
    response = json.loads(subprocess.check_output(cmd))
    assert len(response["annotations"]) == 1


def test_run_prompt_with_flags_rejected():
    result = subprocess.run(
        ["osam", "run", "efficientsam:10m", "--image", str(image_path)]
        + ["--prompt", "{}", "--text", "dog"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--prompt can't be combined" in result.stderr


def test_run_bad_point_rejected():
    result = subprocess.run(
        ["osam", "run", "efficientsam:10m", "--image", str(image_path)]
        + ["--point", "1,2,3,4"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "expected 2 or 3 numbers" in result.stderr


def test_run_point_label_rejected():
    result = subprocess.run(
        ["osam", "run", "efficientsam:10m", "--image", str(image_path)]
        + ["--point", "1,2,2"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "label must be 0 or 1" in result.stderr
