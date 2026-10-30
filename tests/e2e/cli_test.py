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


def test_run_threshold_uses_default_prompt():
    cmd = ["osam", "run", "efficientsam:10m", "--image", str(image_path)]
    output = subprocess.check_output(cmd + ["--score-threshold", "0.5", "--json"])
    assert len(json.loads(output)["annotations"]) == 1


def test_run_prompt_from_file_and_stdin(tmp_path: pathlib.Path):
    prompt = '{"points": [[1439, 504]], "point_labels": [1]}'
    prompt_file = tmp_path / "prompt.json"
    prompt_file.write_text(prompt)
    cmd = ["osam", "run", "efficientsam:10m", "--image", str(image_path), "--json"]

    from_file = subprocess.check_output(cmd + ["--prompt", f"@{prompt_file}"])
    from_stdin = subprocess.check_output(cmd + ["--prompt", "-"], input=prompt.encode())

    assert json.loads(from_file) == json.loads(from_stdin)


def test_run_prompt_file_missing_rejected(tmp_path: pathlib.Path):
    result = subprocess.run(
        ["osam", "run", "efficientsam:10m", "--image", str(image_path)]
        + ["--prompt", f"@{tmp_path / 'missing.json'}"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "missing.json" in result.stderr
