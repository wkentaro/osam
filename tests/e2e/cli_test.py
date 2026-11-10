import io
import json
import os
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


def test_run_invalid_prompt_rejected_before_loading_model():
    result = subprocess.run(
        ["osam", "run", "sam3", "--image", str(image_path)]
        + ["--prompt", '{"points": [[1, 1]]}'],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Initialized inference sessions" not in result.stderr


def test_run_threshold_flag_overrides_prompt():
    cmd = ["osam", "run", "efficientsam:10m", "--image", str(image_path), "--json"]
    prompt = '{"points": [[1439, 504]], "point_labels": [1], "score_threshold": 0.1}'
    output = subprocess.check_output(
        cmd + ["--prompt", prompt, "--score-threshold", "0.5"]
    )
    assert len(json.loads(output)["annotations"]) == 1


def test_run_prompt_in_old_fields_rejected():
    result = subprocess.run(
        ["osam", "run", "efficientsam:10m", "--image", str(image_path)]
        + ["--prompt", '{"points": [[1, 1], [2, 2]], "point_labels": [2, 3]}'],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1  # point_labels are 0 or 1; boxes go in "box"
    result = subprocess.run(
        ["osam", "run", "efficientsam:10m", "--image", str(image_path)]
        + ["--prompt", '{"prompt": {"texts": ["dog"]}}'],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "keys among" in result.stderr


def test_run_boxes_need_text():
    cmd = ["osam", "run", "sam2:tiny", "--image", str(image_path)]
    for flags, message in [
        (["--box", "0,0,9,9", "--box", "1,1,8,8"], "give one --box"),
        (["--negative-box", "0,0,9,9"], "--negative-box needs --text"),
        (["--point", "1,1", "--text", "dog"], "--point can't be combined"),
    ]:
        result = subprocess.run(cmd + flags, capture_output=True, text=True)
        assert result.returncode == 2, flags
        assert message in result.stderr, flags


def test_run_negative_max_annotations_rejected():
    result = subprocess.run(
        ["osam", "run", "efficientsam:10m", "--image", str(image_path)]
        + ["--max-annotations", "-1"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2


def test_unreachable_host_reported_in_one_line():
    result = subprocess.run(
        ["osam", "list"],
        capture_output=True,
        text=True,
        env={**os.environ, "OSAM_HOST": "127.0.0.1:1"},
    )
    assert result.returncode == 1
    # onnxruntime can print device-discovery warnings first, as it does on
    # GitHub's runners from 1.24.
    assert result.stderr.splitlines()[-1].startswith("Error: can't reach 127.0.0.1:1")
    assert "Traceback" not in result.stderr


def test_failed_server_start_reported_in_one_line(tmp_path: pathlib.Path):
    (tmp_path / "uvicorn.py").write_text("raise ImportError('broken uvicorn')\n")
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    env.pop("OSAM_HOST", None)
    result = subprocess.run(["osam", "list"], capture_output=True, text=True, env=env)
    assert result.returncode == 1
    assert result.stderr.splitlines()[-1] == (
        "Error: osam server failed to start: ImportError: broken uvicorn"
    )
    assert "Traceback" not in result.stderr
