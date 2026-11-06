import importlib.util
import pathlib

_PATH = pathlib.Path(__file__).resolve().parents[2] / "examples" / "benchmark.py"
_spec = importlib.util.spec_from_file_location("benchmark", _PATH)
assert _spec is not None and _spec.loader is not None
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)


def test_encode_cell_subtracts_the_cached_image() -> None:
    assert benchmark._encode_cell(new_ms=1000, cached_ms=30) == "970"


def test_encode_cell_is_a_dash_without_an_embedding() -> None:
    # YOLO-World runs in full on a cached image too.
    assert benchmark._encode_cell(new_ms=100, cached_ms=101) == "-"
    assert benchmark._encode_cell(new_ms=101, cached_ms=100) == "-"
