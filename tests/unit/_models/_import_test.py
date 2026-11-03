import subprocess
import sys


def test_a_model_imports_first_in_a_fresh_interpreter() -> None:
    # yoloworld once imported osam._apis, which imports every model, so
    # importing a model before osam._models failed on the cycle.
    subprocess.run(
        [sys.executable, "-c", "from osam._models.sam3 import Sam3"], check=True
    )
