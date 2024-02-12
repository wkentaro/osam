import contextlib
import os
import sys
import tempfile


@contextlib.contextmanager
def suppress():
    original_stdout_fd = os.dup(sys.stdout.fileno())
    original_stderr_fd = os.dup(sys.stderr.fileno())

    with tempfile.TemporaryFile(mode="w+b") as temp_stdout, tempfile.TemporaryFile(
        mode="w+b"
    ) as temp_stderr:
        os.dup2(temp_stdout.fileno(), sys.stdout.fileno())
        os.dup2(temp_stderr.fileno(), sys.stderr.fileno())

        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()

            os.dup2(original_stdout_fd, sys.stdout.fileno())
            os.dup2(original_stderr_fd, sys.stderr.fileno())

            os.close(original_stdout_fd)
            os.close(original_stderr_fd)
