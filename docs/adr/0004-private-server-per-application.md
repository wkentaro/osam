# One private osam server per application

Every application that uses osam runs its own osam server as a child process
and talks to it over the same HTTP API that `osam serve` exposes. The public
Python API becomes `osam.Client`. Applications share the model files on disk
in `~/.cache/osam`, never a running server, unless the user points them at
one with `host=` or `OSAM_HOST`.

## Context

osam runs inside the application today. labelme imports `osam.apis` and
`osam.types.Model` and runs them in a child process it manages itself
(`labelme/_automation/_osam_worker.py`): pipes, an image cache keyed by image
id, an embedding cache, timeouts, retry on an onnxruntime fault, and device
selection through `OSAM_ONNX_PROVIDERS`. Each of those is server work, and
the next application that embeds osam would have to write it again.

SAM models spend most of their time encoding the image, and each prompt
after that is cheap to decode. Only a process that outlives one call can keep
the encoded image, so the cache belongs in a long-lived server.

Applications ship standalone. labelme is installed with pip or as a
PyInstaller bundle and cannot ask the user to install and run a separate
osam.

## Considered options

- **Ask the user to install a shared server, as Ollama does.** Rejected: the
  Ollama CLI and its Python client only connect to `OLLAMA_HOST` and rely on
  the desktop app or a systemd service to keep the server running. An
  application that embeds osam would stop working on its own.
- **Find a running server and share it automatically.** Rejected: two
  applications that bundle different osam versions would reach the same
  server, and the API would not match. Someone has to stop the server after
  the application that started it exits, and two applications starting at
  once race to create it. The gain is
  GPU memory when two applications run at the same time; the model files are
  already shared on disk.
- **One private server per application (chosen).** Language servers work
  this way: the LSP specification leaves starting and stopping the server to
  the client, and each editor runs its own. Playwright for Python starts its
  own driver process per instance. To use a browser that runs elsewhere, the
  caller passes its endpoint to `connect(endpoint)` explicitly.

## Decision

- `osam.Client()` starts a private server by default. It uses
  `multiprocessing` with the spawn start method, so a frozen application
  such as a PyInstaller bundle can start it after calling
  `multiprocessing.freeze_support()`.
- The private server binds `127.0.0.1` on port 0, reports the port to its
  parent, and accepts only requests that carry a random token from the
  parent.
- The private server exits when the pipe to its parent closes, so it does not
  outlive a crashed application. LSP's `--clientProcessId` serves the same
  purpose.
- `osam.Client(host=...)` and `OSAM_HOST` connect to an existing server, such
  as `osam serve` on a GPU machine. The client checks `/api/version` and
  fails with an error naming both versions when major and minor differ, as
  Playwright does.
- The client restarts the private server after an onnxruntime fault. A failed
  CUDA model load leaves memory held in the process, so later loads fail
  until it restarts.
- Image embeddings stay inside the server. Clients send the image with each
  request, and the server reuses the embedding it cached for that image.
- The device is a server setting (`osam serve --device`), not a request
  field, since one server holds one loaded model.
- Where a concept matches Ollama, the API follows Ollama: `keep_alive`, pull
  progress as an NDJSON stream, `/api/ps`, `/api/version`, `OSAM_HOST`, and
  errors as `{"error": ...}`.
- The osam CLI is a client too. `osam.apis` becomes internal to the server.

## Consequences

- This is a breaking release. Applications that call `osam.apis` or
  `Model.generate` move to `osam.Client`. labelme caps `osam<0.11` until it
  moves.
- Every application pays to start a process and load the model once per
  session. Two applications running at once each hold their own model in
  memory.
- Debugging spans two processes, and server logs need to reach the
  application.
- Video tracking keeps state across frames and gets an explicit session
  resource when it is added. It is the only stateful part of the API.
