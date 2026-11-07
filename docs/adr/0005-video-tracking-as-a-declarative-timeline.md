# Video tracking as a declarative timeline

osam tracks objects through a video with one endpoint, `POST /api/track`. A
request names the frames by reference and lists every action the user has
taken so far, in order. The server streams one line per frame. It keeps
tracker state only as a cache keyed by that list, so the client holds the
source of truth. This replaces the session resource that ADR 0004 expected
video tracking to need.

## Context

SAM3's video predictor keeps state across calls: a memory bank per object,
one tracker per batch of objects, cached outputs per frame, and an action
history that decides whether the next propagation runs in full, for the
refined objects only, or returns cached outputs.

ADR 0004 gives each application a private server that the client restarts
after a timeout or an onnxruntime fault. Any state that lives only on the
server is lost at that restart.

`/api/segment` already treats server state as a cache. A client sends an
image once, then refers to it by hash, and the server answers 412 when it no
longer has it.

## Considered options

- **Upstream's session calls** (`start_session`, `add_prompt`, `propagate`,
  `remove_object`, `close_session`, as in `Sam3VideoPredictor`). Rejected: a
  restart drops every session, a retried `add_prompt` adds the prompt twice,
  and a client that never closes a session leaks its memory.
- **REST resources** (`/videos/{id}`, `/videos/{id}/objects/{id}`).
  Rejected for the same restart problem. Propagation is a computation, not a
  resource, and modelling it as one hides the full, partial, and cached modes
  behind create and update calls.
- **One call with the whole video and one prompt.** Rejected as the only
  form, since it can't refine an object. It is the timeline with one action.
- **A declarative timeline (chosen).** It follows `/api/segment`: content
  references, 412 for what the server lacks, and server state as a cache.

## Decision

- A request has `model`, `frames`, and `actions`. Each frame takes the same
  value as `image` in `/api/segment`: a base64 image file, or a reference
  `"sha256:<hex>"` to an image the server has seen. Frames that the server
  lacks get 412 with `"code": "missing_images"` and their references, and the
  client resends those inline.
- osam takes a list of images, not a video file, so it needs no video
  decoder. A labelme folder, frames a user extracted with ffmpeg, and frames
  from a camera are all lists of images.
- Actions:
  - `{"type": "prompt", "frame", ...}` with the fields of `Prompt` is the
    first action, and only the first. With `points` or `box` it tracks one
    object. With `texts` it finds every match. `example_boxes` without
    `texts` is an error.
  - `{"type": "add", "frame", "points", "point_labels", "box"}` adds an
    object.
  - `{"type": "refine", "frame", "object_id", "points", "point_labels"}`
    corrects an object with positive and negative points.
  - `{"type": "remove", "object_id"}` removes an object.

  Coordinates are pixels, as in `/api/segment`.
- Every action after the first carries `base`: the `result` digest of the
  response the user saw when making it. The `done` line reports `result`, a
  digest of the object ids and masks on every frame of the video, whatever
  `first_frame`, `last_frame`, and `include_masks` returned. Before applying
  an action, the server checks that the result it holds or replayed for the
  earlier actions has that digest, and answers 409 with
  `"code": "base_changed"` and the new digest when it does not. This is the
  If-Match of HTTP. It keeps ids safe to refer to, since a replay that numbers
  objects differently, or segments them differently, fails before any edit.
- Action k applies to the result of actions 0 to k-1 propagated over every
  frame, in both directions from the first prompt frame. The server works out
  what it must recompute.
- The response is NDJSON with one line per frame,
  `{"frame", "annotations"}`, then a last line with `"done": true`, each
  object's first and last frame, `result`, and whether the request hit the
  cache.
  `first_frame` and `last_frame` limit the frames returned, and
  `include_masks: false` leaves out the masks.
- The server caches tracker state by model, device, frames, and action
  prefix, and image features by model and image reference. Both caches are
  bounded in bytes. `/api/segment` uses the same feature cache.
- `Annotation` gains `object_id`, numbered in the order objects appear.
  Errors gain a stable `code`, and
  `/api/models` lists each model's `capabilities` and `prompt_kinds`.
- `osam run --video DIR` takes a directory of images.

## Consequences

- A request is idempotent, so a client or an agent can retry it. A server
  restart costs a cache miss, not lost work.
- Undo is the previous list. The server still has that prefix cached, so it
  runs no model.
- A cache miss replays every action, which takes minutes on a long video.
  The `done` line reports a miss so a client can explain the wait.
- The server derives the propagation mode from the list. That logic moves
  from the client-facing API into the runtime, where upstream keeps it too.
- An edit applies only to the result the user saw. Replays on the same
  device reproduce it. A replay on another device, such as after a GPU falls
  back to CPU, usually changes some mask and fails with 409, so the client
  shows the new result and the user makes the edits again.
- Requests grow with the number of frames and actions, by about 70 bytes per
  frame reference.
