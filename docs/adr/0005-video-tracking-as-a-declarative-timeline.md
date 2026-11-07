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
- Actions name objects by where the user clicked, never by id:
  - `{"type": "prompt", "frame", ...}` with the fields of `Prompt` is the
    first action, and only the first. With `points` or `box` it tracks one
    object. With `texts` it finds every match. `example_boxes` without
    `texts` is an error.
  - `{"type": "add", "frame", "points", "point_labels", "box"}` adds an
    object.
  - `{"type": "refine", "frame", "points", "point_labels"}` refines the
    object whose mask on that frame contains the first positive point.
  - `{"type": "remove", "frame", "point"}` removes the object whose mask on
    that frame contains the point.

  When several masks contain the point, the target is the one listed first in
  that frame's annotations. When none does, the server answers 409 with
  `"code": "no_object_at_point"`. Coordinates are pixels, as in
  `/api/segment`.
- Action k applies to the result of actions 0 to k-1 propagated over every
  frame, in both directions from the first prompt frame. The server works out
  what it must recompute.
- The response is NDJSON with one line per frame,
  `{"frame", "annotations"}`, then a last line with `"done": true`, each
  object's first and last frame, and whether the request hit the cache.
  `first_frame` and `last_frame` limit the frames returned, and
  `include_masks: false` leaves out the masks.
- The server caches tracker state by model, device, frames, and action
  prefix, and image features by model and image reference. Both caches are
  bounded in bytes. `/api/segment` uses the same feature cache.
- `Annotation` gains `object_id`, numbered in the order objects appear. Ids
  label objects in responses only. Errors gain a stable `code`, and
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
- A request means the same thing on any device, since no action depends on
  how the server numbered objects. A replay on another device can still
  order near-equal detections differently, which changes the ids in the
  response, or leave no object under a click, which fails with 409 instead
  of editing another object.
- Ids in responses stay the same while the server replays on one device,
  and can change on another. A client that attaches its own names to objects,
  such as labels in labelme, has to match them again after that.
- Requests grow with the number of frames and actions, by about 70 bytes per
  frame reference.
