# Blob download endpoint override

osam downloads each model blob from a content-addressed, SHA-256-verified URL. We
added `OSAM_BLOB_ENDPOINT`, an ordered comma-separated list of endpoints (with the
reserved keyword `direct` for the canonical URL) that `Blob.pull` tries in turn,
fetching each blob from `<endpoint>/<sha256-hex>`.

## Context

Some networks block the canonical model hosts (GitHub release assets, Hugging
Face). Consumers that redistribute osam needed a way to point downloads at a
reachable mirror without forking osam or rewriting URLs at runtime.

## Considered options

- **Monkey-patch `Blob.pull` downstream** to rewrite URLs. Rejected: couples
  callers to a private method and breaks on internal refactors.
- **Single mirror URL env var.** Rejected: no graceful fallback when the mirror
  is incomplete or down.
- **Ordered endpoint list with `direct` fallback (chosen).** Mirrors the
  `GOPROXY` / `pip --index-url` idiom; one knob expresses mirror-only,
  mirror-first, and origin-only.

## Consequences

- Mirrors are keyed by SHA-256 hex; a mirror need only host blobs under their
  digest. Integrity is still enforced by the existing hash check, so a mirror
  cannot serve corrupted data.
- Unset behaviour is unchanged (`direct` only), so existing installs are
  unaffected.
