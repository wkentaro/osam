# Triage Labels

This repository uses the canonical issue and pull-request labels below. The
label descriptions and colors are managed on GitHub; this file defines how
agents use them.

## Role mapping

The `/triage` roles map to these labels by object:

| Role | Issue label | PR label |
| --- | --- | --- |
| `bug` | `type:bug` | — |
| `enhancement` | `type:feature` | — |
| `needs-triage` | `needs-triage` | `recommend-triage` |
| `needs-info` | `needs-info` | `recommend-revise` |
| `ready-for-agent` | `ready-for-agent` | no verdict; non-draft PR |
| `ready-for-human` | `ready-for-human` | `recommend-merge` |
| `wontfix` | `wontfix` | `recommend-close` |

Maintenance, refactor, documentation, and test issues use `type:task`.

## Rules

- A triaged issue carries exactly one `type:` label and one triage state. An
  unlabeled issue is untriaged; `needs-triage` means under evaluation.
- A non-draft PR with no verdict is the agent's to finalize. The draft flag is
  the still-being-built state; there is no label for it.
- The agent emits at most one `recommend-*` verdict per PR head.
  `recommend-revise` hands the PR back to its author; the other three hand it to
  the maintainer.
- Verdicts are recommendations. The agent never merges or closes. Keep
  `recommend-merge`; merge bots may watch that exact string.
- `maintainer-approved` is set only on explicit maintainer direction and may
  coexist with a `recommend-*` label.
- A new push makes any verdict stale. The authority that set it clears and
  renews it.

## Issue labels

### Type axis

Every triaged issue carries exactly one `type:` label:

| Label | Meaning |
| --- | --- |
| `type:bug` | Reporting a defect to fix |
| `type:feature` | Requesting a new capability or improvement |
| `type:task` | Other work: maintenance, refactor, docs |

### Triage axis

Every triaged issue also carries exactly one triage label:

| Label | Meaning |
| --- | --- |
| `needs-triage` | Under evaluation, not yet routed |
| `needs-info` | Waiting on the reporter for more information |
| `ready-for-agent` | Fully specified and ready for an AFK agent |
| `ready-for-human` | Requires human implementation |
| `wontfix` | Will not be actioned |

## Pull-request labels

After finalizing a PR, the agent applies exactly one mutually exclusive verdict:

| Label | Meaning |
| --- | --- |
| `recommend-merge` | Agent finalized and endorses it: review and merge |
| `recommend-close` | Agent recommends closing: your call to review or close |
| `recommend-triage` | Code is sound; merge or close is a product call |
| `recommend-revise` | Review found defects or questions; author must revise and push |

`maintainer-approved` is a separate, explicit-human-only verdict. A maintainer
may apply it after reviewing the current PR head; an agent must never infer it
from an agent verdict, green CI, or mergeability. It may coexist with a
`recommend-*` label. Required checks remain authoritative, and the maintainer
still performs the merge.
