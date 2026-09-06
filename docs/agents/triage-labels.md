# Triage Labels

This repository uses the canonical issue and pull-request labels below. The
label descriptions and colors are managed on GitHub; this file defines how
agents use them.

## Issue labels

### Type axis

Every triaged issue carries exactly one `type:` label:

| Label | Meaning |
| --- | --- |
| `type: bug` | Reporting a defect to fix |
| `type: feature` | Requesting a new capability or improvement |
| `type: task` | Maintenance, refactor, documentation, test, or other work |

### Triage axis

Every triaged issue also carries exactly one triage label:

| Role | Label | Meaning |
| --- | --- | --- |
| Maintainer must evaluate | `needs-triage` | The maintainer must decide whether and how to proceed |
| Waiting for information | `needs-info` | Waiting on the reporter or another outside human |
| Ready for the agent | `ready-for-agent` | Fully specified and ready for an AFK agent |
| Requires human implementation | `ready-for-human` | The maintainer must implement the work |
| Will not proceed | `wontfix` | The issue will not be actioned |

An issue with no triage label is fresh work for the agent to route. Do not use
`needs-triage` for fresh work; it is reserved for a decision that belongs to a
maintainer. `needs-info` is shared with pull requests when an outside human is
blocking progress.

## Pull-request labels

A non-draft PR with no verdict label is fresh work for the agent to finalize.
The native GitHub draft flag is the in-progress state.

After finalizing a PR, the agent applies exactly one mutually exclusive verdict:

| Label | Meaning |
| --- | --- |
| `recommend-merge` | The agent finalized the PR and endorses review and merge |
| `recommend-close` | The agent recommends closing it because it is broken, abandoned, superseded, or out of scope |
| `recommend-triage` | The code is sound, but the maintainer must make the product or scope decision |

`maintainer-approved` is a separate, explicit-human-only verdict. A maintainer
may apply it after reviewing the current PR head; an agent must never infer it
from an agent verdict, green CI, or mergeability. It may coexist with a
`recommend-*` label. Required checks remain authoritative, and the maintainer
still performs the merge.

Verdict labels record recommendations or decisions; they never merge or close a
PR. Both verdict axes apply to one specific diff. When a new commit changes the
PR, the stale verdict labels must be removed and renewed by their authority.
