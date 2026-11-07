---
name: worldrsi
description: Operate or resume the WorldRSI Physics-IQ learning experiment through direct role dispatch and durable file commands.
---

# WorldRSI operation

Read [protocol](../../docs/protocol.md) for experimental decisions and
[operation](../../docs/operation.md) for commands and recovery. Use API-World and
`./wrsi`. Main Codex dispatches the five roles through each packet's `dispatch.json`
and registers real session identities and usage.

Start from the frozen data index and initial Harness. All eleven learning sources
share a frozen round-start Harness and history. Learning and promotion use the
fixed learning view and seeds 42–45. Reflector receives public input, learning
evidence, take-1 GT and its scoring mask; it assesses sufficient gain relative to
the initial mean. Sufficient-gain bundles enter shared history after the round.
Editor receives all closed summaries, including unsuccessful and rejected rounds.

After all sources close, dispatch Manual Manager, publish manuals, then dispatch
Editor. `run candidate` prepares the complete candidate and a baseline reusing the
round's initial results. Each promotion evaluation has its own Scorer. Generate
and score the candidate on the same eleven inputs, verify baseline scores,
finalize both reports, then call `run decide`. A measured score decline returns
`revision_required` for at most three additional candidates. Continue Editor with
a fresh packet containing regression evidence; publish its revised policy/templates
and revision.md, retaining the staged manuals and round-start baseline. Evaluate
all eleven inputs and four seeds again with fresh Executors and a new Scorer.
Keep the round open until a final accepted/rejected decision. Intermediate failures
do not affect the rejection streak; infrastructure retries resume the same attempt. Never substitute branch-best scores
for the synthesized Harness's performance. Ordinary Executors receive the single
synthesized decision skill; learning branches retain experimental instructions.

Each learning source has one Scorer across attempts. Each promotion or test plan
has its own Scorer using gpt-6.1-sol/high, calling scripts only. One batch covers
all planned outputs and seeds. Resume immutable packages, submissions and evidence.

Tests are optional and cover the 33 held-out scenes. `run test --round N` selects
that round's candidate by default; `--attempt N` selects an earlier candidate; `--target active` selects its post-decision
Harness, and round 0 selects the initial Harness. Quick/full controls test views
only. `eval extend` adds missing test views while retaining existing outputs.
Tests never gate the next round or enter learning feedback. Main Codex owns test
reports. For an existing run, consult its local continuation document and perform
one-off state operations directly; there is no migration helper.
