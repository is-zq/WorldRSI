# Operation and recovery

Use `./wrsi` in the API-World environment. All directories passed to commands are
durable paths under `/data`. Main Codex drives role dispatch; file commands produce
packets, immutable artifacts and explicit pending states.

## Preparation and role dispatch

```bash
./wrsi doctor
./wrsi data prepare
./wrsi snapshot freeze
./wrsi run init RUN_NAME --index INDEX_DIR --snapshot SNAPSHOT_DIR --mode quick
./wrsi run begin RUN_DIR
```

Use the immutable snapshot path returned by `snapshot freeze`. No initial test or
reference report is required. `run begin` creates eleven sources from the frozen
round-start Harness and history; repeating it resumes the open round. Quick/full
controls only held-out tests, never the fixed learning or promotion views.

```bash
./wrsi agent packet PACKET_DIR --role executor --task-dir TASK_DIR
./wrsi agent packet PACKET_DIR --role scorer --task-dir LEARNING_ATTEMPT_DIR
./wrsi agent packet PACKET_DIR --role scorer --evaluation-dir EVALUATION_DIR
./wrsi agent packet PACKET_DIR --role reflector --source-dir SOURCE_DIR
./wrsi agent packet PACKET_DIR --role manual_manager --run-dir RUN_DIR
./wrsi agent packet PACKET_DIR --role editor --run-dir RUN_DIR
```

Use a new packet directory for updated evidence. Read dispatch.json and call its
specified collaboration method and arguments. Each learning source task has one
Scorer across its attempts; each promotion and test evaluation has
its own Scorer. Register its first dispatch before preparing follow-up packets.
Scorer uses gpt-6.1-sol/high and only calls scripts. Persistent roles continue
through followup_task within their scope after registration. For a replacement session, prepare a new packet
with --restart --reason TEXT. Register actual model, effort and session identity:

```bash
./wrsi agent register PACKET_DIR --session-id SESSION_ID --model MODEL --effort EFFORT
./wrsi agent usage PACKET_DIR --record USAGE.json
```

Usage supports packet or cumulative scope with input_tokens, output_tokens,
total_tokens and cached_input_tokens. Missing counters remain unknown.

## Learning sources

Executor models initial/ using the task snapshot and the interfaces in tools.md.
Record decisions with `trace before TASK --id STEP --record JSON`, then checked
actions with `trace after`. Before fields: evidence, reason, local_expectation,
downstream_expectation, optional decision {factors,choice,constraints} and
decision_refs. After fields: action, artifacts, checks
[{name,passed,supports,evidence}], local_goal_met and modifications.

```bash
./wrsi tool TASK_DIR STEP_ID -- COMMAND ARGUMENTS
./wrsi h3 package TASK_DIR --render RENDER_DIR --modalities semantic depth --prompt H3_PROMPT
./wrsi eval submit LEARNING_ATTEMPT_DIR --task-id TASK_ID
./wrsi eval collect LEARNING_ATTEMPT_DIR
```

Each attempt is an evaluation plan with one input and seeds 42–45. Seed 42 lives
at the attempt root; replicas use its frozen package. After all four outputs are
locked, dispatch the source task's Scorer and use `score batch` as below. Record
usage and run `eval finalize LEARNING_ATTEMPT_DIR` before dispatching Reflector.
Reflector receives the public task (first frame and prompt), four scores, their
mean, frozen history, source-local execution evidence and `ground_truth`.
The `ground_truth` field contains the current learning view's take-1 GT (`gt_v1`)
and corresponding scoring mask (`mask_v1`), each with path, media metadata and
verified index hash. Reflector performs the analysis itself. An unfinished seed
keeps the attempt pending.
To resume an open legacy single-seed attempt, first run
`wrsi eval prepare-learning ATTEMPT_DIR` (or supply `--index` for a standalone
task). It preserves seed 42 and its submissions/scores, adding the other three
replicas. A lone seed-42 score is not complete learning feedback. Closed sources
and their frozen evidence remain unchanged. Its experiment request has id, hypothesis,
prediction, parent_attempt and executor_instruction:

```bash
./wrsi source branch SOURCE_DIR --record HYPOTHESIS.json
./wrsi source progress SOURCE_DIR
./wrsi source close SOURCE_DIR --record EXPERIENCE.json
```

A sufficient-gain experience has status:"sufficient_gain", best_attempt, reason,
evidence:[source-local paths], observations, and bundle:{trigger,instruction,scope,
implementation_checks}. The helper derives initial_score, best_score and gain from
independent scores. Other closure statuses are stalled, blocked and saturated;
blocked includes recovery_condition. Reflector judges sufficient improvement in
context and returns its finalized experience.

Checked tool updates use `wrsi manual record TASK --tool TOOL --record UPDATE.json`.
The update includes id, scope, operation, claim_review, and evidence entries with
step_id and named passed checks. Updates remain task-local until consolidation.

## Common evaluation

Dispatch a fresh Executor for each planned input at seed 42. Main submits the
same frozen package for all four seeds:

```bash
./wrsi eval submit EVALUATION_DIR --task-id TASK_ID
./wrsi eval collect EVALUATION_DIR
./wrsi eval finalize EVALUATION_DIR
```

After every planned seed output is locked, deliver one evaluation packet to the
evaluation's Scorer. It invokes the scoring script once for all tasks and seeds:

```bash
./wrsi score batch EVALUATION_DIR --definition SCORER_PACKET/input.json
```

The batch writes each output's official components and score. A retry verifies
and skips finished scores. The same commands serve promotion and tests. Each
learning attempt uses the same batch command for its one input × four seeds.
Failure to produce a valid final video is recorded with `wrsi output fail TASK --reason
TEXT`, then scored by that task/evaluation's Scorer. Pending external/scoring failures remain
recoverable from their registered submissions and saved artifacts.

## Round promotion

When all eleven sources are closed, dispatch Manual Manager, publish its complete
manuals, then dispatch Editor. Editor synthesizes current and historical summaries
into one decision skill plus reusable templates. Publish the complete candidate:

```bash
./wrsi run manual-inputs RUN_DIR
./wrsi run publish-manuals RUN_DIR
./wrsi run candidate RUN_DIR --policy CANDIDATE_POLICY --templates CANDIDATE_TEMPLATES
```

The returned `baseline_evaluation` references all eleven initial learning reports;
its `evaluation` is the candidate plan. Each covers 11 fixed learning views × seeds
42–45 regardless of test mode. Dispatch fresh Executors only for the candidate's
seed-42 modeling tasks. Submit/collect the four replicas with the common procedure.
Dispatch a separate Scorer for each evaluation. The baseline Scorer only verifies
existing initial scores; the candidate Scorer measures its complete batch. Record
actual usage, then finalize both reports before deciding:

```bash
./wrsi run decide RUN_DIR
./wrsi run status RUN_DIR
./wrsi run history RUN_DIR
./wrsi run report RUN_DIR
./wrsi run begin RUN_DIR
```

`decide` compares the candidate against this round's baseline and binds both report
hashes. A score decline returns `status: revision_required` while up to three
additional revisions remain. Keep this round open: prepare a new Editor packet,
continue its registered session, and publish its revised policy/templates using
`run candidate`. The Editor writes `revision.md` beside its policy directory.
The first candidate retains `promotion/candidate`; later ones use
`promotion/candidate-02` through `candidate-04`, with Editor outputs under
`editor/attempt-02` through `attempt-04`. Each candidate gets fresh Executors and
its own Scorer; reuse the finalized baseline. Run the full 11 × 4 evaluation and
call `decide` again. Only accepted/rejected final status permits the next round.
Repeated commands resume existing attempts; infrastructure recovery does not use
a revision. `history` supplies sufficient-gain bundles and all promotion attempts;
`report` exposes the current revision state and separated costs.
The next round does not require a test. Keep long commands in named tmux sessions
with logs under `logs/`; report the session and log path.

## Optional held-out tests

Choose the candidate of any published round, including rejected candidates:

```bash
./wrsi run test RUN_DIR --round 2 --target candidate --mode quick
```

`--attempt A` with `--target candidate --round R` selects a particular published
attempt; the default is the latest. Test plans bind the selected attempt, so later
candidate revisions do not change existing test identities. Old reports without
an attempt retain the first candidate binding.

`--target active` selects the Harness active after that round's completed decision.
`--round 0` selects the initial Harness. Target defaults to `candidate`; mode
defaults to the run's test mode. The plan contains only the 33 held-out scenes.
Quick uses the left view (132 outputs), full all views (396 outputs).

Dispatch Executors and the test's own Scorer through the common evaluation
procedure, finalize, then register the report:

```bash
./wrsi run register-test RUN_DIR --report TEST_EVALUATION_DIR
./wrsi eval extend QUICK_EVALUATION_DIR --destination FULL_EVALUATION_DIR
./wrsi eval status FULL_EVALUATION_DIR
```

An extension retains existing outputs and adds the other two views. Complete and
register it with the same commands. Registration checks the selected historical
Harness, so later learning does not change the target. Neither pending tests nor
their scores affect learning or promotion.

Indexes, task bindings, submissions, snapshots and frozen histories retain their
identities. Legacy reports remain readable. One-off changes to an existing run
are direct agent operations documented with that run, not a migration command.
