# Operation and recovery

Use `./wrsi` in the API-World environment. All directories passed to commands are
durable paths under `/data`. Main Codex drives role dispatch; file commands produce
packets, immutable artifacts and explicit pending states.

## Preparation

```bash
./wrsi doctor
./wrsi data prepare
./wrsi snapshot freeze
./wrsi run init RUN_NAME --index INDEX_DIR --snapshot SNAPSHOT_DIR
./wrsi run initial-validation RUN_DIR
```

The snapshot command returns its immutable path. Evaluation preparation writes
evaluation.json and seeds/42/TASK_ID through seeds/45/TASK_ID. Initial validation
automatically prepares initial testing and references its
validation subset. Dispatch one Executor per test input at seed 42, then use the
shared evaluation procedure below. Start learning with `./wrsi run begin RUN_DIR`.
Score the full test plan once, then have the
validation Scorer run its batch to verify and reuse the matching test scores.
Finalize both reports. After initial validation:

```bash
./wrsi run set-reference RUN_DIR
```

`run begin` creates eleven sources and returns their directories, initial tasks,
snapshot and frozen history. Repeating it resumes the current round.
Initial evaluation and the first learning batch run concurrently; register the
initial reference before candidate validation or promotion.

## Role dispatch

Use `--mode quick` on `run init` for first-view validation and testing, or
`--mode full` for all three views. Both modes use four seeds. The run records this
choice. Learning uses the single view already assigned to each scene.

`wrsi run initial-test RUN_DIR` returns the initial test plan also used by initial
validation. Their shared outputs are generated and scored only once. Record the completed
initial test with `wrsi run register-initial-test RUN_DIR`.

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
Scorer across its attempts; each validation and test evaluation has
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
and skips finished scores. The same commands serve validation and tests. Each
learning attempt uses the same batch command for its one input × four seeds.
For overlapping validation and testing, score the test first. Prepare the
validation projection, dispatch/register its own Scorer as above, and have it
call the batch script to verify and reuse the test scores without remeasurement:

```bash
./wrsi eval validation-from-test TEST_EVALUATION_DIR --destination VALIDATION_DIR
./wrsi score batch VALIDATION_DIR --definition VALIDATION_SCORER_PACKET/input.json
./wrsi eval finalize VALIDATION_DIR
```

The projection requires matching Agent, data, intervention, inputs, seeds and
scorer, and retains the source test plan and task/score identities. Existing frozen
validation plans are resumed unchanged; use a new destination to create a
projection. A full extension of a quick projection reuses its existing test views
and evaluates the remaining views in one batch. Failure
to produce a valid final video is recorded with `wrsi output fail TASK --reason
TEXT`, then scored by that task/evaluation's Scorer. Pending external/scoring failures remain
recoverable from their registered submissions and saved artifacts.

## Round promotion and tests

`wrsi eval status EVALUATION_DIR` lists inputs needing modeling and outstanding
generations. Completed outputs are retained when submitting or collecting again.

To complete a quick evaluation:

```bash
./wrsi eval extend QUICK_EVALUATION_DIR --destination FULL_EVALUATION_DIR
./wrsi eval status FULL_EVALUATION_DIR
```

Dispatch Executors for the listed modeling tasks, then use the usual submit,
collect, score and finalize commands. The plan shares the quick evaluation's
task directories and creates tasks for the remaining two views. Its snapshot,
data revision, seeds and boundary remain bound. An epoch's full report can be
registered after its quick report, including after learning has advanced to
another snapshot. Register an initial test extension with
`wrsi run register-initial-test RUN_DIR --report FULL_EVALUATION_DIR`.

When all eleven sources are closed with verified learning evidence, dispatch Manual
Manager, then publish its complete manuals and dispatch Editor. Editor receives
all current and historical source summaries, including unsuccessful outcomes.
It writes the complete synthesized policy in `policy/skills/worldrsi-decision/SKILL.md`
under its output directory and reusable implementations under templates. Policy
publication accepts only this single decision skill; raw source experience
attachments are not part of the Executor policy. Executor packets expose its
frozen path as `decision_experience`:

```bash
./wrsi run manual-inputs RUN_DIR
./wrsi run publish-manuals RUN_DIR
./wrsi run candidate RUN_DIR --policy CANDIDATE_POLICY --templates CANDIDATE_TEMPLATES
```

Execute the candidate's validation evaluation and finalize its report. Then:

```bash
./wrsi run decide RUN_DIR
./wrsi run status RUN_DIR
./wrsi run test RUN_DIR
./wrsi run register-test RUN_DIR --report TEST_REPORT
./wrsi run report RUN_DIR
```

`run test` prepares the current pending boundary. Complete all 198 inputs using
the common four-seed procedure and register the report before another round.
`run history` returns `sufficient_gain` bundles from completed rounds and aggregate
validation outcomes, including rejected candidates.
`run report` includes source closure status, tests and costs by phase. Long commands
run in named tmux sessions with logs under logs/; record the session and log path.

Existing indexes, task bindings, submissions, snapshots and batch-start histories
remain frozen when resuming a run. Newly assembled history uses the source's
`sufficient_gain` status. Legacy transfer artifacts remain archived and their
costs remain reportable; their receipts are not required for source completion.
