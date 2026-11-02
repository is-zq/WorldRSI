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
evaluation.json and seeds/42/TASK_ID through seeds/45/TASK_ID. Dispatch one
Executor per input at seed 42 to build its conditioning, then use the shared
evaluation procedure below. After initial validation:

```bash
./wrsi run set-reference RUN_DIR
./wrsi run begin RUN_DIR
```

`run begin` creates eleven sources and returns their directories, initial tasks,
snapshot and frozen history. Repeating it resumes the current round.

## Role dispatch

Use `--mode quick` on `run init` for first-view validation and testing, or
`--mode full` for all three views. Both modes use four seeds. The run records this
choice; transfer always evaluates the learning scene's other two views.

`wrsi run initial-test RUN_DIR` prepares the initial test independently of
initial validation. They can be dispatched concurrently. Record the completed
initial test with `wrsi run register-initial-test RUN_DIR`.

```bash
./wrsi agent packet PACKET_DIR --role executor --task-dir TASK_DIR
./wrsi agent packet PACKET_DIR --role scorer --task-dir TASK_DIR --index INDEX_DIR
./wrsi agent packet PACKET_DIR --role reflector --source-dir SOURCE_DIR
./wrsi agent packet PACKET_DIR --role manual_manager --run-dir RUN_DIR
./wrsi agent packet PACKET_DIR --role editor --run-dir RUN_DIR
```

Use a new packet directory for updated evidence. Read dispatch.json and call its
specified collaboration method and arguments. Persistent roles continue through
followup_task after registration. For a replacement session, prepare a new packet
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
./wrsi h3 submit TASK_DIR
./wrsi h3 collect TASK_DIR
```

After a locked output, dispatch Scorer; its commands and judgment contract are in
benchmark.md. Then dispatch Reflector. Its experiment request has id, hypothesis,
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

## Transfer and common evaluation

```bash
./wrsi transfer prepare SOURCE_DIR
```

This prepares transfer/baseline and transfer/candidate. For each arm and each of
the two target inputs, dispatch a fresh Executor at seed 42. Candidate Executors
also write implementation.json with experience_sha256, applied:true, assessment
and evidence:[task-local paths]. Main submits the same frozen package for all seeds:

```bash
./wrsi eval submit EVALUATION_DIR --task-id TASK_ID
./wrsi eval collect EVALUATION_DIR
./wrsi eval finalize EVALUATION_DIR
./wrsi transfer finalize SOURCE_DIR
```

Dispatch an independent Scorer for every locked seed output before finalizing.
The same evaluation commands serve transfer, validation and full tests. Failure
to produce a valid final video is recorded with `wrsi output fail TASK --reason
TEXT`, then independently scored. Pending external/scoring failures remain
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

When all eleven sources and transfer outcomes are finalized, dispatch Manual
Manager, then publish its complete manuals and dispatch Editor:

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
`run history` returns passed decision bundles and aggregate validation outcomes.
`run report` includes source/gate status, tests and costs by phase. Long commands
run in named tmux sessions with logs under logs/; record the session and log path.
