# WorldRSI

Use `docs/protocol.md` for the experiment and `docs/operation.md` for execution.
Use the existing `API-World` environment and `./wrsi` file helpers. Source,
configuration, protocols and tests belong in Git; generated artifacts belong in
the ignored project directories under `/data`.

Use the prepared Physics-IQ inputs directly. The first-frame source is
`/data/physics-iq-verified-benchmark/physics-IQ-benchmark-verified/first-frames`.
Data changes belong to the user. Task indexing and scoring bind the source files.

The main Codex session dispatches Executor, Reflector, Scorer, Editor and Manual
Manager using each packet's `dispatch.json`. Helpers persist files and evidence.
Eleven learning sources share a frozen batch-start Agent and decision history.
Executor receives one public input. Each learning task and each validation/test
evaluation has its own Scorer (gpt-6.1-sol/high), which only calls scoring scripts.
Learning uses the mean of four seeds; Reflector performs the analysis. Evaluations
score all planned inputs and seeds in one invocation, and overlapping validation
reuses matching test scores. Reflector receives its public first frame, prompt,
detailed learning evidence, and that view's take-1 GT and corresponding scoring
mask, and judges improvement relative to the initial score. Editor synthesizes
all source summaries and transfer outcomes into one self-contained decision skill,
retaining conditional lessons and uncertainty. Ordinary Executors read that
synthesis; learning and transfer retain experimental instructions. Manual
Manager consolidates checked learning-stage tool operations. Whole-Agent validation
activates policy, templates and manuals together.

Learning, transfer and validation use the regular H3 input queue; complete tests
use `worldrsi-test-inputs`. Keep long-running commands in named tmux sessions with
logs under the project. Preserve registered submissions and frozen artifacts on
recovery.

Choose tests according to the affected behavior. Disposable fixtures and fixes
for change-related failures are authorized. Use the full unittest suite for broad
changes. Skill edits use `skill-creator` and the principles in
`/data/.codex/AGENTS_AND_SKILLS_PRINCIPLES.md`.
