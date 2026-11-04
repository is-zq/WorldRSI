# Physics-IQ cross-view learning

## Data and schedule

`protocol/split.json` binds the 33 learning and 33 validation scene names. Their
Solid Mechanics / Fluid Dynamics / Optics / Thermodynamics / Magnetism counts are
19/7/4/2/1 and 19/8/4/1/1. Every scene has left, center and right views. Public
inputs consist of the factual description and take-1 PNG from
`/data/physics-iq-verified-benchmark/physics-IQ-benchmark-verified/first-frames`.
Scorer uses both real takes and masks. Reflector receives the public first frame
and prompt, learning evidence, frozen history, and its learning view's take-1 GT
and corresponding scoring mask. Executor receives the public task and its
decision policy or experimental intervention. Every output covers five
seconds, 120 frames at 24 fps, 832×480.

Sort learning scenes, shuffle with seed 20261020, then assign left, center, right
cyclically: eleven scenes per learning view. This assignment stays fixed across
epochs. Epoch e shuffles sorted scene names with seed 20261020+e and partitions
them into three batches of eleven. All batch sources use one Agent/history snapshot.

Learning, transfer, validation and tests all use seeds 42–45. Each
input and experimental arm is modeled once; its frozen conditions serve all four
generation seeds. Initial validation establishes the reference on 99 inputs.
Initial evaluation and the first learning batch run concurrently; candidate
validation and promotion require the initial reference.

Evaluation mode is recorded per run. `full` uses all three views; `quick` uses
the first declared view (`perspective-left`) for validation and tests. Both use
seeds 42–45. Quick validation has 33 inputs and quick testing has 66 inputs.
Learning retains its fixed view assignment and the two reserved transfer views.

Run up to four epochs (twelve rounds). After at least one epoch, five consecutive
rejections end learning. Test the current Agent at every epoch boundary and once
at a partial stopping boundary.

## Learning and experience

Executor records P (physical), R (rendered) and M (conditioning) choices before
execution, implements the public task, checks artifacts and records tool updates.
Scorer only invokes the official scoring script and returns numerical artifacts;
it does not write analysis, observations or qualitative judgments. Each learning
attempt receives four seed scores and their arithmetic mean (including zero for
failed outputs). Reflector analyzes the scores, component evidence, outputs and
execution records itself to choose experiments and assess improvement against the
initial four-seed mean.

Reflector judges whether the gain is sufficient using the initial score, remaining
headroom and actual improvement. Low starting scores call for larger gains; high
starting scores allow smaller real gains. Once sufficient, Reflector closes with
the best assessed attempt, positive gain, rationale, implementation evidence and
one bundle of public triggers, decision instructions, scope and checks. Observations
retain remaining uncertainty. Stalled, blocked and initially saturated sources
record their respective reasons and complete with an empty bundle.

Each frozen bundle receives one transfer evaluation. Fresh Executors apply B and
B+E independently to the source's two reserved views. For each view, the mean of
the four paired official score differences must exceed the floating tolerance
1e-12. Candidate implementation evidence binds the experience hash. Both views
must pass. A passed group establishes the tested combination's effect.

The main session retains transfer details. Shared history receives passed bundles,
learning-view evidence and the passing receipt. Editor receives every closed
source's summary from all rounds, including failed transfers, stalled, blocked
and saturated sources and rejected candidates, with verified learning evidence
and transfer outcomes. A failed transfer is evidence about scope and uncertainty;
it does not prevent Editor from learning from the source.
Reflector's history is frozen at batch start. Executor uses a fresh task session.
Each learning source task has one Scorer, continued for its initial and subsequent
attempts. Each validation, test and transfer-arm evaluation has its own Scorer.
Retries stay within that scope; Scorer is not shared across the run. Scorer uses
`gpt-6.1-sol` with `high` reasoning effort. Reflector continues within an open
source; Editor and Manual Manager continue within a run.

Each learning attempt invokes the scoring script once for its four seeds.
Transfer, validation and test evaluations invoke it once for all planned inputs
and seeds. The script retains per-output components, scores and evidence;
infrastructure failures resume the batch and skip verified completed scores.
Reflector receives per-seed scores and their mean, not Scorer-authored analysis.

## Complete-Agent validation

After all eleven source closures and transfer outcomes, Manual Manager combines
checked tool operations into a complete manual tree. Editor synthesizes all
Reflector summaries into one self-contained decision experience at
`policy/skills/worldrsi-decision/SKILL.md`, with reusable implementation templates.
Preserve narrow but useful lessons as conditional rules, including dependencies,
checks, counterexamples and uncertainty. Do not discard a lesson merely because
it appears scene-specific or copy original source bundles into the policy or
templates. Ordinary Executors read this synthesis as their only learned decision
experience. Learning branches and transfer candidates apply their supplied
experimental instructions. Validation evaluates all three components
together on 33 validation scenes × three views × four seeds.

Compare official mean scores using tolerance 1e-12. A higher mean promotes the
candidate. Equal quality compares, in order, H3 generations, local GPU seconds,
model tokens and local CPU seconds per fixed input. The first different known
cost must decrease. Report categories separately. Acceptance updates the three
Agent components and the validation reference atomically.

## Tests and reporting

Full tests execute 198 inputs × four seeds at the actual current Agent. Report
the four seed means, overall mean, sample standard deviation
(ddof=1), categories and learning/validation partitions. The full set includes
both the learning and selection scenes; these partitions describe the exposure
of the evaluated Agent. Test results belong to the main-session report.

A quick evaluation can be extended to full coverage for the same snapshot and
boundary. The full plan references the quick plan's existing tasks and adds the
other two views. Existing submissions, conditioning and scores retain their
identities. A test extension adds 528 generations after all 264 quick generations
are complete. Reports retain both coverage levels and their source relationship.
When validation and testing overlap, validation reuses the corresponding test
outputs and scores for the same complete Agent snapshot, data revision, inputs,
intervention, seeds and scoring protocol. Initial validation is a projection of
the initial test. The test Scorer measures the complete test batch once. The
validation Scorer calls the batch script to verify and reuse its subset without
remeasurement; then finalize both reports. Retain test source bindings and
measured costs in the validation report. Different snapshots cannot share scores.

Evaluation identities bind phase, data, complete snapshot, intervention, seeds,
scoring protocol and test boundary. Recovery resumes the same submissions.
Validation caches retain their measured execution costs. Missing or corrupt final
outputs contribute zero within the fixed denominator; scoring infrastructure
errors remain pending. Costs distinguish learning, transfer, validation, test and
coordination work, with unknown counters recorded explicitly.
