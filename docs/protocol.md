# Physics-IQ learning

## Data and schedule

`protocol/split.json` binds the 33 learning and 33 held-out test scene names (the frozen index retains
the historical `validation` group name). Their
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

Learning, promotion and tests all use seeds 42–45. Each
input and experimental arm is modeled once; its frozen conditions serve all four
generation seeds. Learning and promotion always use each scene's fixed learning view.

Evaluation mode is recorded per run and controls only held-out tests: `full`
uses all three views; `quick` uses `perspective-left`. Both use seeds 42–45.
Quick testing covers 33 inputs / 132 outputs; full testing covers 99 inputs /
396 outputs. Tests are requested explicitly for a frozen Harness and never gate
learning, promotion or stopping. There is no mandatory initial or epoch test.

Run up to four epochs (twelve rounds). After at least one epoch, five consecutive
rejections end learning.

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

Shared history receives all `sufficient_gain` bundles and their verified learning
evidence from completed rounds, including rounds whose candidate was rejected.
Editor receives every closed source's summary from all rounds, including stalled,
blocked and saturated sources and rejected candidates, with verified learning evidence.
Reflector's history is frozen at batch start. Executor uses a fresh task session.
Each learning source task has one Scorer, continued for its initial and subsequent
attempts. Each promotion and test evaluation has its own Scorer.
Retries stay within that scope; Scorer is not shared across the run. Scorer uses
`gpt-6.1-sol` with `high` reasoning effort. Reflector continues within an open
source; Editor and Manual Manager continue within a run.

Each learning attempt invokes the scoring script once for its four seeds.
Promotion and test evaluations invoke it once for all planned inputs
and seeds. The script retains per-output components, scores and evidence;
infrastructure failures resume the batch and skip verified completed scores.
Reflector receives per-seed scores and their mean, not Scorer-authored analysis.

## Complete-Agent promotion

After all eleven source closures, Manual Manager combines checked tool operations
into a complete manual tree. Editor synthesizes all Reflector summaries into one
self-contained decision experience at `policy/skills/worldrsi-decision/SKILL.md`
and reusable templates. Retain conditional lessons, dependencies, checks,
counterexamples and uncertainty. Ordinary Executors read this synthesis as their
only learned decision experience; learning branches apply experimental instructions.

Evaluate the complete candidate (policy, templates and manuals) on this round's
11 fixed learning inputs × four seeds. Fresh Executors receive ordinary public
inputs and the candidate Harness, without learning branches, GT or Reflector
instructions. The baseline reuses the same round's initial outputs and scores,
bound to the exact round-start Harness, data, inputs, seeds and scoring protocol.
Its own Scorer verifies all 44 existing measurements without remeasurement.
Candidate scoring also uses one batch invocation for all 44 outputs. Reports
retain original costs for reused results and distinguish incremental costs.

Compare the arithmetic means over the same 44 outputs using tolerance 1e-12.
Never compare scores from different rounds' input sets. Higher quality promotes
all three components atomically. Equal quality compares H3 generations, local GPU
seconds, model tokens and local CPU seconds per input, in that order. The first
different known cost must decrease; an unknown required cost prevents promotion.
A measured score decline beyond the tolerance returns the candidate to Editor
for re-synthesis, up to three additional candidates (four evaluations total).
Editor receives per-input, per-seed differences, component and execution evidence,
all prior attempts and learning summaries. It analyzes causes and uncertainty,
revises the previous policy/templates and records its rationale; manuals remain
fixed. Each revision uses fresh ordinary Executors and its own Scorer on all 44
outputs, against the same verified baseline. Intermediate failures keep the round
open and do not change the rejection streak. Only final acceptance or rejection
settles the round; exhausted retries retain the round-start Harness. Equal-score
and cost-based failures retain the existing final decision rules. Pending outputs
or infrastructure failures resume the same attempt without consuming a revision.
Keep every candidate, assessment, feedback and report immutable with bound hashes.
Repeated publication of the same snapshot resumes its existing attempt.

Shared history includes learning summaries and promotion outcomes, including
rejected candidates. Held-out test scores never enter role feedback or promotion.

## Held-out tests and reporting

Tests evaluate only the 33 formerly designated validation scenes. Select a round's
frozen candidate (including rejected candidates and earlier attempts), its active Harness after the
promotion decision, or the initial Harness with round 0. Reports retain the run,
round, target and exact complete snapshot, independent of later learning.
Report four seed means, overall mean, sample standard deviation (ddof=1),
categories and measured/incremental costs. Tests belong to the main-session report.

`eval extend` completes a quick held-out test with the other two views, adding
264 outputs and retaining the original 132 output identities. Both reports can
be registered even after the active Harness advances. Historical reports using
the former validation or 66-scene test scope remain readable with their original
identities; their scores do not become promotion evidence.

Evaluation identities bind phase, data, complete snapshot, intervention, seeds,
scoring protocol and round/target. Resume the same submissions after interruption.
Invalid final outputs contribute zero within the fixed denominator; unfinished
outputs and scoring infrastructure failures remain pending. Report learning,
promotion, test and coordination costs separately, preserving unknown counters.
