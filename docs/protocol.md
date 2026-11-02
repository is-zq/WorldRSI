# Physics-IQ cross-view learning

## Data and schedule

`protocol/split.json` binds the 33 learning and 33 validation scene names. Their
Solid Mechanics / Fluid Dynamics / Optics / Thermodynamics / Magnetism counts are
19/7/4/2/1 and 19/8/4/1/1. Every scene has left, center and right views. Public
inputs consist of the factual description and take-1 PNG from
`/data/physics-iq-verified-benchmark/physics-IQ-benchmark-verified/first-frames`.
Both real takes and masks are private scoring inputs. Every output covers five
seconds, 120 frames at 24 fps, 832×480.

Sort learning scenes, shuffle with seed 20261020, then assign left, center, right
cyclically: eleven scenes per learning view. This assignment stays fixed across
epochs. Epoch e shuffles sorted scene names with seed 20261020+e and partitions
them into three batches of eleven. All batch sources use one Agent/history snapshot.

Learning explores with seed 42. Transfer, validation and tests use 42–45. Each
input and experimental arm is modeled once; its frozen conditions serve all four
generation seeds. Initial validation establishes the reference on 99 inputs.

Run up to four epochs (twelve rounds). After at least one epoch, five consecutive
rejections end learning. Test the current Agent at every epoch boundary and once
at a partial stopping boundary.

## Learning and experience

Executor records P (physical), R (rendered) and M (conditioning) choices before
execution, implements the public task, checks artifacts and records tool updates.
Scorer independently measures the locked output. Reflector uses its learning
view's evidence to choose further experiments and assess improvement against that
source's initial score.

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

The main session retains transfer details and failed bundles. Shared history and
Editor receive passed bundles, learning-view evidence and the passing receipt.
Reflector's history is frozen at batch start. Executor and Scorer use fresh task
sessions; Reflector continues within an open source; Editor and Manual Manager
continue within a run.

## Complete-Agent validation

After all eleven source closures and transfer outcomes, Manual Manager combines
checked tool operations into a complete manual tree. Editor integrates passed
bundles into a policy/template candidate. Validation evaluates all three components
together on 33 validation scenes × three views × four seeds.

Compare official mean scores using tolerance 1e-12. A higher mean promotes the
candidate. Equal quality compares, in order, H3 generations, local GPU seconds,
model tokens and local CPU seconds per fixed input. The first different known
cost must decrease. Report categories separately. Acceptance updates the three
Agent components and the validation reference atomically.

## Tests and reporting

Full tests independently execute 198 inputs × four seeds at the actual current
Agent. Report the four seed means, overall mean, sample standard deviation
(ddof=1), categories and learning/validation partitions. The full set includes
both the learning and selection scenes; these partitions describe the exposure
of the evaluated Agent. Test results belong to the main-session report.

Evaluation identities bind phase, data, complete snapshot, intervention, seeds,
scoring protocol and test boundary. Recovery resumes the same submissions.
Validation caches retain their measured execution costs. Missing or corrupt final
outputs contribute zero within the fixed denominator; scoring infrastructure
errors remain pending. Costs distinguish learning, transfer, validation, test and
coordination work, with unknown counters recorded explicitly.
