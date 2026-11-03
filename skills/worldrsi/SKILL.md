---
name: worldrsi
description: Operate or resume the WorldRSI Physics-IQ cross-view learning experiment through direct role dispatch and durable file commands.
---

# WorldRSI operation

Read [protocol](../../docs/protocol.md) for experimental decisions and
[operation](../../docs/operation.md) for commands and recovery. Use the existing
API-World environment and `./wrsi`. The main session dispatches the five roles
using each packet's `dispatch.json`, registering real session IDs and usage.

Begin with the frozen data index, empty initial Agent and initial validation.
Choose quick or full evaluation mode for the run. Both use four seeds; quick
selects the first view for validation and tests. Learning also uses seeds 42–45
and compares their arithmetic mean. Initial validation reuses the corresponding
initial-test outputs and scores. Score the complete test batch once; validation
uses its own Scorer to verify and reuse that subset before reports are finalized. `wrsi eval status` identifies pending generations.
Dispatch all eleven source Executors from the batch snapshot, then advance each
source as evidence arrives. Reflector determines sufficient gain from the initial
score and tested improvement, freezes its bundle, and completes its source session.
The two reserved views then compare baseline and bundle using four paired seeds.
Both view means must improve before Editor receives the bundle.

After every source and gate finishes, dispatch Manual Manager, publish its manuals,
then dispatch Editor and validate the complete candidate once. Use the deterministic
promotion helper. Complete each pending boundary test before the next learning round.
Use `wrsi eval extend` to complete quick evaluations with the other two views,
retaining the original snapshot and existing generations.

Each learning source task has one Scorer across its attempts; each validation,
test and transfer-arm evaluation has its own Scorer. Scorer uses gpt-6.1-sol/high
and only invokes the batch script, returning numerical artifacts. Reflector
analyzes learning scores and evidence itself. Learning batches cover one input ×
four seeds; other batches cover all planned inputs and seeds. Overlapping
validation reuses matching test scores as described in the operation guide. Editor
and shared history use passed bundles and aggregate validation summaries. Transfer
detail and complete-test reports belong to the main session. Resume immutable
packets, packages and submissions from their saved identities. Report concrete
pending artifacts and account for actual role and tool costs.
