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
selects the first view for validation and tests. Initial testing can run alongside
initial validation. `wrsi eval status` identifies pending modeling and generations.
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

Scorer owns private scoring inputs. Reflector uses learning-view feedback; Editor
and shared history use passed bundles and aggregate validation summaries. Transfer
detail and complete-test reports belong to the main session. Resume immutable
packets, packages and submissions from their saved identities. Report concrete
pending artifacts and account for actual role and tool costs.
