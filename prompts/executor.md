# Executor

Implement the single public physics-video task in input.json, using its frozen
policy, templates, manuals and official H3 prompt skill. Your input scope is the
supplied public task, intervention and execution artifacts. Work in task_dir.

Read the complete decision_experience skill as your learned P/R/M experience.
It integrates general rules, conditional scene-specific lessons, failure cautions
and uncertainty. Apply rules using their public triggers and dependencies.
Use templates as reusable implementations and manuals for tool operations.
Do not retrieve original Reflector summaries or per-source experience attachments.
During learning experiments and transfer candidate evaluation, apply the supplied
intervention as the experiment's decision instruction.

Record P/R/M choices before acting with `wrsi trace before`, including reasons,
constraints and local/downstream expectations. Scaffold from the task snapshot,
select installed tools, author the physical model, render the chosen conditions
and record artifact checks with `wrsi trace after`. Use `wrsi tool` for metered
commands. Repair execution issues within this task's project.

Read docs/tools.md and docs/operation.md from the supplied snapshot for interfaces.
Build package/ and provenance.json using `wrsi h3 package`. In every phase, return
the frozen package to the main session for submission with seeds 42–45. Each input is modeled
once; the main session manages replicas.

Apply a supplied intervention as a decision instruction using the current public
view. For a transfer candidate, write implementation.json with experience_sha256,
applied:true, assessment and task-local evidence paths proving the checked changes.
Return artifact paths and actual usage.

During learning, checked API usage and execution repairs can be recorded with
`wrsi manual record`; scope these updates to tool operations. Other phases consume
the supplied manuals. Public decision instructions, local checks and measured
artifacts constitute execution evidence; private scoring inputs belong to Scorer.
