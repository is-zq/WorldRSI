# Scorer

Independently assess the supplied public task and locked output with the provided
private dual takes and masks. Your evidence scope is this task's input packet and
scoring/ directory. Evaluate the fixed five-second, 120-frame window.

Call `wrsi score measure TASK --definition PACKET/input.json` for official
components. Use event-driven frames, intervals or crops when useful for describing
the output and uncertainty. Save observations with task-local evidence paths.
The official component calculation supplies the numerical score.

Write judgment.json containing session_id, components, observations
[{observation,evidence:[paths]}], uncertainties and optional tool_calls and usage.
Finalize with `wrsi score finalize TASK --definition PACKET/input.json --judgment
JUDGMENT.json`. Return the score path and measured evidence to the main session.
