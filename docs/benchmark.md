# Official scoring

The adapter imports the installed official Physics-IQ code, binds its source
hashes, and measures each generated view against both real takes and masks.
It preserves the official native-GT quarter-resolution loading and list parsing.
`IQTable.compute_scores_scenario_by_view()` supplies the four clipped component
scores and their arithmetic mean (`final_score_view`). All four experiment phases
use this same numerical interpretation.

Source preparation records first-frame paths, factual descriptions, media hashes
and completed source decoding. Scorer receives one public input, its locked output
and its private scoring row. Measurements save raw arrays, generated masks,
component values and evidence hashes in the task's `scoring/physics_iq/` directory.
Descriptive observations and uncertainty accompany the independent score.

Preparation can reuse complete media validation from
`benchmark/source-validation.json`. The cache binds the official implementation,
source paths, file hashes and decoded media metadata. Each current source is
hashed before reuse; sources with changed hashes receive fresh media validation.

```bash
./wrsi score measure TASK --definition SCORER_PACKET/input.json
./wrsi score sample TASK --source VIDEO --frames 0 35 119 --reason 'Inspect contact and final position'
./wrsi score finalize TASK --definition SCORER_PACKET/input.json --judgment JUDGMENT.json
```

Judgment fields are `session_id`, `components` (the measured JSON path),
`observations` (`observation` and evidence paths), `uncertainties`, and optional
`tool_calls` and `usage`. Evidence belongs in this task's scoring directory.
Scores are computed from components and bind the complete task context and output.

`wrsi eval finalize` requires all planned outputs to have final independent scores.
Failed final outputs receive zero. Each seed report averages over the same fixed
inputs; the report then averages the four seed results and computes their sample
standard deviation. It retains per-input scores, category means, partition means,
costs and score-file hashes. File helpers preserve pending measurement errors for
recovery. A completed report is immutable.
