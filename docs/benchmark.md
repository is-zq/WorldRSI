# Official scoring

The adapter imports the installed official Physics-IQ code, binds its source
hashes, and measures each generated view against both real takes and masks.
It preserves the official native-GT quarter-resolution loading and list parsing.
`IQTable.compute_scores_scenario_by_view()` supplies the four clipped component
scores and their arithmetic mean (`final_score_view`). All experiment phases
use this same numerical interpretation.

Source preparation records first-frame paths, factual descriptions, media hashes
and completed source decoding. Each learning source task has its own Scorer;
each promotion and test evaluation has its own Scorer. The configured
subagent is gpt-6.1-sol/high. Scorer only invokes scripts and returns artifact paths.
It supplies no analysis or authored judgment. Reflector analyzes learning scores,
generated outputs, components and execution evidence itself.

Measurements save raw arrays, generated masks, numerical values and evidence
hashes in each output's `scoring/physics_iq/` directory. Learning uses the mean of
all four seeds (42–45), including failed-output zeros. Initial and subsequent
attempts use the same seed set and frozen conditioning within each attempt.

Preparation can reuse complete media validation from
`benchmark/source-validation.json`. The cache binds the official implementation,
source paths, file hashes and decoded media metadata. Each current source is
hashed before reuse; sources with changed hashes receive fresh media validation.

```bash
./wrsi score batch EVALUATION_DIR --definition SCORER_PACKET/input.json
```

Invoke once after all outputs are locked: one learning input × four seeds, or
all evaluation inputs × four seeds. The script measures and finalizes each output
without a Scorer-authored judgment. It binds the registered task/evaluation session.
Retries retain verified scores and resume unfinished measurements. Promotion
baselines verify the round's initial score files without another measurement.

Reflector receives the public task, source-local components, scores, outputs,
and the learning view's take-1 GT and matching scoring mask with verified index
hashes. It can inspect these references and request evidence
frames with `wrsi score sample TASK --source VIDEO --frames 0 35 119 --reason TEXT`.
This samples evidence without changing numerical scores.

`wrsi eval finalize` requires all planned outputs to have final independent scores.
Failed final outputs receive zero. Each seed report averages over the same fixed
inputs; the report then averages the four seed results and computes their sample
standard deviation. It retains per-input scores, category means, partition means,
costs and score-file hashes. File helpers preserve pending measurement errors for
recovery. A completed report is immutable.
