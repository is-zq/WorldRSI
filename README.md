# WorldRSI

WorldRSI learns physics-video decisions from Physics-IQ-Verified. Each epoch learns
33 scenes in three rounds. A scene's experience bundle is tested on its other two
views before Editor integrates it. Complete-Agent validation selects policy,
templates and manuals together.

All stages use the official dual-take Benchmark score and take-1 images from
`/data/physics-iq-verified-benchmark/physics-IQ-benchmark-verified/first-frames`.
The scene partition is frozen in `protocol/split.json`.

Runs select `quick` (first view) or `full` (three views) for validation and tests,
with four seeds in either mode. `wrsi eval extend` completes a quick evaluation
using its existing tasks and adding the remaining views.

- [Protocol](docs/protocol.md): views, roles, sufficient gain and promotion.
- [Operation](docs/operation.md): commands, role packets and recovery.
- [Tools](docs/tools.md): simulation, rendering and condition interfaces.
- [Scoring](docs/benchmark.md): measurements, reports and evidence.

Use the existing `API-World` Conda environment:

```bash
./wrsi doctor
./wrsi data prepare
./wrsi snapshot freeze
conda run -n API-World python -m unittest discover -s tests -v
```

The main Codex session directly dispatches roles. `./wrsi` builds their packets,
binds their sessions, persists experiments, submits frozen H3 packages and reports
results. Generated data lives in `prepared/`, `runs/`, `snapshots/` and `logs/`.
