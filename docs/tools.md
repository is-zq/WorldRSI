# WorldRSI tool interfaces

Use `./wrsi tools` in the existing `API-World` environment. `tools_runtime`
provides execution, state capture/replay, condition encoding and artifact checks.
Executor selects the simulator and authors the task model using the supplied interfaces.
Installed simulator discovery reports names, executable paths, availability and
devices. A backend without a general executable exposes its source directory
and requires an authored driver command. It does not recommend a simulator for a scene or material.

```bash
./wrsi tools --help
./wrsi tools simulators --task-dir TASK_DIR
./wrsi tools scaffold TASK_DIR/project --task-dir TASK_DIR
```

The initial project contains only the runtime files from `tools_runtime/assets/godot/`
and a black background semantic code. With `--task-dir`, these runtime assets also
come from the verified task snapshot. It has no scene.gd, camera, geometry,
lighting, solver driver, physical model, class catalog or model parameters.
Executor authors these in its task workspace. The selected snapshot's learned
files are overlaid when present. The repository's initial learned template tree
has no implementations and an empty profile catalog.

## Task project contract

Implement `scene.gd` extending `Node3D` with these callbacks:

| Callback | Interface |
| --- | --- |
| `build(action: Dictionary)` | Construct the task-authored scene and camera |
| `step(delta: float)` | Advance task-authored state during native simulation |
| `export_state() -> Dictionary` | Return serializable custom state |
| `restore_state(state: Dictionary)` | Restore custom state and any dynamic topology |

The runtime records node transforms and camera state, calls these callbacks,
and freezes a 24 fps trajectory. Native simulation invokes `step`; replay does
not advance simulation. Native dynamic bodies require enabled collision shapes.
Mesh metadata uses `semantic_id` from the project palette and stable `instance_id`
for object identity. Project state and referenced resources must stay inside the
project; resource paths use `res://`.

`palette.json` maps canonical nonnegative integer-string IDs to
`{"label": STRING, "color": [R,G,B]}`. Labels and integer RGB triples must be
unique. ID `0` is the reserved black background; the Agent defines all other
classes and codes. The renderer encodes these values without assigning categories.

## Simulation and replay

A request specifies `project`, `frames`, and an `action` JSON path. Optional
`state` restores an ending checkpoint. `simulation.backend` in the action must
be chosen explicitly from the installed names; there is no default backend.
Godot accepts integer `substeps` in 1–64 (one step if omitted). An external
backend additionally needs a project-relative `driver`, its execution `command`
(argv list, no shell), and a recorded selection `reason`.

The external driver is called once with `--request REQUEST_JSON --output DIRECTORY`.
The request supplies backend, project, fps, frames, action, initial state, output
and cache resource paths. It writes `trajectory.json` with `fps: 24` and the
requested frames. Every frame contains `frame`, `time = frame / 24`, `custom`,
and optional node poses. All numerical states must be finite.

Optional `simulation.contract: 1` binds checkpoints and solver reports. Ending
`custom.physics` contains backend, profile (if selected), model_sha256,
`checkpoint` as a project-local resource path, and checkpoint_sha256.
`solver_report.json` contains matching backend/profile/model_sha256 plus version
and device. Continuation must retain the physical model and valid checkpoint bytes.

```bash
./wrsi tools simulate --request SIMULATION_REQUEST.json --output TASK_DIR/simulation
./wrsi tools render --request RENDER_REQUEST.json --output TASK_DIR/render
./wrsi tools validate-render TASK_DIR/render
./wrsi tools restore --project PROJECT --state STATE.json --action ACTION.json --output OUTPUT --modes MODE
```

Render requests require an explicit ordered `modes` list. With `trajectory`,
render replays the frozen states; without it, render first performs the explicitly
configured simulation. Use `recapture: true` to rebuild captured Godot nodes from
an external driver's custom states or an edited rendering. Simulation and render
copy task files unchanged into a new output directory for every attempt.

Tool requests accept 1–336 frames. Benchmark execution uses 120 frames at 24 fps.
Sorted distinct `frame_indices` can select inspection frames from a frozen
trajectory. Packaging and scoring retain the complete benchmark timeline.

## Condition interfaces

All selected conditions share one trajectory, camera, timeline, 832×480 resolution,
24 fps and frame count. The five interface names define data, not selection advice:

| Name | Data definition |
| --- | --- |
| `rgb` | RGB from the authored scene |
| `semantic` | Task-authored categorical RGB codes |
| `depth` | Camera-Z depth in scene units; fixed clip-wide scale |
| `normal` | World-space XYZ normals mapped from [-1,1] to RGB [0,255] |
| `instance` | Stable per-object RGB codes |

Depth uses near-bright/far-dark encoding with black background. Normals and
instance codes also reserve black for background. Internal native buffers and a
semantic anchor support artifact checks; only the explicitly selected ordered
references are submitted to H3.

Optional `proxy.json` has `schema_version: 1` and `depth`/`instance` mappings.
Depth supports `range: "clip"` (one scale across valid frames) or `range: "fixed"`
with finite `0 <= near < far`, plus positive finite `gamma`. Encoding defaults
are clip range and gamma 1. Instance encoding accepts integer
`1 <= min_channel < max_channel <= 255` and string `salt`; codec defaults are
64, 255 and the empty string. These are serialization settings, not a preferred
conditioning policy. Explicit changes reassign the complete instance reservation
set; colors stay stable under those settings.

Render output includes the project, trajectory, ending state, native depth/normal/
instance arrays, camera metadata, condition videos, a manifest and checks.
Validation checks hashes, finite states, complete video decoding, common media
specifications, camera/timeline correspondence, and stable identity codes. These
checks establish interface integrity; Scorer assesses task fulfillment independently.

## Template configuration

The complete learned tree has `godot/` and `physics/` subdirectories. Editor may
add reusable implementations supported by learning evidence and submit the complete
candidate through validation. Initial `physics/profiles.json` is
`{"schema_version": 1, "profiles": {}}`; there are no supplied profiles.

Each learned profile declares backend, driver, scope and model defaults. Optional
`action`, `parameters`, `files`, `labels`, `driver_profile` and `solvers` describe
its authored implementation. Driver and dependency paths are relative to physics/.
Use `--profile NAME` only for a profile present in the selected snapshot.

Profile scaffolding copies physics support files, merges profile model/action
values with authored `model.json`/`action.json`, and adds missing semantic labels.
Nested mappings merge; explicit values, including lists and nulls, win. Physics
files take precedence over same-named Godot overlay files. Selected backend/profile
must remain consistent. Catalog discovery uses the selected snapshot; an absent catalog yields an empty set.

Python entries are `scaffold(destination, profile=None, *, templates=None, runtime_assets=None)`, `simulate`,
`render`, `validate_render`, `restore(..., modes=..., action=...)` and `prepare_first_frame`.
For standalone work `--templates DIRECTORY` selects a complete learned tree;
experiment work uses `--task-dir` to bind the frozen Agent snapshot.

## First-frame and execution services

`wrsi tools first-frame SOURCE --output DIRECTORY` normalizes the public frame
using at most 5% center crop and 832×480 resize. `--estimate-depth` explicitly
requests VGGT-Omega inference. Its depth scale is arbitrary, and confidence values
express relative confidence in the predictions. Godot depth is camera-Z in scene
units. Interpret depth using the corresponding coordinate and scaling conventions.

Godot capture uses `/data/bin/godot`, Xvfb and `/data/bin/vglrun -d egl`; rendering
requires hardware graphics. Installed solver entrypoints use their existing
`/data/simulators/bin/` environments. Check `nvidia-smi -L` and free memory before
accelerator work. Use named tmux sessions and task-local logs for long jobs.

Synthetic interface tests live in `tests/`. Run the full suite after runtime or
protocol changes; set `WORLDRSI_GPU_TESTS=1` to include hardware rendering checks.
