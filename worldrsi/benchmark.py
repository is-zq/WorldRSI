"""Physics-IQ index, historical result import, and independent report helpers.

The main Codex session owns execution. This module persists indexes and calculates
official metrics on locked outputs.
"""
from __future__ import annotations

import csv
import io
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import time

from .common import (FPS, HEIGHT, ROOT, WIDTH, canonical, digest, identifier, immutable_json,
                     jsonl, read_json, sha256, write_jsonl, write_text)
from .media import check_video, probe, verify_locked
from .metrics import BENCHMARK_ROOT, MASK_PARAMETERS, generate_mask, physics_modules, source_evidence
from .records import metered

RESULTS_ROOT = BENCHMARK_ROOT / "results/WoldRSI"
VIEWS = ("perspective-left", "perspective-center", "perspective-right")
LIST_KEYS = ("spatiotemporal_iou_v1", "spatiotemporal_iou_v2", "v1_mse", "v2_mse",
             "variance_spatiotemporal_iou", "variance_mse")
SCALAR_KEYS = ("spatial_iou_v1", "spatial_iou_v2", "weighted_spatial_iou_v1",
               "weighted_spatial_iou_v2", "variance_spatial", "variance_weighted_spatial")
RAW_KEYS = LIST_KEYS + SCALAR_KEYS
SCORER_VERSION = "physics-iq-components-v1"


def scorer_protocol():
    return {"version": SCORER_VERSION, "metric": "final_score_view",
            "sampling": "autonomous_evidence_driven", "valid_frames": 120}


def _scorer_config(config=None, *, root=ROOT, context=None):
    root = Path(root)
    config = config or read_json(root / "protocol/scorer.json")
    fixed = read_json(root / "config/runtime.json")["roles"]["scorer"]
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError("Benchmark Scorer configuration differs from the fixed role")
    # Scorer is a script-calling evaluator, configured independently of the
    # frozen Agent that produced the output. Keep that original Agent untouched.
    return config


def scorer_payload(task_dir, index_dir, *, root=ROOT):
    from .data import task
    context = read_json(Path(task_dir) / "task.json")
    public, private = task(index_dir, context["phase"], context["public"]["task_id"])
    if public != context["public"]:
        raise ValueError("Scorer input must match the indexed public task")
    return {"public": public, "private": private, "output": verify_locked(task_dir),
            "task_dir": str(Path(task_dir).resolve()), "scorer": _scorer_config(root=root, context=context),
            "scoring_protocol": scorer_protocol()}


def _source_media(path, *, decode=True):
    """Validate native benchmark GT/masks against the frozen media contract."""
    media = probe(path, decode=decode)
    if abs(media["fps"] - FPS) > 1e-6 or abs(media["duration"] - 5) > 0.002:
        raise ValueError(f"Benchmark source must be exactly 5 seconds at 24 fps: {path}")
    if decode and (media["frames"] != 120 or not media["uniform"] or abs(media["start_time"]) > 1e-4):
        raise ValueError(f"Benchmark source must decode to exactly 120 uniform frames from t=0: {path}")
    return media


def build_index(destination, source_root=BENCHMARK_ROOT, *, expected_scenes=66, decode=True):
    """Freeze take-1 first-frame/factual-description tasks and separate private GT.

    Default source validation decodes both takes and both real mask videos in
    full. ``decode=False`` creates a preparatory index that cannot be evaluated
    until ``validate_index`` has successfully completed the full decode.
    """
    import yaml
    destination, source_root = Path(destination).resolve(), Path(source_root).resolve()
    if (destination / "manifest.json").is_file():
        existing = read_json(destination / "manifest.json")
        if (existing["binding"]["implementation"] != source_evidence(source_root)
                or existing["binding"]["scene_count"] != expected_scenes):
            raise ValueError("Existing index binds different benchmark code or scene count; use a new revision directory")
        validate_index(destination, decode=decode and existing["status"] != "validated")
        return existing
    source_data = source_root / "physics-IQ-benchmark-verified"
    descriptions = source_root / "descriptions/data.yaml"
    scenes = yaml.safe_load(descriptions.read_text(encoding="utf-8"))["scenes"]
    if len(scenes) != expected_scenes:
        raise ValueError(f"Expected {expected_scenes} scenes; found {len(scenes)}")
    if len({row["scene"] for row in scenes}) != len(scenes):
        raise ValueError("Duplicate benchmark scenes")
    public, private, source_files, seen_ids = [], [], {}, set()
    cache_path = destination / "source-validation.json"
    cache = read_json(cache_path) if cache_path.exists() else None
    if cache and (digest(cache["binding"]) != cache["revision"]
                  or cache["binding"]["implementation"] != source_evidence(source_root)):
        raise ValueError("Source validation cache must bind the current measurement implementation")
    source_files[str(descriptions)] = sha256(descriptions)
    for scene in scenes:
        takes = {row["take"]: row for row in scene["takes"]}
        if set(takes) != {1, 2} or len(scene["takes"]) != 2:
            raise ValueError(f"Scene must contain exactly takes 1 and 2: {scene['scene']}")
        description = scene.get("description-factual") or scene.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Missing factual description: {scene['scene']}")
        for take in takes.values():
            if set(take) != {"take", "left", "center", "right"}:
                raise ValueError(f"Scene must contain exactly three views: {scene['scene']}")
        for view in VIEWS:
            perspective = view.removeprefix("perspective-")
            ids = [takes[take][perspective] for take in (1, 2)]
            for id_ in ids:
                if type(id_) is not int or id_ <= 0 or id_ in seen_ids:
                    raise ValueError("Benchmark IDs must be unique positive integers")
                seen_ids.add(id_)
            task_id = identifier(f"test_{ids[0]:04d}")
            scenario = f"trimmed-{scene['scene']}.mp4"
            first = source_data / "first-frames" / f"{ids[0]:04d}_first-frames_24FPS_{view}_take-1_{scenario[:-4]}.png"
            files = {"gt_v1": source_data / "split-videos/testing/24FPS" / f"{ids[0]:04d}_testing-videos_24FPS_{view}_take-1_{scenario}",
                     "gt_v2": source_data / "split-videos/testing/24FPS" / f"{ids[1]:04d}_testing-videos_24FPS_{view}_take-2_{scenario}",
                     "mask_v1": source_data / "video-masks/real/24FPS" / f"{ids[0]:04d}_video-masks_24FPS_{view}_take-1_{scenario}",
                     "mask_v2": source_data / "video-masks/real/24FPS" / f"{ids[1]:04d}_video-masks_24FPS_{view}_take-2_{scenario}"}
            hashes, media = {}, {}
            for name, path in {"first_frame": first, **files}.items():
                if not path.is_file():
                    raise FileNotFoundError(f"Fixed benchmark source is missing: {path}")
                hashes[name] = sha256(path)
                source_files[str(path)] = hashes[name]
                if name != "first_frame":
                    try:
                        checked = cache["binding"]["sources"].get(str(path)) if cache else None
                        saved = checked.get("media", {}) if checked else {}
                        reusable = (decode and checked and checked.get("sha256") == hashes[name]
                                    and saved.get("frames") == 120 and saved.get("uniform") is True
                                    and abs(saved.get("fps", 0) - FPS) < 1e-6
                                    and abs(saved.get("duration", 0) - 5) < 0.002
                                    and abs(saved.get("start_time", 1)) < 1e-4
                                    and saved.get("width", 0) > 0 and saved.get("height", 0) > 0)
                        media[name] = saved if reusable else _source_media(path, decode=decode)
                    except (ValueError, OSError, subprocess.CalledProcessError) as error:
                        raise ValueError(f"Fixed benchmark source failed validation: {path}: {error}") from error
                else:
                    from PIL import Image
                    with Image.open(path) as image:
                        image.load()
                        if image.width <= 0 or image.height <= 0:
                            raise ValueError(f"Invalid first-frames input: {path}")
            first_target = first
            prompt_target = destination / "inputs" / task_id / "prompt.txt"
            prompt = description.strip() + "\n"
            if prompt_target.exists() and prompt_target.read_text(encoding="utf-8") != prompt:
                raise ValueError(f"Existing factual prompt has changed: {prompt_target}")
            if not prompt_target.exists():
                write_text(prompt_target, prompt)
            public.append({"task_id": task_id, "category": scene["category"], "split": "test",
                           "first_frame": str(first_target), "prompt": str(prompt_target),
                           "input_hashes": {"first_frame": sha256(first_target), "prompt": sha256(prompt_target)},
                           "duration_seconds": 5, "valid_duration": 5, "generation_duration": 5,
                           "valid_frames": 120, "generation_frames": 120, "fps": FPS,
                           "width": WIDTH, "height": HEIGHT, "start_seconds": 0,
                           "spatial_transform": "center_crop_resize", "temporal_transform": "resample_preserve_speed"})
            private.append({"task_id": task_id, "scenario": scenario, "view": view,
                            "take1_id": ids[0], "take2_id": ids[1], "gt": str(files["gt_v1"]),
                            "gt_sha256": hashes["gt_v1"], **{key: str(value) for key, value in files.items()},
                            "sha256": hashes, "media": media,
                            "generated_video_name": f"{ids[0]:04d}_{view}_{scenario}",
                            "description_source": "description-factual" if scene.get("description-factual") else "description"})
    public.sort(key=lambda row: row["task_id"])
    private.sort(key=lambda row: row["task_id"])
    binding = {"schema_version": 1, "benchmark": "Physics-IQ Verified", "scene_count": expected_scenes,
               "task_count": expected_scenes * 3, "sources": source_files, "public": public,
               "private": private, "source_validation": "full_decode" if decode else "metadata_only",
               "source_validation_cache": cache["revision"] if cache else None,
               "first_frame_source": str(source_data / "first-frames"),
               "implementation": source_evidence(source_root)}
    revision = digest(binding)
    public = [{**row, "data_revision": revision} for row in public]
    for side, rows in (("public", public), ("private", private)):
        target = destination / side / "test.jsonl"
        if target.exists() and jsonl(target) != rows:
            raise ValueError(f"Frozen test index already exists with different content: {target}")
        if not target.exists():
            write_jsonl(target, rows)
    manifest = {"revision": revision, "directory": str(destination), "binding": binding,
                "public": str(destination / "public/test.jsonl"), "private": str(destination / "private/test.jsonl"),
                "status": "validated" if decode else "pending_full_decode"}
    return immutable_json(destination / "manifest.json", manifest)


def validate_index(index_dir, *, decode=True):
    """Verify every frozen source/input hash, optionally checking full decodability."""
    index_dir = Path(index_dir)
    manifest = read_json(index_dir / "manifest.json")
    _require_first_frames_manifest(manifest)
    if digest(manifest["binding"]) != manifest["revision"]:
        raise ValueError("Frozen benchmark manifest has changed")
    implementation = manifest["binding"]["implementation"]
    if implementation != source_evidence(implementation["root"]):
        raise ValueError("Scoring implementation must match the frozen benchmark revision")
    public, private = jsonl(index_dir / "public/test.jsonl"), jsonl(index_dir / "private/test.jsonl")
    expected_public = [{**row, "data_revision": manifest["revision"]} for row in manifest["binding"]["public"]]
    if public != expected_public or private != manifest["binding"]["private"]:
        raise ValueError("Public or private index changed after freezing")
    for path, expected in manifest["binding"]["sources"].items():
        if sha256(path) != expected:
            raise ValueError(f"Benchmark source hash changed: {path}")
    for task in public:
        for name, expected in task["input_hashes"].items():
            if sha256(task[name]) != expected:
                raise ValueError(f"Public benchmark input changed: {task[name]}")
    if decode:
        for task in private:
            for name in ("gt_v1", "gt_v2", "mask_v1", "mask_v2"):
                _source_media(task[name])
    result = {"revision": manifest["revision"], "task_count": len(public), "status": "passed",
              "decoded_all_sources": decode}
    if decode:
        immutable_json(index_dir / "validation.json", result)
    return result


def _require_first_frames_manifest(manifest):
    source_root = Path(manifest["binding"]["implementation"]["root"])
    required = str(source_root / "physics-IQ-benchmark-verified/first-frames")
    if manifest["binding"].get("first_frame_source") != required:
        raise ValueError("Index must bind the benchmark first-frames directory")


def _native_view(task, locked, directory, root):
    """Keep the official native-GT quarter-resolution/loading rules unchanged."""
    import numpy as np
    module, _ = physics_modules(root)
    directory.mkdir(parents=True, exist_ok=True)
    generation = directory / task["generated_video_name"]
    if not generation.exists():
        shutil.copyfile(locked["path"], generation)
    if sha256(generation) != locked["sha256"]:
        raise ValueError("Archived benchmark output differs from its lock")
    mask = generate_mask(generation, directory / "masks", benchmark_root=root)
    check_video(mask, 120)
    paths = module.ViewPaths(real_v1=task["gt_v1"], real_v2=task["gt_v2"], generated=str(generation),
                             mask_v1=task["mask_v1"], mask_v2=task["mask_v2"], mask_generated=str(mask))
    frames = module.load_view(paths, 0, 120, 120)
    if frames is None or frames.real_v1.shape[0] != 120:
        raise ValueError("The benchmark view does not contain the frozen 120 frames")
    raw = module.compute_view_metrics(frames)
    raw = {key: [float(x) for x in value] if isinstance(value, list) else float(value) for key, value in raw.items()}
    shape = list(frames.real_v1.shape)
    del frames
    arrays = directory / "raw_arrays.npz"
    np.savez_compressed(arrays, **{key: np.asarray(value) for key, value in raw.items()})
    evidence = {"generated": {"path": str(generation), "sha256": sha256(generation)},
                "generated_mask": {"path": str(mask), "sha256": sha256(mask)},
                "raw_arrays": {"path": str(arrays), "sha256": sha256(arrays)},
                "gt_hashes": task["sha256"], "measurement_shape": shape}
    return raw, evidence


def _view_context(task_dir, private):
    context = read_json(Path(task_dir) / "task.json")
    locked = verify_locked(task_dir)
    task = context["public"]
    if context.get("phase") not in ("learning", "transfer", "validation", "test") or task.get("split") != context["phase"]:
        raise ValueError("Scorer requires a registered benchmark phase")
    if private["task_id"] != task["task_id"] or locked["task_id"] != task["task_id"] or locked["task_config"] != task:
        raise ValueError("Private benchmark view and locked task do not match")
    for key, expected in (("valid_frames", 120), ("generation_frames", 120), ("valid_duration", 5),
                          ("generation_duration", 5), ("fps", FPS), ("width", WIDTH), ("height", HEIGHT), ("start_seconds", 0)):
        if task.get(key) != expected:
            raise ValueError("Benchmark Scorer requires the frozen five-second/120-frame task")
    return context, locked


def _private_record(value):
    if isinstance(value, (str, Path)):
        value = read_json(value)
    # The immutable Scorer input packet is a convenient existing CLI input;
    # it avoids making a second copy of the private row merely to call a tool.
    if isinstance(value, dict) and "private" in value:
        value = value["private"]
    if not isinstance(value, dict) or "task_id" not in value:
        raise ValueError("A private benchmark row or its Scorer input packet is required")
    return value


def _view_binding(context, locked, private, root):
    return {"task_id": private["task_id"], "snapshot_id": context["snapshot_id"],
            "data_revision": context["public"]["data_revision"], "output_lock_sha256": digest(locked),
            "private_sha256": digest(private), "implementation": source_evidence(root),
            "protocol_sha256": digest(scorer_protocol()), "task_context_sha256": digest(context)}


def _verify_evidence(record):
    for item in (record or {}).values():
        if isinstance(item, dict) and {"path", "sha256"} <= set(item):
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError("Test component evidence changed after measurement")


@metered("scorer")
def measure_view(task_dir, private, *, benchmark_root=BENCHMARK_ROOT):
    """Measure one view using its frozen inputs and the official loader."""
    started = time.monotonic()
    task_dir = Path(task_dir)
    private = _private_record(private)
    context, locked = _view_context(task_dir, private)
    for name in ("gt_v1", "gt_v2", "mask_v1", "mask_v2"):
        if sha256(private[name]) != private["sha256"][name]:
            raise ValueError(f"Frozen test source hash changed: {private[name]}")
    binding = _view_binding(context, locked, private, benchmark_root)
    directory = task_dir / "scoring/physics_iq"
    immutable_json(directory / "measurement_inputs.json", binding)
    path = directory / "components.json"
    if path.exists():
        result = read_json(path)
        if result["binding"] != binding:
            raise ValueError("Test components belong to a different frozen input")
        _verify_evidence(result.get("evidence"))
        return result
    raw, evidence = None, None
    if locked["status"] == "valid":
        check_video(locked["path"], 120)
        raw, evidence = _native_view(private, locked, directory, benchmark_root)
    result = {"schema_version": 1, "protocol": SCORER_VERSION, "binding": binding,
              "status": "measured" if raw is not None else "invalid_output",
              "raw": raw, "evidence": evidence, "mask_parameters": MASK_PARAMETERS,
              "preprocessing": "official_native_gt_quarter_resolution_fixed_timeline",
              "elapsed_seconds": time.monotonic() - started}
    return immutable_json(path, result)


def _evidence_references(task_dir, paths):
    references = []
    for name in paths:
        path = Path(name).resolve()
        if not path.is_relative_to((Path(task_dir) / "scoring").resolve()):
            raise ValueError("Independent benchmark Scorer evidence must be in this task's scoring directory")
        references.append({"path": str(path), "sha256": sha256(path)})
    if not references:
        raise ValueError("A test observation/tool call requires evidence")
    return references


def finalize_scorer(task_dir, private, judgment, scorer_config=None, *,
                    root=ROOT, benchmark_root=BENCHMARK_ROOT):
    """Persist official view scores and independent evidence."""
    task_dir = Path(task_dir)
    private = _private_record(private)
    context, locked = _view_context(task_dir, private)
    config = _scorer_config(scorer_config, root=root, context=context)
    session = judgment["session_id"]
    # Collaboration session identities may be canonical paths, not file names.
    # Preserve the exact identity accepted by sessions.register.
    if not isinstance(session, str) or not session.strip() or any(ord(c) < 32 for c in session):
        raise ValueError("A real, nonempty subagent session identifier is required")
    if judgment.get("scorer_config", config) != config:
        raise ValueError("Actual benchmark Scorer configuration differs from the frozen role")
    component_path = task_dir / "scoring/physics_iq/components.json"
    if Path(judgment["components"]).resolve() != component_path.resolve():
        raise ValueError("Benchmark Scorer must cite this task's official test measurement")
    components = read_json(component_path)
    if (components.get("protocol") != SCORER_VERSION
            or components["binding"] != _view_binding(context, locked, private, benchmark_root)
            or (components["status"] == "measured") != (locked["status"] == "valid")):
        raise ValueError("Test components do not bind this exact output, snapshot, and private view")
    _verify_evidence(components.get("evidence"))
    observations, tool_calls = [], []
    for observation in judgment.get("observations", []):
        if not observation.get("observation"):
            raise ValueError("Test observations require a description")
        observations.append({"observation": observation["observation"],
                             "evidence": _evidence_references(task_dir, observation.get("evidence", []))})
    for call in judgment.get("tool_calls", []):
        if not call.get("name"):
            raise ValueError("Scorer tool calls require a name")
        tool_calls.append({"name": call["name"], "evidence": _evidence_references(task_dir, call.get("evidence", []))})
    uncertainties = judgment.get("uncertainties", [])
    if not isinstance(uncertainties, list) or any(not isinstance(value, str) or not value for value in uncertainties):
        raise ValueError("Test uncertainties must be explicit descriptive strings")
    result = {"schema_version": 1, "protocol": SCORER_VERSION, "task_id": private["task_id"],
              "snapshot_id": context["snapshot_id"], "data_revision": context["public"]["data_revision"],
              "output_lock_sha256": digest(locked), "private_sha256": digest(private), "scorer": config,
              "session_id": session, "sampling": "official_components_only", "valid_output": locked["status"] == "valid",
              "zero_view_credit": locked["status"] != "valid", "observations": observations,
              "uncertainties": uncertainties, "tool_calls": tool_calls,
              "usage": judgment.get("usage", {"model_tokens": None}),
              "components": {"path": str(component_path.resolve()), "sha256": sha256(component_path), "values": components},
              "phase": context["phase"], "task_context_sha256": digest(context)}
    raw = components.get("raw") if locked["status"] == "valid" else None
    row = {"scenario": private["scenario"], **{f"{k}_{private['view']}": v for k, v in (raw or {}).items()}}
    summary, _, _ = _score_raw([row], {(private["scenario"], private["view"]): raw is not None}, benchmark_root)
    result["score"] = summary["final_score_view"]
    return immutable_json(task_dir / "scoring/score.json", result)


def _score_raw(rows, availability, root):
    """Use IQTable's official formulas and retain missing views with zero credit.

    Internal absence markers map to zero credit in the fixed-denominator mean.
    """
    import numpy as np
    import pandas as pd
    _, _, module = physics_modules(root, scoring=True)
    from physiq.calculate_iq_score import parse_list_of_floats
    engine_rows = []
    for row in rows:
        parsed = {"scenario": row["scenario"]}
        for view in VIEWS:
            for key in RAW_KEYS:
                value = row.get(f"{key}_{view}")
                parsed[f"{key}_{view}"] = (parse_list_of_floats(value) if key in LIST_KEYS else value) if value is not None else ([np.nan] * 120 if key in LIST_KEYS else np.nan)
        engine_rows.append(parsed)
    table = module.IQTable(pd.DataFrame(engine_rows))
    # This is the same clipping used by IQTable.get_output_dict. Perfect-MSE
    # ratios can legitimately be +inf before the official clipping step.
    scores = module.clip(table.compute_scores_scenario_by_view())
    selected = []
    for i, row in enumerate(rows):
        for j, view in enumerate(VIEWS):
            key = (row["scenario"], view)
            if key not in availability:
                continue
            selected.append(i * 3 + j)
            if not availability[key]:
                scores.iloc[i * 3 + j, :] = 0.0
    selected_scores = scores.iloc[selected]
    if not np.isfinite(selected_scores.to_numpy()).all():
        raise ValueError("Official score calculation produced non-finite values for an available output")
    summary = {"final_score_view": float(selected_scores["final_arithmetic"].mean())}
    for source, target in (("spatial_iou_v1", "spatial"), ("weighted_spatial_iou_v1", "weighted_spatial"),
                           ("spatiotemporal_iou_v1", "spatiotemporal"), ("v1_mse", "mse")):
        summary["score_" + target + "_view"] = float(np.clip(selected_scores[source + "_score"].to_numpy(), 0, 1).mean())
    official = table.get_output_dict() if len(selected) == len(rows) * 3 and all(availability.values()) else None
    if official is not None:
        official = {key: float(value) for key, value in official.items()}
        if not all(math.isfinite(value) for value in official.values()):
            raise ValueError("Official global scores are undefined for the real-take variance in this data")
    per_view = []
    for i, row in enumerate(rows):
        for j, view in enumerate(VIEWS):
            if (row["scenario"], view) not in availability:
                continue
            record = {"scenario": row["scenario"], "view": view, "available": availability[(row["scenario"], view)]}
            record.update({key: float(value) for key, value in scores.iloc[i * 3 + j].items()})
            per_view.append(record)
    return summary, official, per_view


