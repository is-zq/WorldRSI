"""Immutable schema-v2 handoff to the existing external H3 service."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import tempfile

from .common import ROOT, MODALITIES, MIN_GENERATION_FRAMES, MAX_GENERATION_FRAMES, digest, identifier, inside, lock, read_json, sha256, tree, write_json, immutable_json
from .media import probe, check_video, lock_output
from .records import cost


# Test packages use the dedicated lower-priority input queue.
# Generation settings still come from the task's frozen runtime.
INPUT_DIRECTORIES = ("inputs", "worldrsi-test-inputs")


def validate_package(package, task, provenance):
    package = Path(package)
    manifest = read_json(package / "request.json")
    if (set(manifest) - {"schema_version", "proxy_videos", "seed"}
            or not {"schema_version", "proxy_videos"} <= set(manifest) or manifest["schema_version"] != 2):
        raise ValueError("WorldRSI requires request.json schema_version=2")
    if "seed" in manifest and (type(manifest["seed"]) is not int or not 0 <= manifest["seed"] < 2 ** 63):
        raise ValueError("request.json seed must be a nonnegative integer below 2**63")
    references = manifest["proxy_videos"]
    if not isinstance(references, list) or not references:
        raise ValueError("At least one proxy is required")
    images = [p for p in (package / "first_frame.png", package / "first_frame.jpg") if p.is_file()]
    if len(images) != 1:
        raise ValueError("Exactly one first frame is required")
    inside(package, images[0].name)
    if sha256(images[0]) != task["input_hashes"]["first_frame"]:
        raise ValueError("First frame differs from frozen public input")
    timeline = {"frames": task["generation_frames"], "fps": 24, "start_seconds": 0}
    trajectory = read_json(provenance["trajectory"])
    states = trajectory.get("frames", [])
    if trajectory.get("fps") != 24 or len(states) != task["generation_frames"]:
        raise ValueError("Frozen trajectory does not cover the generation timeline")
    cameras = []
    for i, frame in enumerate(states):
        if frame.get("frame") != i or abs(frame.get("time", -1) - i / 24) > 1e-6:
            raise ValueError("Frozen trajectory has an inconsistent timeline")
        selected = [n for n in frame.get("nodes", {}).values() if n.get("current") and "camera_transform" in n]
        if len(selected) != 1:
            raise ValueError("Every frozen state must bind one current camera")
        cameras.append(selected[0])
    if read_json(provenance["camera"]) != cameras:
        raise ValueError("Camera provenance differs from the frozen physical trajectory")
    expected_binding = {"trajectory_sha256": sha256(provenance["trajectory"]),
                        "camera_sha256": sha256(provenance["camera"]), "timeline_sha256": digest(timeline)}
    prompt = inside(package, "prompt.txt").read_text(encoding="utf-8")
    required = {"<Picture 1>", *(f"<Video {i + 1}>" for i in range(len(references)))}
    labels = set(re.findall(r"<(?:Picture|Video|Audio)\s+\d+>", prompt))
    if labels != required:
        raise ValueError(f"Reference labels disagree: expected {sorted(required)}, got {sorted(labels)}")
    sections = ("subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music")
    if any(name not in prompt for name in sections):
        raise ValueError("Use the frozen official H3 Ref2VA prompt schema")
    shape, seen_paths, seen_modes, media, files = None, set(), set(), [], [images[0].name, "prompt.txt", "request.json"]
    if type(task["generation_frames"]) is not int or not MIN_GENERATION_FRAMES <= task["generation_frames"] <= MAX_GENERATION_FRAMES:
        raise ValueError("WorldRSI generation must cover 4–14 seconds")
    for i, entry in enumerate(references):
        if set(entry) != {"path", "modality"} or entry["modality"] not in MODALITIES:
            raise ValueError("Invalid modality/reference entry")
        path = inside(package, entry["path"], ".mp4")
        if entry["path"] in seen_paths or entry["modality"] in seen_modes:
            raise ValueError("Duplicate proxy path or modality")
        seen_paths.add(entry["path"])
        seen_modes.add(entry["modality"])
        decoded = probe(path)
        this_shape = (decoded["width"], decoded["height"], decoded["frames"])
        if shape is not None and this_shape != shape:
            raise ValueError("Proxy dimensions/frame counts must be identical")
        shape = this_shape
        check_video(path, task["generation_frames"], decoded["width"], decoded["height"])
        binding = provenance["videos"][entry["path"]]
        if any(binding.get(k) != v for k, v in expected_binding.items()) or binding.get("sha256") != sha256(path):
            raise ValueError("All references must bind the same frozen trajectory, camera and timeline")
        media.append({**entry, **decoded, "label": f"<Video {i + 1}>", "order": i + 1})
        files.append(entry["path"])
    return {"manifest": manifest, "media": media, "binding": expected_binding,
            "files": {p: sha256(inside(package, p)) for p in files},
            "references": [{"order": 0, "type": "image", "path": images[0].name, "label": "<Picture 1>"}] +
                          [{"order": i + 1, "type": "video", "label": f"<Video {i + 1}>", **r} for i, r in enumerate(references)]}


def prepare_package(task_dir, rendered, modalities, prompt_path):
    """Build a queue package directly from an independently checked renderer export."""
    from tools_runtime import validate_render
    task_dir = Path(task_dir)
    rendered = Path(rendered)
    if rendered.is_dir():
        rendered /= "render.json"
    saved = read_json(rendered)
    check = validate_render(rendered)
    artifacts = saved["artifacts"]
    task = read_json(task_dir / "task.json")["public"]
    if check["frames"] != task["generation_frames"]:
        raise ValueError("Render does not cover the fixed generation window")
    if not modalities or len(set(modalities)) != len(modalities) or any(m not in check["modes"] for m in modalities):
        raise ValueError("Choose a nonempty ordered subset of the actually rendered modalities")
    package = task_dir / "package"
    if package.exists():
        provenance_path = task_dir / "provenance.json"
        provenance = read_json(provenance_path)
        if provenance.get("render_manifest_sha256") != sha256(rendered) or sha256(package / "prompt.txt") != sha256(prompt_path):
            raise ValueError("Existing package belongs to a different render or prompt")
        if [r["modality"] for r in read_json(package / "request.json")["proxy_videos"]] != modalities:
            raise ValueError("Existing package has a different frozen modality order")
        return {"package": str(package.resolve()), "provenance": str(provenance_path.resolve()),
                "spec": validate_package(package, task, provenance)}
    package.mkdir(parents=True)
    first = Path(task["first_frame"])
    shutil.copy2(first, package / ("first_frame" + first.suffix.lower()))
    shutil.copy2(prompt_path, package / "prompt.txt")
    trajectory = read_json(artifacts["trajectory"])
    camera_path = package / "camera.json"
    cameras = [[n for n in f["nodes"].values() if n.get("current") and "camera_transform" in n][0] for f in trajectory["frames"]]
    write_json(camera_path, cameras)
    provenance = {"trajectory": artifacts["trajectory"], "camera": str(camera_path.resolve()), "videos": {},
                  "render_manifest": str(rendered.resolve()), "render_manifest_sha256": sha256(rendered)}
    references = []
    for modality in modalities:
        name = f"proxy_{modality}.mp4"
        videos = Path(artifacts["proxy_videos"])
        shutil.copy2(videos / f"{modality}.mp4", package / name)
        references.append({"path": name, "modality": modality})
        provenance["videos"][name] = {"sha256": sha256(package / name), "trajectory_sha256": sha256(provenance["trajectory"]),
                                       "camera_sha256": sha256(camera_path),
                                       "timeline_sha256": digest({"frames": task["generation_frames"], "fps": 24, "start_seconds": 0})}
    write_json(package / "request.json", {"schema_version": 2, "proxy_videos": references})
    write_json(task_dir / "provenance.json", provenance)
    spec = validate_package(package, task, provenance)
    return {"package": str(package.resolve()), "provenance": str((task_dir / "provenance.json").resolve()), "spec": spec}


def submit(task_dir, package, provenance, queue=None, root=ROOT):
    root, task_dir, package = Path(root), Path(task_dir), Path(package)
    context = read_json(task_dir / "task.json")
    runtime = (read_json(Path(context["snapshot_path"]) / "snapshot.json")["binding"]["runtime"]
               if context.get("snapshot_path") else read_json(root / "config/runtime.json"))["h3"]
    runtime = dict(runtime)
    if "generation_seed" in context:
        from .evaluation import generation_seed
        runtime["seed"] = generation_seed(context)
    queue = Path(queue or runtime["queue"]).resolve()
    task = read_json(task_dir / "task.json")["public"]
    with lock(task_dir / ".h3.lock"):
        intent_path = task_dir / "h3.submission.json"
        if intent_path.exists():
            # Recovery retains the durable package and its submitted identity.
            intent = read_json(intent_path)
            if intent["expected"]["seed"] != runtime["seed"]:
                raise ValueError("Existing H3 submission belongs to a different generation seed")
            queue = Path(intent["queue"])
            if not Path(intent["input_path"]).exists() and not Path(intent["output_path"]).exists():
                _publish(queue, intent, task_dir / "h3-inputs")
            return intent
        if (task_dir / "output.lock.json").exists():
            raise ValueError("Cannot generate after locking the output")
        spec = validate_package(package, task, provenance)
        if spec["manifest"].get("seed", runtime["seed"]) not in (42, 43, 44, 45):
            raise ValueError("Package seed must belong to the configured generation seed set")
        local = task_dir / "h3-inputs"
        if local.exists():
            # Only an unpublished pre-intent copy is replaceable.
            shutil.rmtree(local)
        local.mkdir(parents=True)
        for filename in spec["files"]:
            target = local / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(package / filename, target)
        if "generation_seed" in context:
            write_json(local / "request.json", {**spec["manifest"], "seed": runtime["seed"]})
            spec = validate_package(local, task, provenance)
        expected = {k: runtime[k] for k in ("width", "height", "fps", "steps", "seed", "model")}
        queue_id = "worldrsi-" + digest({"task_dir": str(task_dir.resolve()), "spec": spec,
                                        "expected": expected})[:40]
        input_directory = INPUT_DIRECTORIES[1] if context.get("phase") == "test" else INPUT_DIRECTORIES[0]
        intent = {"queue_id": queue_id, "task_id": task["task_id"], "queue": str(queue.resolve()),
                  "input_path": str(queue / input_directory / queue_id), "output_path": str(queue / "outputs" / queue_id),
                  "spec": spec, "expected": expected}
        immutable_json(intent_path, intent)
        _publish(queue, intent, local)
        return intent


def _publish(queue, intent, local):
    queue = Path(queue).resolve()
    qid = identifier(intent["queue_id"])
    destination = Path(intent["input_path"]).resolve()
    input_roots = tuple(queue / name for name in INPUT_DIRECTORIES)
    if destination.parent not in input_roots or destination.name != qid:
        raise ValueError("Submission input path must identify its task in an H3 input directory")
    with lock(queue / ".worldrsi-locks" / f"{qid}.lock"):
        # Both queues share submission identities and completion storage.
        for location in (queue / "outputs" / qid / "inputs", *(path / qid for path in input_roots)):
            if location.exists():
                for p, expected in intent["spec"]["files"].items():
                    if sha256(inside(location, p)) != expected:
                        raise ValueError("Queue ID exists with different inputs")
                return
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{qid}-", dir=destination.parent))
        try:
            for relative, expected in intent["spec"]["files"].items():
                source = inside(local, relative)
                if sha256(source) != expected:
                    raise ValueError("Durable input package changed")
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            os.rename(stage, destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)


def collect(task_dir):
    task_dir = Path(task_dir)
    with lock(task_dir / ".h3.lock"):
        intent = read_json(task_dir / "h3.submission.json")
        context = read_json(task_dir / "task.json")
        if "generation_seed" in context:
            from .evaluation import generation_seed
            if intent["expected"]["seed"] != generation_seed(context):
                raise ValueError("H3 submission belongs to a different test generation seed")
        source = Path(intent["output_path"])
        if not (source / "complete.json").is_file() or not (source / "video.mp4").is_file():
            return {"status": "pending", "queue_id": intent["queue_id"], "output_path": str(source)}
        done = read_json(source / "complete.json")
        if done.get("task_id") != intent["queue_id"]:
            raise ValueError("Completion task ID mismatch")
        cost(task_dir, intent["queue_id"], "h3", {"h3_generations": 1},
             {"queue_id": intent["queue_id"], "input_path": intent["input_path"], "output_path": intent["output_path"]})
        if done["inputs"].get("schema_version") != 2 or done["inputs"].get("references") != intent["spec"]["references"]:
            raise ValueError("Actual H3 references differ from submitted order/modalities")
        metadata = done["metadata"]
        if any(metadata.get(k) != v for k, v in intent["expected"].items()):
            raise ValueError("Actual H3 configuration differs from frozen runtime")
        task = read_json(task_dir / "task.json")["public"]
        refs = metadata.get("reference_videos", [])
        if len(refs) != len(intent["spec"]["media"]):
            raise ValueError("H3 omitted reference videos")
        for expected, actual in zip(intent["spec"]["media"], refs):
            for key in ("path", "modality", "order", "label", "width", "height", "fps", "frames"):
                if expected[key] != actual.get(key):
                    raise ValueError(f"H3 reference {key} mismatch")
            frames = task["generation_frames"]
            if actual.get("model_frames") != frames + (5 - frames) % 17:
                raise ValueError("H3 internal padding differs from contract")
        for relative, expected in intent["spec"]["files"].items():
            if sha256(inside(source / "inputs", relative)) != expected:
                raise ValueError("H3 input snapshot differs from frozen package")
        check_video(source / "video.mp4", task["generation_frames"])
        archive = task_dir / "h3-output"
        if not archive.exists():
            stage = task_dir / ".h3-output-copy"
            if stage.exists():
                shutil.rmtree(stage)
            shutil.copytree(source, stage)
            os.rename(stage, archive)
        if sha256(archive / "video.mp4") != sha256(source / "video.mp4"):
            raise ValueError("Archived raw output differs from completion")
        output = lock_output(task_dir, archive / "video.mp4", task)
        return {"status": "locked", "queue_id": intent["queue_id"], "output": output}
