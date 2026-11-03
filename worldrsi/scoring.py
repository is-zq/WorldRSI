"""Script-only numerical scoring and Reflector evidence sampling."""
from pathlib import Path
from .common import digest, immutable_json, lock, read_json, sha256
from .media import normalize, verify_locked
from .records import metered

@metered("reflector")
def evidence(task_dir, source, frames, reason, crop=None):
    """Reflector chooses arbitrary frame indices/crops; sampling itself is not a score."""
    directory = Path(task_dir)
    record = verify_locked(directory)
    if record["status"] != "valid":
        raise ValueError("Invalid outputs receive zero without sampling")
    task = record["task_config"]
    if not reason or not frames or any(type(f) is not int or f < 0 or f >= task["valid_frames"] for f in frames):
        raise ValueError("Sampling needs a reason and indices within the fixed valid window")
    source = Path(source).resolve()
    source_hash = sha256(source)
    selection = {"source": str(source), "source_sha256": source_hash, "frames": frames,
                 "reason": reason, "crop": crop, "valid_frames": task["valid_frames"]}
    key = digest(selection)
    dest = directory / "scoring/evidence" / key
    if (dest / "selection.json").exists():
        return read_json(dest / "selection.json")
    dest.mkdir(parents=True, exist_ok=True)
    normalized = dest / "normalized.mp4"
    if not normalized.exists():
        normalize(source, normalized, task["valid_frames"])
    import cv2
    cap = cv2.VideoCapture(str(normalized))
    images = []
    try:
        for frame in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok:
                raise ValueError("Cannot decode requested evidence frame")
            if crop is not None:
                x, y, width, height = crop
                if min(x, y) < 0 or min(width, height) <= 0 or x + width > image.shape[1] or y + height > image.shape[0]:
                    raise ValueError("Crop falls outside image")
                image = image[y:y + height, x:x + width]
            path = dest / f"frame-{frame:04d}.png"
            if not cv2.imwrite(str(path), image):
                raise ValueError("Cannot save evidence image")
            images.append({"frame": frame, "seconds": frame / 24, "path": str(path), "sha256": sha256(path)})
    finally:
        cap.release()
    return immutable_json(dest / "selection.json", {**selection, "images": images})



def batch(directory, definition):
    """One script invocation scores all outputs/seeds; retries resume saved scores."""
    from . import benchmark, evaluation
    definition = Path(definition).resolve()
    payload = read_json(definition)
    session = read_json(definition.parent / "session.json")
    spec = read_json(definition.parent / "dispatch.json")
    plan = evaluation.load(directory)
    if (payload.get("evaluation_dir") != plan["directory"] or payload.get("evaluation_key") != plan["key"]
            or spec["role"] != "scorer" or session["dispatch_sha256"] != sha256(definition.parent / "dispatch.json")
            or session.get("scope") != payload.get("scorer_scope")):
        raise ValueError("Batch scoring requires the registered task/evaluation Scorer packet")
    registry = read_json(Path(session["scope"]) / "sessions/scorer.json")
    if registry["session_id"] != session["session_id"]:
        raise ValueError("Use the active Scorer session for this task or evaluation")
    with lock(Path(session["scope"]) / "sessions/.scoring.lock"):
        tasks = [Path(d) for rows in plan["outputs"].values() for d in rows.values()]
        # Preflight the entire batch before starting any numerical work.
        for task in tasks:
            verify_locked(task)
        reused = set()
        if plan.get("reuse_test"):
            test = evaluation.load(plan["reuse_test"]["path"])
            reused = {Path(d) for rows in test["outputs"].values() for d in rows.values()}
            for task in set(tasks) & reused:
                evaluation.score(task)  # Missing test scores stay pending; never remeasure them here.
        for task in tasks:
            if (task / "scoring/score.json").exists():
                evaluation.score(task)
                continue
            private = benchmark.scorer_payload(task, plan["index_dir"])
            benchmark.measure_view(task, private)
            components = task / "scoring/physics_iq/components.json"
            judgment = {"session_id": session["session_id"], "components": str(components)}
            benchmark.finalize_scorer(task, private, judgment)
        return immutable_json(Path(plan["directory"]) / "scoring/batch.json",
                              {"evaluation_key": plan["key"], "scores": {
                                  str(t / "scoring/score.json"): sha256(t / "scoring/score.json") for t in tasks}})
