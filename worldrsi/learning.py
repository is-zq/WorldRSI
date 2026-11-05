"""Adaptive source experiments, Reflector closure, and verified learning experience."""
from pathlib import Path

from .common import digest, identifier, immutable_json, lock, read_json, sha256, write_json
from . import data, evaluation

COST_SCOPES = {"construction", "simulation_render", "generation", "execution"}


def evidence(source, paths):
    source = Path(source).resolve()
    if not isinstance(paths, list) or not paths:
        raise ValueError("Experience requires source-local implementation evidence")
    result = []
    for name in paths:
        p = Path(name)
        if not p.is_absolute():
            local = source / p
            legacy = p.resolve()
            p = legacy if not local.exists() and legacy.is_relative_to(source) else local
        p = p.resolve()
        if not p.is_relative_to(source):
            raise ValueError("Evidence must belong to its learning source")
        result.append({"path": str(p), "sha256": sha256(p)})
    return result


def attempt_score(directory):
    """Learning decisions use the fixed four-seed mean, never a best seed."""
    directory = Path(directory)
    report = evaluation.report(directory)
    if report["phase"] != "learning" or len(report["task_ids"]) != 1 or report["seeds"] != evaluation.SEEDS:
        raise ValueError("Learning feedback requires one input and all four seeds")
    tid = report["task_ids"][0]
    per_seed = {str(seed): report["scores"][str(seed)][tid] for seed in evaluation.SEEDS}
    if abs(sum(per_seed.values()) / 4 - report["mean"]) > evaluation.EPSILON:
        raise ValueError("Learning score must equal the mean of all four seeds")
    return {"score": report["mean"], "per_seed": per_seed, "seeds": report["seeds"],
            "outputs": {seed: rows[tid] for seed, rows in report["outputs"].items()},
            "evidence": report["evidence"],
            "report": {"path": str(directory / "report.json"), "sha256": sha256(directory / "report.json")}}


def progress(source_dir):
    source_dir = Path(source_dir)
    source = read_json(source_dir / "source.json")
    results = {}
    for name, path in [("initial", source_dir / "initial"), *[(b["id"], Path(b["directory"])) for b in source["branches"]]]:
        if (path / "evaluation.json").exists() and (path / "report.json").exists():
            results[name] = attempt_score(path)
    best = max(results, key=lambda k: results[k]["score"]) if results else None
    return {"initial_score": results.get("initial", {}).get("score"), "best_attempt": best,
            "best_score": results[best]["score"] if best else None, "attempts": results}


def branch(source_dir, request):
    source_dir = Path(source_dir).resolve()
    with lock(source_dir / ".source.lock"):
        source = read_json(source_dir / "source.json")
        if (source_dir / "experience.json").exists():
            raise ValueError("Experiments require an open learning source")
        bid = identifier(request["id"])
        if not request.get("hypothesis") or not request.get("executor_instruction") or not request.get("prediction"):
            raise ValueError("Experiment requires hypothesis, executable decision instruction, and prediction")
        parent = request.get("parent_attempt", "initial")
        p = progress(source_dir)
        if parent not in p["attempts"]:
            raise ValueError("Experiment parent requires a completed independent score")
        d = source_dir / "branches" / bid
        immutable_json(d / "hypothesis.json", request)
        public, _ = data.task(source["index_dir"], "learning", source["task_id"])
        evaluation.prepare(d, source["index_dir"], source["snapshot_path"], "learning",
                           task_ids=[public["task_id"]], source_dir=source_dir,
                           intervention={"instruction": request["executor_instruction"], "id": bid})
        entry = {"id": bid, "directory": str(d)}
        if entry not in source["branches"]:
            source["branches"].append(entry)
            write_json(source_dir / "source.json", source)
        return entry


def close(source_dir, request):
    source_dir = Path(source_dir).resolve()
    with lock(source_dir / ".source.lock"):
        source = read_json(source_dir / "source.json")
        p = progress(source_dir)
        if p["initial_score"] is None or len(p["attempts"]) != 1 + len(source["branches"]):
            raise ValueError("Closure requires all registered attempts to have final scores")
        if request["best_attempt"] not in p["attempts"] or p["attempts"][request["best_attempt"]]["score"] != p["best_score"]:
            raise ValueError("Closure must select a highest-scoring assessed attempt")
        status = request["status"]
        if status not in ("sufficient_gain", "stalled", "blocked", "saturated") or not request.get("reason"):
            raise ValueError("Reflector must record a closure status and its reasoning")
        gain = p["best_score"] - p["initial_score"]
        bundle = request.get("bundle")
        if status == "sufficient_gain":
            if gain <= evaluation.EPSILON or not isinstance(bundle, dict):
                raise ValueError("Sufficient gain requires positive measured improvement and a decision bundle")
            if any(not bundle.get(k) for k in ("trigger", "instruction", "scope", "implementation_checks")):
                raise ValueError("Bundle requires public trigger, instruction, scope, and implementation checks")
            refs = evidence(source_dir, request.get("evidence"))
        else:
            bundle = None
            refs = evidence(source_dir, request["evidence"]) if request.get("evidence") else []
        if status == "saturated" and p["initial_score"] < 1 - evaluation.EPSILON:
            raise ValueError("Saturation requires the initial official score ceiling")
        if status == "blocked" and not request.get("recovery_condition"):
            raise ValueError("Blocked sources require a recovery condition")
        from .records import verify_manual_update
        updates = []
        for d in [source_dir / "initial", *[Path(b["directory"]) for b in source["branches"]]]:
            for file in sorted((d / "manual-updates").glob("*/*.json")):
                verify_manual_update(file)
                updates.append({"path": str(file), "sha256": sha256(file)})
        result = {"task_id": source["task_id"], "scene_id": source["scene_id"], "status": status, "source_sha256": digest(source),
                  "best_attempt": request["best_attempt"], "initial_score": p["initial_score"],
                  "best_score": p["best_score"], "gain": gain, "reason": request["reason"],
                  "recovery_condition": request.get("recovery_condition"), "observations": request.get("observations", []),
                  "bundle": bundle, "evidence": refs, "manual_updates": updates,
                  "scores": {k: digest(v) for k, v in p["attempts"].items()}}
        immutable_json(source_dir / "experience.json", result)
        return result


def verify_experience(source_dir):
    source_dir = Path(source_dir).resolve()
    source, experience = read_json(source_dir / "source.json"), read_json(source_dir / "experience.json")
    if experience["source_sha256"] != digest(source):
        raise ValueError("Frozen experience must retain its source and snapshot binding")
    paths = {"initial": source_dir / "initial", **{b["id"]: Path(b["directory"]) for b in source["branches"]}}
    for name, expected in experience["scores"].items():
        current = attempt_score(paths[name])
        if digest(current) != expected:
            raise ValueError("Learning evidence must retain its frozen score binding")
    for ref in experience["evidence"]:
        if sha256(ref["path"]) != ref["sha256"]:
            raise ValueError("Frozen implementation evidence must retain its hash")
    return source, experience


def reviewed_experience(source_dir):
    _, experience = verify_experience(source_dir)
    return {k: experience[k] for k in ("task_id", "scene_id", "status", "best_attempt", "initial_score",
            "best_score", "gain", "reason", "recovery_condition", "observations", "bundle", "evidence", "scores")} | {
            "experience_sha256": digest(experience)}


def accepted_bundle(source_dir):
    experience = reviewed_experience(source_dir)
    if experience["status"] != "sufficient_gain":
        return None
    return {"scene_id": experience["scene_id"], "bundle": experience["bundle"],
            "learning": {k: experience[k] for k in ("initial_score", "best_score", "gain", "reason", "evidence")},
            "experience_sha256": experience["experience_sha256"]}
