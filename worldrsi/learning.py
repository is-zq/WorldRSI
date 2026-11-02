"""Adaptive source experiments, Reflector closure, and paired-view transfer."""
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
        p = Path(name).resolve()
        if not p.is_relative_to(source):
            raise ValueError("Evidence must belong to its learning source")
        result.append({"path": str(p), "sha256": sha256(p)})
    return result


def progress(source_dir):
    source_dir = Path(source_dir)
    source = read_json(source_dir / "source.json")
    results = {}
    for name, path in [("initial", source_dir / "initial"), *[(b["id"], Path(b["directory"])) for b in source["branches"]]]:
        if (path / "scoring/score.json").exists():
            results[name] = evaluation.score(path)
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
        evaluation.create_task(d, public, source["snapshot_path"], source_dir=source_dir,
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
            bundle, refs = None, []
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
        if bundle is None:
            immutable_json(source_dir / "transfer/decision.json", {"passed": False, "status": "source_closed", "experience_sha256": digest(result)})
        return result


def verify_experience(source_dir):
    source_dir = Path(source_dir).resolve()
    source, experience = read_json(source_dir / "source.json"), read_json(source_dir / "experience.json")
    if experience["source_sha256"] != digest(source):
        raise ValueError("Frozen experience must retain its source and snapshot binding")
    paths = {"initial": source_dir / "initial", **{b["id"]: Path(b["directory"]) for b in source["branches"]}}
    for name, expected in experience["scores"].items():
        if digest(read_json(paths[name] / "scoring/score.json")) != expected:
            raise ValueError("Learning evidence must retain its frozen score binding")
    for ref in experience["evidence"]:
        if sha256(ref["path"]) != ref["sha256"]:
            raise ValueError("Frozen implementation evidence must retain its hash")
    return source, experience


def prepare_transfer(source_dir):
    source_dir = Path(source_dir).resolve()
    source, experience = verify_experience(source_dir)
    if experience["status"] != "sufficient_gain":
        raise ValueError("Transfer requires a frozen sufficiently improved experience")
    rows = [r for r in data.tasks(source["index_dir"], "transfer") if r["scene_id"] == source["scene_id"]]
    ids = [r["task_id"] for r in rows]
    intervention = {"bundle": experience["bundle"], "experience_sha256": digest(experience)}
    result = {}
    for arm in ("baseline", "candidate"):
        d = source_dir / "transfer" / arm
        result[arm] = evaluation.prepare(d, source["index_dir"], source["snapshot_path"], "transfer",
                                         task_ids=ids, intervention=intervention if arm == "candidate" else None)
    return result


def transfer_gains(baseline, candidate):
    if any(baseline[k] != candidate[k] for k in ("task_ids", "seeds", "snapshot_id", "data_revision", "scorer")):
        raise ValueError("Transfer arms require matching baseline bindings")
    gains = {t: sum(candidate["scores"][str(s)][t] - baseline["scores"][str(s)][t] for s in evaluation.SEEDS) / 4
             for t in baseline["task_ids"]}
    return gains, len(gains) == 2 and all(v > evaluation.EPSILON for v in gains.values())


def finalize_transfer(source_dir):
    source_dir = Path(source_dir).resolve()
    experience = read_json(source_dir / "experience.json")
    plans = prepare_transfer(source_dir)
    a, b = [evaluation.report(plans[arm]["directory"]) for arm in ("baseline", "candidate")]
    if a["intervention"] is not None or b["intervention"] != {"bundle": experience["bundle"], "experience_sha256": digest(experience)}:
        raise ValueError("Transfer reports must bind the frozen experience")
    checks = {}
    for tid, directory in b["outputs"]["42"].items():
        p = Path(directory) / "implementation.json"
        proof = read_json(p)
        if proof.get("experience_sha256") != digest(experience) or proof.get("applied") is not True:
            raise ValueError("Target-view execution must confirm the frozen experience was applied")
        refs = evidence(directory, proof.get("evidence"))
        if not proof.get("assessment"):
            raise ValueError("Implementation proof requires a checked assessment")
        checks[tid] = {"proof": {"path": str(p), "sha256": sha256(p)}, "evidence": refs}
    gains, passed = transfer_gains(a, b)
    return immutable_json(source_dir / "transfer/decision.json", {"passed": passed, "status": "evaluated",
                          "experience_sha256": digest(experience), "gains": gains, "implementation": checks,
                          "reports": {arm: {"path": str(Path(plans[arm]["directory"]) / "report.json"),
                                            "sha256": sha256(Path(plans[arm]["directory"]) / "report.json")} for arm in plans}})


def accepted_bundle(source_dir):
    source_dir = Path(source_dir)
    decision = read_json(source_dir / "transfer/decision.json")
    experience = read_json(source_dir / "experience.json")
    if decision["experience_sha256"] != digest(experience):
        raise ValueError("Transfer decision must bind the source experience")
    if not decision["passed"]:
        return None
    verify_experience(source_dir)
    for ref in decision["reports"].values():
        if sha256(ref["path"]) != ref["sha256"]:
            raise ValueError("Transfer receipt must retain its scored report binding")
        evaluation.report(ref["path"])
    for implementation in decision["implementation"].values():
        for ref in [implementation["proof"], *implementation["evidence"]]:
            if sha256(ref["path"]) != ref["sha256"]:
                raise ValueError("Passed experience must retain its checked target implementation")
    reports = [evaluation.report(decision["reports"][arm]["path"]) for arm in ("baseline", "candidate")]
    gains, passed = transfer_gains(*reports)
    if not passed or gains != decision["gains"]:
        raise ValueError("Passing receipt requires improvement on each reserved view")
    return {"scene_id": experience["scene_id"], "bundle": experience["bundle"],
            "learning": {k: experience[k] for k in ("initial_score", "best_score", "gain", "reason", "evidence")},
            "transfer": {"passed": True, "receipt_sha256": digest(decision)}}
