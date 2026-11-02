"""Persistent round scheduling and complete-Agent promotion."""
from pathlib import Path
import random

from .common import ROOT, digest, identifier, immutable_json, lock, read_json, sha256, write_json, tree
from . import data, evaluation, learning, snapshots


def epoch_order(scenes, epoch):
    if len(scenes) != 33 or len(set(scenes)) != 33 or type(epoch) is not int or not 1 <= epoch <= 4:
        raise ValueError("An epoch requires 33 distinct learning scenes and an index in 1..4")
    order = sorted(scenes)
    random.Random(20261020 + epoch).shuffle(order)
    return [order[i:i+11] for i in range(0, 33, 11)]


def init_run(name, index, snapshot, root=ROOT):
    root, index = Path(root).resolve(), Path(index).resolve()
    m, snap = data.verify(index, check_inputs=True), snapshots.verify(snapshot)
    directory = root / "runs" / identifier(name)
    scenes = [r["scene_id"] for r in data.tasks(index, "learning")]
    record = {"protocol": "physics-iq-cross-view-v1", "run_id": name, "directory": str(directory),
              "index_dir": str(index), "data_revision": m["revision"], "initial_snapshot": snap["path"],
              "current_snapshot": snap["path"], "reference": None, "rounds": [], "tests": [],
              "rejection_streak": 0, "schedule": {str(e): epoch_order(scenes, e) for e in range(1, 5)}}
    path = directory / "run.json"
    if path.exists():
        old = read_json(path)
        if any(old[k] != record[k] for k in ("protocol", "index_dir", "data_revision", "initial_snapshot")):
            raise ValueError("Run name must retain its original bindings")
        return old
    return immutable_json(path, record)


def initial_reference(run_dir):
    state = read_json(Path(run_dir) / "run.json")
    return evaluation.prepare(Path(run_dir) / "initial-validation", state["index_dir"], state["initial_snapshot"], "validation")


def set_reference(run_dir):
    run_dir = Path(run_dir)
    with lock(run_dir / ".run.lock"):
        state = read_json(run_dir / "run.json")
        path = run_dir / "initial-validation/report.json"
        report = evaluation.report(path)
        snap = snapshots.verify(state["initial_snapshot"])
        if state["rounds"] or report["snapshot_id"] != snap["snapshot_id"] or report["data_revision"] != state["data_revision"] or report["count"] != 99:
            raise ValueError("Initial reference requires the initial complete Agent and full validation")
        state["reference"] = {"path": str(path.resolve()), "sha256": sha256(path)}
        write_json(run_dir / "run.json", state)
    return state["reference"]


def stopping(state):
    count = sum(r.get("decision") is not None for r in state["rounds"])
    return count >= 12 or count >= 3 and state["rejection_streak"] >= 5


def pending_test(state):
    count = sum(r.get("decision") is not None for r in state["rounds"])
    if count and (count % 3 == 0 or stopping(state)) and not any(t["after_round"] == count for t in state["tests"]):
        return {"after_round": count, "epoch": (count-1)//3+1, "label": "epoch" if count % 3 == 0 else "partial_epoch"}
    return None


def history(run_dir):
    state = read_json(Path(run_dir) / "run.json")
    bundles, decisions = [], []
    for r in state["rounds"]:
        if r.get("decision") is None:
            continue
        for s in r["sources"]:
            accepted = learning.accepted_bundle(s["directory"])
            if accepted:
                bundles.append({"round": r["round"], **accepted})
        decisions.append({"round": r["round"], **r["decision"]})
    return {"bundles": bundles, "validation": decisions}


def begin_round(run_dir):
    run_dir = Path(run_dir).resolve()
    with lock(run_dir / ".run.lock"):
        state = read_json(run_dir / "run.json")
        if state["rounds"] and state["rounds"][-1].get("decision") is None:
            return state["rounds"][-1]
        if state["reference"] is None:
            raise ValueError("Learning requires an initial validation reference")
        if pending_test(state) or stopping(state):
            raise ValueError("Next learning round requires an open schedule with completed boundary tests")
        number = len(state["rounds"]) + 1
        epoch, batch = (number-1)//3+1, (number-1)%3
        public = {r["scene_id"]: r for r in data.tasks(state["index_dir"], "learning")}
        snap = snapshots.verify(state["current_snapshot"])
        folder = run_dir / "rounds" / f"r{number:03d}"
        prior = immutable_json(folder / "history.json", history(run_dir))
        sources = []
        for scene in state["schedule"][str(epoch)][batch]:
            p = public[scene]
            d = folder / "sources" / scene
            record = {"run_dir": str(run_dir), "round": number, "scene_id": scene, "task_id": p["task_id"],
                      "index_dir": state["index_dir"], "snapshot_path": snap["path"], "snapshot_id": snap["snapshot_id"],
                      "history": {"path": str(folder / "history.json"), "sha256": digest(prior)}, "branches": []}
            if not (d / "source.json").exists():
                immutable_json(d / "source.json", record)
            evaluation.create_task(d / "initial", p, snap, source_dir=d)
            sources.append({"scene_id": scene, "task_id": p["task_id"], "directory": str(d)})
        result = {"round": number, "epoch": epoch, "batch": batch+1, "directory": str(folder),
                  "snapshot_path": snap["path"], "sources": sources, "manuals": None, "candidate": None, "decision": None}
        state["rounds"].append(result)
        write_json(run_dir / "run.json", state)
        return result


def completed_sources(current):
    if len(current["sources"]) != 11:
        raise ValueError("Consolidation requires all eleven learning sources")
    for s in current["sources"]:
        read_json(Path(s["directory"]) / "experience.json")
        learning.accepted_bundle(s["directory"])


def manual_inputs(run_dir):
    from .records import verify_manual_update
    state = read_json(Path(run_dir) / "run.json")
    current = state["rounds"][-1]
    completed_sources(current)
    updates = []
    for r in state["rounds"]:
        if r != current and r.get("decision", {}).get("accepted"):
            continue
        for s in r["sources"]:
            e = read_json(Path(s["directory"]) / "experience.json")
            for ref in e["manual_updates"]:
                if sha256(ref["path"]) != ref["sha256"]:
                    raise ValueError("Manual update must retain its frozen hash")
                verify_manual_update(ref["path"])
                updates.append(ref)
    return {"base_manuals": str(Path(current["snapshot_path"]) / "manuals"), "updates": updates,
            "output_manuals": str(Path(current["directory"]) / "manual-manager/manuals"),
            "merge_notes": str(Path(current["directory"]) / "manual-manager/merge.md")}


def publish_manuals(run_dir):
    run_dir = Path(run_dir)
    with lock(run_dir / ".run.lock"):
        state = read_json(run_dir / "run.json")
        current = state["rounds"][-1]
        inputs = manual_inputs(run_dir)
        files = tree(inputs["output_manuals"])
        if not files or not set(tree(inputs["base_manuals"])) <= set(files) or any(Path(p).suffix != ".md" for p in files):
            raise ValueError("Manual publication requires the complete Markdown manual tree")
        if not Path(inputs["merge_notes"]).read_text().strip():
            raise ValueError("Manual publication requires merge notes")
        receipt = {**inputs, "hashes": files, "notes_sha256": sha256(inputs["merge_notes"])}
        immutable_json(Path(current["directory"]) / "manual-manager/publication.json", receipt)
        current["manuals"] = receipt
        write_json(run_dir / "run.json", state)
        return receipt


def candidate(run_dir, policy=None, templates=None):
    run_dir = Path(run_dir)
    with lock(run_dir / ".run.lock"):
        state = read_json(run_dir / "run.json")
        current = state["rounds"][-1]
        completed_sources(current)
        if not current["manuals"] or tree(current["manuals"]["output_manuals"]) != current["manuals"]["hashes"]:
            raise ValueError("Candidate requires the staged, frozen manuals")
        base = Path(current["snapshot_path"])
        snap = snapshots.freeze(policy or base / "policy", templates=templates or base / "source/templates", manuals=current["manuals"]["output_manuals"])
        original = snapshots.verify(base)["binding"]
        fixed = lambda b: {k: v for k, v in b["source"].items() if not k.startswith("templates/")}
        if snap["binding"]["runtime"] != original["runtime"] or fixed(snap["binding"]) != fixed(original):
            raise ValueError("Candidate must retain the run's fixed implementation and role configuration")
        if current["candidate"]:
            if current["candidate"]["snapshot_id"] != snap["snapshot_id"]:
                raise ValueError("Round candidate is already frozen")
            return current["candidate"]
        reference = evaluation.report(state["reference"]["path"])
        if snap["snapshot_id"] == reference["snapshot_id"]:
            location = str(Path(state["reference"]["path"]).parent)
        else:
            location = str(Path(current["directory"]) / "validation")
            evaluation.prepare(location, state["index_dir"], snap["path"], "validation")
        current["candidate"] = {"snapshot_id": snap["snapshot_id"], "snapshot_path": snap["path"], "evaluation": location}
        write_json(run_dir / "run.json", state)
        return current["candidate"]


def decide(run_dir):
    run_dir = Path(run_dir)
    with lock(run_dir / ".run.lock"):
        state = read_json(run_dir / "run.json")
        current = state["rounds"][-1]
        if current["decision"]:
            return current["decision"]
        if sha256(state["reference"]["path"]) != state["reference"]["sha256"]:
            raise ValueError("Reference report must retain its registered hash")
        c = current["candidate"]
        report = evaluation.report(c["evaluation"])
        if report["snapshot_id"] != c["snapshot_id"]:
            raise ValueError("Validation must evaluate the frozen candidate")
        decision = evaluation.compare(report, evaluation.report(state["reference"]["path"]))
        current["decision"] = {**decision, "mean": report["mean"], "categories": report["categories"]}
        if decision["accepted"]:
            state["current_snapshot"] = c["snapshot_path"]
            p = Path(c["evaluation"]) / "report.json"
            state["reference"] = {"path": str(p), "sha256": sha256(p)}
            state["rejection_streak"] = 0
        else:
            state["rejection_streak"] += 1
        write_json(run_dir / "run.json", state)
        immutable_json(Path(current["directory"]) / "decision.json", current["decision"])
        return current["decision"]


def prepare_test(run_dir):
    state = read_json(Path(run_dir) / "run.json")
    boundary = pending_test(state)
    if boundary is None:
        raise ValueError("A complete test requires a pending epoch or stopping boundary")
    d = Path(run_dir) / "tests" / f"after-r{boundary['after_round']:03d}"
    return evaluation.prepare(d, state["index_dir"], state["current_snapshot"], "test", boundary={**boundary, "run_id": state["run_id"]})


def register_test(run_dir, report_path):
    run_dir = Path(run_dir)
    with lock(run_dir / ".run.lock"):
        state = read_json(run_dir / "run.json")
        expected = pending_test(state)
        r = evaluation.report(report_path)
        if (expected is None or r["phase"] != "test" or r["count"] != 198
                or r["boundary"] != {**expected, "run_id": state["run_id"]}
                or r["snapshot_id"] != snapshots.verify(state["current_snapshot"])["snapshot_id"]):
            raise ValueError("Boundary test must evaluate the current complete Agent on all 198 inputs")
        p = Path(report_path)
        p = p / "report.json" if p.is_dir() else p
        state["tests"].append({**expected, "path": str(p.resolve()), "sha256": sha256(p)})
        write_json(run_dir / "run.json", state)
        return {"complete": stopping(state), "test": state["tests"][-1]}
