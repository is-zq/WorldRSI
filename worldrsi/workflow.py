"""Persistent round scheduling and complete-Agent promotion."""
from pathlib import Path
import random

from .common import ROOT, digest, identifier, immutable_json, lock, read_json, sha256, write_json, tree
from . import data, evaluation, learning, snapshots

PROTOCOL = "physics-iq-learning-promotion-v2"
MAX_PROMOTION_ATTEMPTS = 4


def promotion_attempts(current):
    """Legacy single candidates retain their original identity as attempt one."""
    if current.get("promotion_attempts"):
        return current["promotion_attempts"]
    c = current.get("candidate")
    return [{**c, "attempt": 1, "assessment": current.get("decision")}] if c else []


def revision_required(current):
    attempts = promotion_attempts(current)
    assessment = attempts[-1].get("assessment") if attempts else None
    return (current.get("decision") is None and len(attempts) < MAX_PROMOTION_ATTEMPTS
            and bool(assessment) and assessment.get("reason") == "score"
            and assessment.get("gain", 0) < -evaluation.EPSILON)


def file_reference(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256(path)}


def promotion_feedback(candidate_report, baseline_report):
    """Numerical comparisons and bound evidence; causal analysis belongs to Editor."""
    reports = {"candidate": candidate_report, "baseline": baseline_report}
    refs = {name: file_reference(Path(r["directory"]) / "report.json") for name, r in reports.items()}
    inputs, evidence = [], {}
    for tid in candidate_report["task_ids"]:
        entry = {"task_id": tid}
        for name, report in reports.items():
            per_seed = {str(s): report["scores"][str(s)][tid] for s in evaluation.SEEDS}
            outputs = {}
            for seed in evaluation.SEEDS:
                d = Path(report["outputs"][str(seed)][tid])
                score_path = d / "scoring/score.json"
                context = read_json(d / "task.json")
                entry["public"] = {k: v for k, v in context["public"].items() if k != "split"}
                files = [d / "task.json"]
                output = {"task_dir": str(d)}
                if str(score_path) in report["evidence"]:
                    score = read_json(score_path)
                    component = score["components"]
                    if sha256(component["path"]) != component["sha256"]:
                        raise ValueError("Promotion component evidence changed")
                    output.update(score=file_reference(score_path),
                                  components={k: component[k] for k in ("path", "sha256")},
                                  valid_output=score["valid_output"])
                    files.extend([score_path, Path(component["path"])])
                files.extend(p for p in (d / "output.lock.json", d / "provenance.json", d / "h3.submission.json") if p.exists())
                files.extend(sorted((d / "trace").glob("*/*.json")))
                output["execution_evidence"] = [file_reference(p) for p in files]
                evidence.update({str(p.resolve()): sha256(p) for p in files})
                outputs[str(seed)] = output
            entry[name] = {"per_seed": per_seed, "mean": sum(per_seed.values()) / 4, "outputs": outputs}
        entry["gain"] = entry["candidate"]["mean"] - entry["baseline"]["mean"]
        inputs.append(entry)
    return {"reports": refs, "inputs": sorted(inputs, key=lambda x: (x["gain"], x["task_id"])),
            "mean": candidate_report["mean"], "baseline_mean": baseline_report["mean"],
            "gain": candidate_report["mean"] - baseline_report["mean"], "evidence": evidence}


def editor_revision(current):
    if current.get("decision") is not None:
        raise ValueError("Editor requires an open round")
    attempts = promotion_attempts(current)
    if attempts and not revision_required(current):
        raise ValueError("Editor revision requires a completed score decline with retries remaining")
    feedback = []
    for attempt in attempts:
        revision = attempt.get("revision")
        if revision and sha256(revision["path"]) != revision["sha256"]:
            raise ValueError("Editor revision rationale changed")
        ref = attempt["feedback"]
        if sha256(ref["path"]) != ref["sha256"]:
            raise ValueError("Promotion feedback changed")
        record = read_json(ref["path"])
        for path, expected in record["evidence"].items():
            if sha256(path) != expected:
                raise ValueError("Promotion feedback evidence changed")
        for report in record["reports"].values():
            if sha256(report["path"]) != report["sha256"]:
                raise ValueError("Promotion feedback report changed")
            evaluation.report(report["path"])
        summary = {k: record[k] for k in ("mean", "baseline_mean", "gain", "reports")}
        summary["inputs"] = [{**{k: item[k] for k in ("task_id", "public", "gain")},
                              **{name: {k: item[name][k] for k in ("mean", "per_seed")}
                                 for name in ("candidate", "baseline")}} for item in record["inputs"]]
        feedback.append({"attempt": attempt["attempt"], "snapshot_path": attempt["snapshot_path"],
                         "assessment": attempt["assessment"], "feedback": ref, "comparison": summary,
                         "revision": attempt.get("revision")})
    number = len(attempts) + 1
    base = Path(attempts[-1]["snapshot_path"] if attempts else current["snapshot_path"])
    output = Path(current["directory"]) / "editor"
    if attempts:
        output = output / f"attempt-{number:02d}"
    return {"attempt": number, "max_attempts": MAX_PROMOTION_ATTEMPTS,
            "policy": str(base / "policy"), "templates": str(base / "source/templates"),
            "promotion_feedback": feedback, "output_directory": str(output)}


def learning_run(run_dir):
    state = read_json(Path(run_dir) / "run.json")
    if state["protocol"] != PROTOCOL:
        raise ValueError(f"Learning requires run protocol {PROTOCOL}")
    return state


def epoch_order(scenes, epoch):
    if len(scenes) != 33 or len(set(scenes)) != 33 or type(epoch) is not int or not 1 <= epoch <= 4:
        raise ValueError("An epoch requires 33 distinct learning scenes and an index in 1..4")
    order = sorted(scenes)
    random.Random(20261020 + epoch).shuffle(order)
    return [order[i:i+11] for i in range(0, 33, 11)]


def init_run(name, index, snapshot, root=ROOT, *, mode="full"):
    root, index = Path(root).resolve(), Path(index).resolve()
    m, snap = data.verify(index, check_inputs=True), snapshots.verify(snapshot)
    directory = root / "runs" / identifier(name)
    scenes = [r["scene_id"] for r in data.tasks(index, "learning")]
    evaluation.views_for(mode)
    record = {"protocol": PROTOCOL, "run_id": name, "directory": str(directory),
              "index_dir": str(index), "data_revision": m["revision"], "initial_snapshot": snap["path"],
              "current_snapshot": snap["path"], "rounds": [], "tests": [], "mode": mode,
              "rejection_streak": 0, "schedule": {str(e): epoch_order(scenes, e) for e in range(1, 5)}}
    path = directory / "run.json"
    if path.exists():
        old = read_json(path)
        if any(old[k] != record[k] for k in ("protocol", "index_dir", "data_revision", "initial_snapshot", "mode")):
            raise ValueError("Run name must retain its original bindings")
        return old
    return immutable_json(path, record)


def stopping(state):
    count = sum(r.get("decision") is not None for r in state["rounds"])
    return count >= 12 or count >= 3 and state["rejection_streak"] >= 5


def history(run_dir):
    state = learning_run(run_dir)
    bundles, decisions = [], []
    for r in state["rounds"]:
        if r.get("decision") is None:
            continue
        for s in r["sources"]:
            accepted = learning.accepted_bundle(s["directory"])
            if accepted:
                bundles.append({"round": r["round"], **accepted})
        decisions.append({"round": r["round"], **r["decision"], "promotion_attempts": promotion_attempts(r)})
    return {"bundles": bundles, "promotion": decisions}


def begin_round(run_dir):
    run_dir = Path(run_dir).resolve()
    with lock(run_dir / ".run.lock"):
        state = learning_run(run_dir)
        if state["rounds"] and state["rounds"][-1].get("decision") is None:
            return state["rounds"][-1]
        if stopping(state):
            raise ValueError("Learning schedule is complete")
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
            evaluation.prepare(d / "initial", state["index_dir"], snap["path"], "learning",
                               task_ids=[p["task_id"]], source_dir=d)
            sources.append({"scene_id": scene, "task_id": p["task_id"], "directory": str(d)})
        result = {"round": number, "epoch": epoch, "batch": batch+1, "directory": str(folder),
                  "snapshot_path": snap["path"], "sources": sources, "manuals": None, "candidate": None,
                  "promotion_attempts": [], "decision": None}
        state["rounds"].append(result)
        write_json(run_dir / "run.json", state)
        return result


def completed_sources(current):
    if len(current["sources"]) != 11:
        raise ValueError("Consolidation requires all eleven learning sources")
    return [learning.reviewed_experience(s["directory"]) for s in current["sources"]]


def editor_experiences(run_dir):
    """All closed sources, including unsuccessful rounds, remain synthesis inputs."""
    state = learning_run(run_dir)
    result = []
    for current in state["rounds"]:
        result.extend({"round": current["round"], **experience}
                      for experience in completed_sources(current))
    return result


def manual_inputs(run_dir):
    from .records import verify_manual_update
    state = learning_run(run_dir)
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
        state = learning_run(run_dir)
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
        state = learning_run(run_dir)
        current = state["rounds"][-1]
        completed_sources(current)
        if not current["manuals"] or tree(current["manuals"]["output_manuals"]) != current["manuals"]["hashes"]:
            raise ValueError("Candidate requires the staged, frozen manuals")
        base = Path(current["snapshot_path"])
        snapshots.decision_experience(policy if policy is not None else base / "policy")
        snap = snapshots.derive(base, policy=policy, templates=templates, manuals=current["manuals"]["output_manuals"])
        original = snapshots.verify(base)["binding"]
        fixed = lambda b: {k: v for k, v in b["source"].items() if not k.startswith("templates/")}
        if snap["binding"]["runtime"] != original["runtime"] or fixed(snap["binding"]) != fixed(original):
            raise ValueError("Candidate must retain the run's fixed implementation and role configuration")
        attempts = promotion_attempts(current)
        for prior in attempts:
            if prior["snapshot_id"] == snap["snapshot_id"]:
                return prior
        if current.get("decision") is not None or (attempts and not revision_required(current)):
            raise ValueError("New candidate requires a score decline with retries remaining in an open round")
        number = len(attempts) + 1
        revision = None
        if number > 1:
            rationale = Path(policy).parent / "revision.md" if policy is not None else None
            if rationale is None or not rationale.is_file() or not rationale.read_text().strip():
                raise ValueError("Revised candidate requires revision.md beside its policy directory")
            revision = file_reference(rationale)
        ids = [source["task_id"] for source in current["sources"]]
        boundary = {"run_id": state["run_id"], "round": current["round"],
                    "baseline_snapshot_id": snapshots.verify(base)["snapshot_id"]}
        baseline = str(Path(current["directory"]) / "promotion/baseline")
        evaluation.prepare(baseline, state["index_dir"], base, "promotion", task_ids=ids, boundary=boundary,
                           learning_attempts=[Path(source["directory"]) / "initial" for source in current["sources"]])
        location = str(Path(current["directory"]) / "promotion" / ("candidate" if number == 1 else f"candidate-{number:02d}"))
        evaluation.prepare(location, state["index_dir"], snap["path"], "promotion", task_ids=ids, boundary=boundary)
        current["candidate"] = {"attempt": number, "snapshot_id": snap["snapshot_id"], "snapshot_path": snap["path"],
                                "evaluation": location, "baseline_evaluation": baseline, "assessment": None}
        if revision:
            current["candidate"]["revision"] = revision
        current["promotion_attempts"] = [*attempts, current["candidate"]]
        write_json(run_dir / "run.json", state)
        return current["candidate"]


def decide(run_dir):
    run_dir = Path(run_dir)
    with lock(run_dir / ".run.lock"):
        state = learning_run(run_dir)
        current = state["rounds"][-1]
        if current["decision"]:
            return current["decision"]
        c = current["candidate"]
        if not c:
            raise ValueError("Decision requires a frozen candidate")
        attempts = promotion_attempts(current)
        c = attempts[-1]
        report = evaluation.report(c["evaluation"])
        baseline = evaluation.report(c["baseline_evaluation"])
        ids = sorted(source["task_id"] for source in current["sources"])
        expected_boundary = {"run_id": state["run_id"], "round": current["round"],
                             "baseline_snapshot_id": snapshots.verify(current["snapshot_path"])["snapshot_id"]}
        if (state["current_snapshot"] != current["snapshot_path"]
                or report["snapshot_id"] != c["snapshot_id"] or baseline["snapshot_id"] != expected_boundary["baseline_snapshot_id"]
                or sorted(report["task_ids"]) != ids or report["boundary"] != expected_boundary
                or report["data_revision"] != state["data_revision"]):
            raise ValueError("Promotion must compare the frozen candidate and round-start Agent on this round's inputs")
        decision = evaluation.compare(report, baseline)
        assessment = {**decision, "mean": report["mean"], "baseline_mean": baseline["mean"],
                               "categories": report["categories"],
                               "reports": {name: {"path": str(Path(path) / "report.json"),
                                                   "sha256": sha256(Path(path) / "report.json")}
                                           for name, path in (("candidate", c["evaluation"]), ("baseline", c["baseline_evaluation"]))}}
        immutable_json(Path(c["evaluation"]) / "assessment.json", assessment)
        c["assessment"] = assessment
        if decision["reason"] == "score" and decision["gain"] < -evaluation.EPSILON:
            feedback_path = Path(c["evaluation"]) / "editor-feedback.json"
            immutable_json(feedback_path, promotion_feedback(report, baseline))
            c["feedback"] = file_reference(feedback_path)
        current["promotion_attempts"] = attempts
        current["candidate"] = c
        if revision_required(current):
            write_json(run_dir / "run.json", state)
            return {"status": "revision_required", "attempt": c["attempt"],
                    "remaining_revisions": MAX_PROMOTION_ATTEMPTS - len(attempts),
                    "assessment": assessment, "feedback": c["feedback"]}
        current["decision"] = {**assessment, "attempt": c["attempt"],
                               "status": "accepted" if decision["accepted"] else "rejected"}
        if decision["accepted"]:
            state["current_snapshot"] = c["snapshot_path"]
            state["rejection_streak"] = 0
        else:
            state["rejection_streak"] += 1
        current["result_snapshot"] = state["current_snapshot"]
        immutable_json(Path(current["directory"]) / "decision.json", current["decision"])
        write_json(run_dir / "run.json", state)
        return current["decision"]


def test_snapshot(state, round_number, target, attempt=None):
    if type(round_number) is not int or round_number < 0 or target not in ("candidate", "active"):
        raise ValueError("Test selection requires a nonnegative round and candidate/active target")
    if attempt is not None and (type(attempt) is not int or attempt < 1 or target != "candidate" or round_number == 0):
        raise ValueError("Attempt selection requires a positive candidate attempt and round")
    if round_number == 0:
        return state["initial_snapshot"]
    current = next((r for r in state["rounds"] if r["round"] == round_number), None)
    if current is None:
        raise ValueError("Unknown test round")
    if target == "candidate":
        if not current.get("candidate"):
            raise ValueError("Test round has no frozen candidate")
        candidates = promotion_attempts(current)
        selected = candidates[-1] if attempt is None else next((c for c in candidates if c["attempt"] == attempt), None)
        if selected is None:
            raise ValueError("Unknown candidate attempt")
        return selected["snapshot_path"]
    if current.get("decision") is None:
        raise ValueError("Active Harness requires a completed round decision")
    return current["candidate"]["snapshot_path"] if current["decision"]["accepted"] else current["snapshot_path"]


def prepare_test(run_dir, mode=None, *, round_number, target="candidate", attempt=None):
    state = read_json(Path(run_dir) / "run.json")
    snapshot = test_snapshot(state, round_number, target, attempt)
    mode = mode or state["mode"]
    boundary = {"run_id": state["run_id"], "round": round_number, "target": target}
    suffix = ""
    if round_number > 0 and target == "candidate":
        current = next(r for r in state["rounds"] if r["round"] == round_number)
        attempt = attempt or promotion_attempts(current)[-1]["attempt"]
        boundary["attempt"] = attempt
        suffix = f"-a{attempt:02d}"
    directory = Path(run_dir) / "tests" / f"r{round_number:03d}-{target}{suffix}-{mode}"
    extension = None
    if (directory / "evaluation.json").exists():
        plan = evaluation.load(directory)
        if plan["snapshot_id"] != snapshots.verify(snapshot)["snapshot_id"] or plan["boundary"] != boundary:
            raise ValueError("Test selection changed after preparation")
        extension = plan["extended_from"]["path"] if plan["extended_from"] else None
    return evaluation.prepare(directory, state["index_dir"], snapshot, "test", mode=mode, boundary=boundary,
                              extend_from=extension)


def register_test(run_dir, report_path):
    run_dir = Path(run_dir)
    with lock(run_dir / ".run.lock"):
        state = read_json(run_dir / "run.json")
        r = evaluation.report(report_path)
        boundary = r["boundary"] or {}
        attempt = boundary.get("attempt")
        if attempt is None and boundary.get("target") == "candidate" and boundary.get("round", 0) > 0:
            attempt = 1  # Reports created before revision support bound the first candidate.
        snapshot = test_snapshot(state, boundary.get("round"), boundary.get("target"), attempt)
        expected = {"run_id": state["run_id"], "round": boundary["round"], "target": boundary["target"]}
        if "attempt" in boundary:
            expected["attempt"] = boundary["attempt"]
        if (r["phase"] != "test" or r.get("population") != "heldout" or r["boundary"] != expected
                or r["snapshot_id"] != snapshots.verify(snapshot)["snapshot_id"]
                or r["data_revision"] != state["data_revision"] or r["count"] != evaluation.input_count("test", r["mode"])):
            raise ValueError("Test report must cover the selected Harness and held-out inputs")
        expected_ids = {row["task_id"] for row in data.tasks(state["index_dir"], "validation")
                        if row["view"] in evaluation.views_for(r["mode"])}
        if set(r["task_ids"]) != expected_ids:
            raise ValueError("Test report must cover the exact held-out inputs")
        path = Path(report_path)
        path = path / "report.json" if path.is_dir() else path
        receipt = {**expected, "path": str(path.resolve()), "sha256": sha256(path),
                   "snapshot_id": r["snapshot_id"], "mode": r["mode"], "mean": r["mean"]}
        for prior in state["tests"]:
            if (all(prior.get(k) == receipt[k] for k in ("round", "target", "mode"))
                    and prior.get("attempt", 1) == receipt.get("attempt", 1)):
                if prior != receipt or sha256(prior["path"]) != prior["sha256"]:
                    raise ValueError("Registered test must retain its frozen report")
                return {"test": prior}
        state["tests"].append(receipt)
        write_json(run_dir / "run.json", state)
        return {"test": receipt}
