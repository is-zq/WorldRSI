"""Seed-paired benchmark evaluations with frozen conditioning and resumable reports."""
from pathlib import Path
import statistics

from .common import ROOT, digest, immutable_json, read_json, sha256
from . import data, snapshots
from .records import total_cost
from .media import verify_locked

SEEDS = [42, 43, 44, 45]
EPSILON = 1e-12


def require_active_phase(phase):
    if phase not in ("learning", "promotion", "test"):
        raise ValueError("Active phases are learning, promotion and test")


def views_for(mode):
    from .benchmark import VIEWS
    if mode not in ("quick", "full"):
        raise ValueError("Evaluation mode must be quick or full")
    return list(VIEWS[:1] if mode == "quick" else VIEWS)


def input_count(phase, mode):
    return 11 if phase == "promotion" else 33 * len(views_for(mode))


def generation_seed(context):
    seed = context["generation_seed"]
    allowed = SEEDS
    if type(seed) is not int or seed not in allowed:
        raise ValueError("Generation seed must belong to the phase's frozen seed set")
    return seed


def create_task(directory, public, snapshot, *, seed=42, source_dir=None, intervention=None, modeling_task=None):
    require_active_phase(public["split"])
    snap = snapshot if isinstance(snapshot, dict) else snapshots.verify(snapshot)
    if digest(snap["binding"]) != snap["snapshot_id"]:
        raise ValueError("Task requires a complete, verified snapshot binding")
    for field, expected in public["input_hashes"].items():
        if sha256(public[field]) != expected:
            raise ValueError("Public input must match its frozen hash")
    context = {"public": public, "phase": public["split"], "generation_seed": seed,
               "snapshot_id": snap["snapshot_id"], "snapshot_path": snap["path"],
               "roles": snap["binding"]["runtime"]["roles"], "role": "executor",
               "policy_snapshot": str(Path(snap["path"]) / "policy"),
               "manual_snapshot": str(Path(snap["path"]) / "manuals"),
               "template_snapshot": str(Path(snap["path"]) / "source/templates"),
               "intervention": intervention, "source_dir": str(Path(source_dir).resolve()) if source_dir else None,
               "modeling_task": str(modeling_task) if modeling_task else None}
    generation_seed(context)
    return immutable_json(Path(directory) / "task.json", context)


def prepare(directory, index, snapshot, phase, *, task_ids=None, intervention=None, boundary=None,
            mode="full", extend_from=None, source_dir=None, learning_attempts=None):
    require_active_phase(phase)
    directory, index = Path(directory).resolve(), Path(index).resolve()
    snap = snapshots.verify(snapshot)
    rows = data.tasks(index, phase)
    selected_views = views_for(mode)
    if phase == "test":
        heldout = {r["scene_id"] for r in data.tasks(index, "validation")}
        rows = [r for r in rows if r["scene_id"] in heldout and r["view"] in selected_views]
        if task_ids is not None or len(rows) != input_count(phase, mode):
            raise ValueError("Testing requires the complete held-out input set")
    else:
        count = 1 if phase == "learning" else 11
        if not task_ids or len(task_ids) != count or len(set(task_ids)) != count:
            raise ValueError(f"{phase} requires {count} distinct learning inputs")
        rows = [r for r in rows if r["task_id"] in task_ids]
        if len(rows) != count:
            raise ValueError("Inputs must belong to the indexed learning views")
        if phase == "promotion" and (intervention is not None or not boundary):
            raise ValueError("Promotion requires a round binding and no intervention")
        mode = "full"  # Learning views are fixed; quick/full only controls tests.
    views = sorted({r["view"] for r in rows})
    config = read_json(ROOT / "protocol/scorer.json")
    extra = {"population": "heldout"} if phase == "test" else {}
    extension, inherited = None, None
    if extend_from is not None:
        inherited = load(extend_from)
        if (phase != "test" or inherited["phase"] != phase or inherited.get("population") != "heldout"
                or inherited["mode"] != "quick" or mode != "full"
                or inherited["snapshot_id"] != snap["snapshot_id"] or inherited["data_revision"] != rows[0]["data_revision"]
                or inherited["scorer"] != config or inherited["boundary"] != boundary
                or inherited["intervention"] != intervention or inherited["seeds"] != SEEDS
                or not set(inherited["task_ids"]) <= {r["task_id"] for r in rows}):
            raise ValueError("Full completion requires the same quick test snapshot, inputs, seeds and boundary")
        path = Path(inherited["directory"]) / "evaluation.json"
        extension = {"path": str(path), "sha256": sha256(path)}
    outputs = {str(seed): {r["task_id"]: str(directory / "seeds" / str(seed) / r["task_id"]) for r in rows} for seed in SEEDS}
    shared_ids = set()
    if phase == "learning":
        outputs["42"][rows[0]["task_id"]] = str(directory)
    if learning_attempts is not None:
        if phase != "promotion" or len(learning_attempts) != 11:
            raise ValueError("Promotion baseline requires eleven initial learning reports")
        refs = []
        for attempt in learning_attempts:
            source = report(attempt)
            if (source["phase"] != "learning" or source["snapshot_id"] != snap["snapshot_id"]
                    or source["data_revision"] != rows[0]["data_revision"] or source["scorer"] != config
                    or source["seeds"] != SEEDS or source["intervention"] is not None
                    or len(source["task_ids"]) != 1):
                raise ValueError("Baseline must reuse initial results of the same Agent, data, seeds and scorer")
            tid = source["task_ids"][0]
            if tid not in outputs["42"] or tid in shared_ids:
                raise ValueError("Baseline initial inputs must exactly match the round")
            shared_ids.add(tid)
            path = Path(source["directory"]) / "report.json"
            refs.append({"path": str(path), "sha256": sha256(path)})
            for seed in SEEDS:
                outputs[str(seed)][tid] = source["outputs"][str(seed)][tid]
        extra["reuse_learning"] = refs
    if inherited:
        shared_ids.update(inherited["task_ids"])
        for seed in SEEDS:
            outputs[str(seed)].update(inherited["outputs"][str(seed)])
    record = {"directory": str(directory), "index_dir": str(index), "phase": phase,
              "snapshot_id": snap["snapshot_id"], "snapshot_path": snap["path"],
              "data_revision": rows[0]["data_revision"], "scorer": config,
              "task_ids": [r["task_id"] for r in rows], "seeds": SEEDS, "views": views, "mode": mode,
              "extended_from": extension, "outputs": outputs, **extra,
              "intervention": intervention, "boundary": boundary, "expected_generations": len(rows) * 4,
              "additional_generations": (len(rows) - len(shared_ids)) * 4}
    record["key"] = plan_key(record)
    immutable_json(directory / "evaluation.json", record)
    for seed in SEEDS:
        for public in rows:
            if public["task_id"] in shared_ids:
                continue
            create_task(outputs[str(seed)][public["task_id"]], public, snap, seed=seed, intervention=intervention, source_dir=source_dir,
                        modeling_task=outputs["42"][public["task_id"]] if seed != 42 else None)
    return record


def plan_key(plan):
    extras = {k: plan[k] for k in ("reuse_test", "reuse_learning", "population") if k in plan}
    return digest({**extras, "snapshot": plan["snapshot_id"], "revision": plan["data_revision"],
                   "phase": plan["phase"], "ids": plan["task_ids"], "intervention": plan["intervention"],
                   "seeds": plan["seeds"], "views": plan["views"], "mode": plan["mode"],
                   "extended_from": plan["extended_from"], "scorer": plan["scorer"], "boundary": plan["boundary"]})


def prepare_learning(directory, index=None):
    """Add the three replicas to an existing primary task without rewriting it."""
    directory = Path(directory).resolve()
    if (directory / "evaluation.json").exists():
        plan = load(directory)
        if plan["phase"] != "learning":
            raise ValueError("Learning preparation requires a learning task")
        return plan
    context = read_json(directory / "task.json")
    if context["phase"] != "learning" or context["generation_seed"] != 42:
        raise ValueError("Learning preparation uses the primary seed-42 task")
    source = Path(context["source_dir"]) if context.get("source_dir") else None
    if source:
        if (source / "experience.json").exists():
            raise ValueError("Do not change a closed source's frozen learning evidence")
        index = index or read_json(source / "source.json")["index_dir"]
    if not index:
        raise ValueError("Learning preparation requires its source index or --index")
    return prepare(directory, index, context["snapshot_path"], "learning",
                   task_ids=[context["public"]["task_id"]], source_dir=source,
                   intervention=context["intervention"])


def load(directory):
    p = Path(directory)
    plan = read_json(p / "evaluation.json" if p.is_dir() else p)
    if plan["seeds"] != SEEDS or set(plan["outputs"]) != {str(s) for s in SEEDS}:
        raise ValueError("Evaluation requires four declared generation seeds")
    key = plan_key(plan)
    if key != plan["key"] or any(set(rows) != set(plan["task_ids"]) for rows in plan["outputs"].values()):
        raise ValueError("Evaluation plan must retain its frozen identity and input membership")
    if plan["extended_from"] and sha256(plan["extended_from"]["path"]) != plan["extended_from"]["sha256"]:
        raise ValueError("Completion must retain its source evaluation binding")
    if plan.get("reuse_test"):
        origin = plan["reuse_test"]
        if sha256(origin["path"]) != origin["sha256"]:
            raise ValueError("Validation must retain its source test binding")
        test = load(origin["path"])
        if (plan["phase"] != "validation" or test["phase"] != "test"
                or any(plan[k] != test[k] for k in ("snapshot_id", "data_revision", "scorer", "seeds", "intervention"))):
            raise ValueError("Validation reuse must bind a compatible test")
        for seed in SEEDS:
            for tid in set(plan["task_ids"]) & set(test["task_ids"]):
                if plan["outputs"][str(seed)][tid] != test["outputs"][str(seed)][tid]:
                    raise ValueError("Validation must reuse the original test tasks and scores")
    if plan.get("reuse_learning"):
        seen = set()
        for ref in plan["reuse_learning"]:
            if sha256(ref["path"]) != ref["sha256"]:
                raise ValueError("Baseline must retain its initial report binding")
            source = report(ref["path"])
            if (plan["phase"] != "promotion" or source["phase"] != "learning"
                    or any(source[k] != plan[k] for k in ("snapshot_id", "data_revision", "scorer", "seeds", "intervention"))
                    or len(source["task_ids"]) != 1):
                raise ValueError("Baseline requires compatible initial learning reports")
            tid = source["task_ids"][0]
            if tid in seen or any(plan["outputs"][str(seed)].get(tid) != source["outputs"][str(seed)][tid] for seed in SEEDS):
                raise ValueError("Baseline must reuse the original initial outputs")
            seen.add(tid)
        if len(seen) != 11 or seen != set(plan["task_ids"]):
            raise ValueError("Baseline must cover the eleven round inputs")
    return plan


def extend(directory, destination):
    p = load(directory)
    if p["phase"] != "test":
        raise ValueError("Mode completion applies to held-out tests")
    if Path(destination).resolve() == Path(p["directory"]):
        raise ValueError("Full completion requires its own evaluation directory")
    return prepare(destination, p["index_dir"], p["snapshot_path"], p["phase"], mode="full",
                   intervention=p["intervention"], boundary=p["boundary"], extend_from=p["directory"])


def pending(directory):
    p = load(directory)
    modeling, waiting, completed = {}, [], []
    for tid, d in p["outputs"]["42"].items():
        task = Path(d)
        if not any((task / name).exists() for name in ("package", "h3-inputs", "output.lock.json")):
            modeling[tid] = d
    for seed, rows in p["outputs"].items():
        for tid, d in rows.items():
            (completed if (Path(d) / "output.lock.json").exists() else waiting).append({"task_id": tid, "seed": int(seed), "task_dir": d})
    return {"mode": p["mode"], "expected_generations": p["expected_generations"], "completed_generations": len(completed),
            "pending_generations": len(waiting), "modeling_tasks": modeling, "pending": waiting}


def is_completion(previous, current):
    fields = ("phase", "snapshot_id", "data_revision", "scorer", "seeds", "boundary")
    return (all(previous[k] == current[k] for k in fields) and previous["mode"] == "quick" and current["mode"] == "full"
            and all(current["outputs"][str(s)].get(t) == d for s in SEEDS for t, d in previous["outputs"][str(s)].items()))


def conditioning(intent):
    spec = intent["spec"]
    return {"files": {k: v for k, v in spec["files"].items() if k != "request.json"},
            "proxy_videos": spec["manifest"]["proxy_videos"], "binding": spec["binding"]}


def verify_seed(directory, seed, completed=False):
    directory = Path(directory)
    if generation_seed(read_json(directory / "task.json")) != seed:
        raise ValueError("Task seed must match the evaluation")
    intent = read_json(directory / "h3.submission.json")
    if intent["expected"]["seed"] != seed:
        raise ValueError("Submission seed must match the evaluation")
    if completed:
        done = read_json(directory / "h3-output/complete.json")
        if done.get("task_id") != intent["queue_id"] or done.get("metadata", {}).get("seed") != seed:
            raise ValueError("Completion must bind the submitted task and seed")
    return intent


def submit(directory, task_id, queue=None):
    from . import h3
    plan = load(directory)
    require_active_phase(plan["phase"])
    if task_id not in plan["task_ids"]:
        raise ValueError("Task must belong to this evaluation")
    primary = Path(plan["outputs"]["42"][task_id])
    package = primary / ("h3-inputs" if (primary / "h3.submission.json").exists() else "package")
    provenance = read_json(primary / "provenance.json")
    results, shared = [], None
    for seed in SEEDS:
        d = Path(plan["outputs"][str(seed)][task_id])
        if (d / "output.lock.json").exists():
            locked = verify_locked(d)
            if locked["status"] != "valid":
                results.append({"seed": seed, "task_dir": str(d), "status": "failed"})
                continue
            intent = verify_seed(d, seed, completed=True)
        else:
            intent = h3.submit(d, package, provenance, queue=queue)
        current = conditioning(intent)
        if shared is not None and current != shared:
            raise ValueError("Seed replicates require identical conditioning")
        shared = current
        results.append({"seed": seed, "task_dir": str(d), "queue_id": intent["queue_id"]})
    immutable_json(Path(plan["directory"]) / "conditioning" / (task_id + ".json"), shared)
    return results


def collect(directory):
    from .h3 import collect as gather
    plan = load(directory)
    results = []
    for s, rows in plan["outputs"].items():
        for t, d in rows.items():
            if (Path(d) / "output.lock.json").exists():
                status = {"status": "locked", "output": verify_locked(d)}
            else:
                status = gather(d) if (Path(d) / "h3.submission.json").exists() else {"status": "prepared"}
            results.append({"seed": s, "task_id": t, **status})
    return results


def score(directory):
    from .benchmark import _score_raw, BENCHMARK_ROOT, scorer_protocol
    d = Path(directory)
    context, locked = read_json(d / "task.json"), verify_locked(d)
    record = read_json(d / "scoring/score.json")
    if (record["snapshot_id"] != context["snapshot_id"] or record["data_revision"] != context["public"]["data_revision"]
            or record["task_context_sha256"] != digest(context) or record["output_lock_sha256"] != digest(locked)):
        raise ValueError("Score must bind the exact task context and locked output")
    component = record["components"]
    if sha256(component["path"]) != component["sha256"] or read_json(component["path"]) != component["values"]:
        raise ValueError("Score component evidence must retain its bound hash")
    from .benchmark import _verify_evidence
    _verify_evidence(component["values"].get("evidence"))
    binding = component["values"]["binding"]
    if (binding["task_context_sha256"] != digest(context) or binding["output_lock_sha256"] != digest(locked)
            or binding["protocol_sha256"] != digest(scorer_protocol()) or not record.get("session_id")):
        raise ValueError("Official measurement must bind this task, output, protocol, and independent session")
    public = context["public"]
    raw = component["values"]["raw"] if locked["status"] == "valid" else None
    scenario = "trimmed-" + public["scene_id"] + ".mp4"
    row = {"scenario": scenario, **{f"{k}_{public['view']}": v for k, v in (raw or {}).items()}}
    expected, _, _ = _score_raw([row], {(scenario, public["view"]): raw is not None}, BENCHMARK_ROOT)
    value = record["score"]
    if not isinstance(value, (int, float)) or not 0 <= value <= 1 or abs(value - expected["final_score_view"]) > EPSILON:
        raise ValueError("Official score must be within [0,1]")
    return record


def finalize(directory):
    plan = load(directory)
    snapshots.verify(plan["snapshot_path"])
    data.verify(plan["index_dir"], check_inputs=True)
    rows = {r["task_id"]: r for r in data.tasks(plan["index_dir"], plan["phase"])}
    scores, signatures, evidence, costs, categories, partitions = {}, {}, {}, [], {}, {}
    reused, incremental = [], []
    inherited = load(plan["extended_from"]["path"]) if plan["extended_from"] else None
    shared = {str(Path(d)) for rows in inherited["outputs"].values() for d in rows.values()} if inherited else set()
    test = load(plan["reuse_test"]["path"]) if plan.get("reuse_test") else None
    test_shared = {d for rows in test["outputs"].values() for d in rows.values()} if test else set()
    shared |= test_shared
    if plan.get("reuse_learning"):
        shared.update(d for outputs in plan["outputs"].values() for d in outputs.values())
    manifest = data.verify(plan["index_dir"])
    learning_scenes = manifest["binding"]["learning_views"]
    for seed in SEEDS:
        values = {}
        for tid in plan["task_ids"]:
            d = Path(plan["outputs"][str(seed)][tid])
            context = read_json(d / "task.json")
            public = context["public"]
            if str(d) in test_shared:
                public = {**public, "split": "validation"}
            elif plan.get("reuse_learning"):
                public = {**public, "split": "promotion"}
            if (context["snapshot_id"] != plan["snapshot_id"] or public != rows[tid]
                    or generation_seed(context) != seed or context["intervention"] != plan["intervention"]):
                raise ValueError("Evaluation task must match its frozen plan")
            record = score(d)
            if record["valid_output"]:
                sig = conditioning(verify_seed(d, seed, completed=True))
                if tid in signatures and signatures[tid] != sig:
                    raise ValueError("Seed replicates must share their frozen conditioning")
                signatures[tid] = sig
            values[tid] = record["score"]
            evidence[str(d / "scoring/score.json")] = sha256(d / "scoring/score.json")
            costs.append(d)
            (reused if str(d) in shared or (d / "reuse.json").is_file() else incremental).append(d)
        scores[str(seed)] = values
        for category in {rows[t]["category"] for t in values}:
            categories.setdefault(category, []).append(statistics.mean(v for t, v in values.items() if rows[t]["category"] == category))
        for label in ("learning", "heldout" if plan.get("population") == "heldout" else "validation"):
            selected = [v for t, v in values.items() if (rows[t]["scene_id"] in learning_scenes) == (label == "learning")]
            if selected:
                partitions.setdefault(label, []).append(statistics.mean(selected))
    means = [statistics.mean(scores[str(s)].values()) for s in SEEDS]
    record = {**plan, "count": len(plan["task_ids"]), "scores": scores, "per_seed": means,
              "mean": statistics.mean(means), "sample_std": statistics.stdev(means),
              "categories": {c: statistics.mean(v) for c, v in categories.items()},
              "partitions": {c: {"per_seed": v, "mean": statistics.mean(v), "sample_std": statistics.stdev(v)} for c, v in partitions.items()},
              "costs": total_cost(costs + [Path(plan["directory"])]), "evidence": evidence,
              "reused_generations": len(reused), "fresh_generations": len(incremental),
              "incremental_costs": total_cost(incremental + [Path(plan["directory"])]), "reused_costs": total_cost(reused)}
    origin = Path(plan["directory"]) / "reuse-source.json"
    if not origin.exists() and inherited:
        origin = Path(inherited["directory"]) / "reuse-source.json"
    if origin.exists():
        record["generation_origin"] = {**read_json(origin), "path": str(origin), "sha256": sha256(origin)}
    record["report_sha256"] = digest(record)
    return immutable_json(Path(plan["directory"]) / "report.json", record)


def report(path):
    p = Path(path)
    p = p / "report.json" if p.is_dir() else p
    result = read_json(p)
    if result.get("report_sha256") != digest({k: v for k, v in result.items() if k != "report_sha256"}):
        raise ValueError("Report content must match its immutable identity")
    plan = load(p.parent)
    if any(result.get(k) != v for k, v in plan.items()):
        raise ValueError("Report must preserve its frozen evaluation plan")
    for file, expected in result["evidence"].items():
        if sha256(file) != expected:
            raise ValueError("Report evidence must retain its bound hash")
    if result.get("generation_origin"):
        origin = result["generation_origin"]
        if sha256(origin["path"]) != origin["sha256"]:
            raise ValueError("Imported test origin must retain its recorded binding")
    return result


def compare(candidate, baseline):
    for field in ("data_revision", "scorer", "seeds", "views", "task_ids", "phase", "boundary"):
        if candidate[field] != baseline[field]:
            raise ValueError("Promotion requires compatible round promotion reports")
    if (candidate["phase"] != "promotion" or candidate["count"] != 11 or baseline["count"] != 11
            or len(set(candidate["task_ids"])) != 11):
        raise ValueError("Promotion requires all eleven round inputs")
    delta = candidate["mean"] - baseline["mean"]
    if abs(delta) > EPSILON:
        return {"accepted": delta > 0, "reason": "score", "gain": delta}
    for metric in ("h3_generations", "local_gpu_seconds", "model_tokens", "local_cpu_seconds"):
        if any(metric in r["costs"]["unknown_fields"] for r in (candidate, baseline)):
            return {"accepted": False, "reason": "cost_pending", "metric": metric}
        a, b = (r["costs"]["totals"][metric] / r["count"] for r in (candidate, baseline))
        if a != b:
            return {"accepted": a < b, "reason": "cost", "metric": metric, "candidate": a, "baseline": b}
    return {"accepted": False, "reason": "equivalent"}
