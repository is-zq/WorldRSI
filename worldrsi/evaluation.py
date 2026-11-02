"""Seed-paired benchmark evaluations with frozen conditioning and resumable reports."""
from pathlib import Path
import statistics

from .common import ROOT, digest, immutable_json, read_json, sha256
from . import data, snapshots
from .records import total_cost
from .media import verify_locked

SEEDS = [42, 43, 44, 45]
EPSILON = 1e-12


def generation_seed(context):
    seed = context["generation_seed"]
    allowed = [42] if context["phase"] == "learning" else SEEDS
    if type(seed) is not int or seed not in allowed:
        raise ValueError("Generation seed must belong to the phase's frozen seed set")
    return seed


def create_task(directory, public, snapshot, *, seed=42, source_dir=None, intervention=None, modeling_task=None):
    snap = snapshots.verify(snapshot)
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


def prepare(directory, index, snapshot, phase, *, task_ids=None, intervention=None, boundary=None):
    directory, index = Path(directory).resolve(), Path(index).resolve()
    snap = snapshots.verify(snapshot)
    rows = data.tasks(index, phase)
    if phase == "transfer":
        if not task_ids or len(task_ids) != 2 or len(set(task_ids)) != 2:
            raise ValueError("Transfer evaluates the two reserved views of one scene")
        rows = [r for r in rows if r["task_id"] in task_ids]
        if len(rows) != 2 or len({r["scene_id"] for r in rows}) != 1:
            raise ValueError("Transfer inputs must share one learning scene")
    elif phase not in ("validation", "test") or len(rows) != {"validation": 99, "test": 198}[phase] or task_ids is not None:
        raise ValueError("Evaluation requires the complete phase input set")
    config = read_json(ROOT / "protocol/scorer.json")
    key = digest({"snapshot": snap["snapshot_id"], "revision": rows[0]["data_revision"],
                  "phase": phase, "ids": [r["task_id"] for r in rows], "intervention": intervention,
                  "seeds": SEEDS, "scorer": config, "boundary": boundary})
    outputs = {str(seed): {r["task_id"]: str(directory / "seeds" / str(seed) / r["task_id"]) for r in rows} for seed in SEEDS}
    record = {"key": key, "directory": str(directory), "index_dir": str(index), "phase": phase,
              "snapshot_id": snap["snapshot_id"], "snapshot_path": snap["path"],
              "data_revision": rows[0]["data_revision"], "scorer": config,
              "task_ids": [r["task_id"] for r in rows], "seeds": SEEDS, "outputs": outputs,
              "intervention": intervention, "boundary": boundary, "expected_generations": len(rows) * 4}
    immutable_json(directory / "evaluation.json", record)
    for seed in SEEDS:
        for public in rows:
            create_task(outputs[str(seed)][public["task_id"]], public, snap["path"], seed=seed, intervention=intervention,
                        modeling_task=outputs["42"][public["task_id"]] if seed != 42 else None)
    return record


def load(directory):
    p = Path(directory)
    plan = read_json(p / "evaluation.json" if p.is_dir() else p)
    if plan["seeds"] != SEEDS or set(plan["outputs"]) != {str(s) for s in SEEDS}:
        raise ValueError("Evaluation requires four declared generation seeds")
    return plan


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
    if task_id not in plan["task_ids"]:
        raise ValueError("Task must belong to this evaluation")
    primary = Path(plan["outputs"]["42"][task_id])
    package = primary / ("h3-inputs" if (primary / "h3.submission.json").exists() else "package")
    provenance = read_json(primary / "provenance.json")
    results, shared = [], None
    for seed in SEEDS:
        d = Path(plan["outputs"][str(seed)][task_id])
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
    return [{"seed": s, "task_id": t, **(gather(d) if (Path(d) / "h3.submission.json").exists() else {"status": "prepared"})}
            for s, rows in plan["outputs"].items() for t, d in rows.items()]


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
    manifest = data.verify(plan["index_dir"])
    learning_scenes = manifest["binding"]["learning_views"]
    for seed in SEEDS:
        values = {}
        for tid in plan["task_ids"]:
            d = Path(plan["outputs"][str(seed)][tid])
            context = read_json(d / "task.json")
            if (context["snapshot_id"] != plan["snapshot_id"] or context["public"] != rows[tid]
                    or context["intervention"] != plan["intervention"]):
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
        scores[str(seed)] = values
        for category in {rows[t]["category"] for t in values}:
            categories.setdefault(category, []).append(statistics.mean(v for t, v in values.items() if rows[t]["category"] == category))
        for label in ("learning", "validation"):
            selected = [v for t, v in values.items() if (rows[t]["scene_id"] in learning_scenes) == (label == "learning")]
            if selected:
                partitions.setdefault(label, []).append(statistics.mean(selected))
    means = [statistics.mean(scores[str(s)].values()) for s in SEEDS]
    record = {**plan, "count": len(plan["task_ids"]), "scores": scores, "per_seed": means,
              "mean": statistics.mean(means), "sample_std": statistics.stdev(means),
              "categories": {c: statistics.mean(v) for c, v in categories.items()},
              "partitions": {c: {"per_seed": v, "mean": statistics.mean(v), "sample_std": statistics.stdev(v)} for c, v in partitions.items()},
              "costs": total_cost(costs), "evidence": evidence}
    return immutable_json(Path(plan["directory"]) / "report.json", record)


def report(path):
    p = Path(path)
    p = p / "report.json" if p.is_dir() else p
    result = read_json(p)
    plan = load(p.parent)
    if any(result.get(k) != v for k, v in plan.items()):
        raise ValueError("Report must preserve its frozen evaluation plan")
    for file, expected in result["evidence"].items():
        if sha256(file) != expected:
            raise ValueError("Report evidence must retain its bound hash")
    return result


def compare(candidate, baseline):
    for field in ("data_revision", "scorer", "seeds", "task_ids", "phase"):
        if candidate[field] != baseline[field]:
            raise ValueError("Promotion requires compatible validation reports")
    if candidate["phase"] != "validation" or candidate["count"] != 99 or baseline["count"] != 99:
        raise ValueError("Promotion requires 99 validation inputs")
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
