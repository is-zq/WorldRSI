"""Role-specific packets for direct main-session dispatch."""
from pathlib import Path

from .common import ROOT, digest, immutable_json, read_json, sha256, write_text
from . import data, evaluation, learning, sessions, snapshots, workflow


def packet(role, destination, *, task_dir=None, source_dir=None, run_dir=None, index_dir=None,
           evaluation_dir=None, restart=False, reason=None):
    destination = Path(destination).resolve()
    runtime = read_json(ROOT / "config/runtime.json")
    scope = None
    instructions = ROOT / "prompts" / (role + ".md")
    settings = runtime["roles"][role]
    if role == "executor":
        c = read_json(Path(task_dir) / "task.json")
        evaluation.require_active_phase(c["phase"])
        snapshots.verify(c["snapshot_path"])
        instructions = Path(c["snapshot_path"]) / "source/prompts/executor.md"
        settings = c["roles"][role]
        if c.get("modeling_task"):
            raise ValueError("Executor modeling uses the primary seed task")
        payload = {k: c[k] for k in ("public", "phase", "snapshot_path", "policy_snapshot", "manual_snapshot", "template_snapshot", "intervention")}
        payload["task_dir"] = str(Path(task_dir).resolve())
        policy = Path(c["snapshot_path"]) / "policy"
        if Path(c["policy_snapshot"]).resolve() != policy.resolve():
            raise ValueError("Executor policy must belong to its frozen snapshot")
        payload["decision_experience"] = snapshots.decision_experience(policy)
        payload["tool_documentation"] = str(Path(c["snapshot_path"]) / "source/docs/tools.md")
    elif role == "scorer":
        if bool(task_dir) == bool(evaluation_dir):
            raise ValueError("Scorer requires one learning attempt or one complete evaluation")
        target = Path(task_dir or evaluation_dir).resolve()
        plan = evaluation.prepare_learning(target, index_dir) if task_dir else evaluation.load(target)
        evaluation.require_active_phase(plan["phase"])
        if plan["phase"] == "learning":
            context = read_json(target / "task.json")
            scope = Path(context["source_dir"]).resolve() if context.get("source_dir") else target
        else:
            scope = target
        payload = {"evaluation_dir": str(target), "evaluation_key": plan["key"],
                   "cost_directory": str(target), "scorer_scope": str(scope)}
    elif role == "reflector":
        scope = Path(source_dir).resolve()
        if (scope / "experience.json").exists():
            raise ValueError("Reflector deliveries belong to an open learning source")
        source = read_json(scope / "source.json")
        public, ground_truth = data.reflection_inputs(source["index_dir"], source["task_id"])
        if (public["scene_id"] != source["scene_id"]
                or read_json(scope / "initial/task.json")["public"] != public):
            raise ValueError("Reflector inputs must match the source's initial public task")
        prior = read_json(source["history"]["path"])
        if digest(prior) != source["history"]["sha256"]:
            raise ValueError("Reflector history must match its batch-start binding")
        p = learning.progress(scope)
        directories = {"initial": scope / "initial", **{b["id"]: Path(b["directory"]) for b in source["branches"]}}
        summaries = {}
        for bid, d in directories.items():
            score = p["attempts"].get(bid)
            feedback = ({k: score[k] for k in ("score", "per_seed", "seeds", "outputs", "evidence", "report", "valid_output") if k in score}
                        if score is not None else None)
            if score and score.get("components"):
                feedback["components"] = {k: score["components"][k] for k in ("path", "sha256")}
            summaries[bid] = {"score": feedback, "decisions": [
                {"before": read_json(f), "after": read_json(f.with_name("after.json")) if f.with_name("after.json").exists() else None}
                for f in sorted((d / "trace").glob("*/before.json"))], "task_dir": str(d)}
        payload = {"source_dir": str(scope), "progress": {k: v for k, v in p.items() if k != "attempts"},
                   "attempts": summaries, "history": prior, "public": public, "ground_truth": ground_truth,
                   "cost_directory": str(scope / "reflection")}
    elif role in ("editor", "manual_manager"):
        scope = Path(run_dir).resolve()
        state = read_json(scope / "run.json")
        current = state["rounds"][-1]
        workflow.completed_sources(current)
        if role == "manual_manager":
            payload = workflow.manual_inputs(scope)
        else:
            if current["manuals"] is None:
                raise ValueError("Editor requires the staged manual publication")
            payload = {"round": current["round"], "policy": str(Path(current["snapshot_path"]) / "policy"),
                       "templates": str(Path(current["snapshot_path"]) / "source/templates"),
                       "manuals": current["manuals"]["output_manuals"],
                       "experiences": workflow.editor_experiences(scope),
                       "history": {"promotion": workflow.history(scope)["promotion"]},
                       "output_directory": str(Path(current["directory"]) / "editor")}
        payload["cost_directory"] = str(Path(current["directory"]) / role)
    else:
        raise ValueError("Role must be one of the configured experiment roles")
    immutable_json(destination / "input.json", payload)
    write_text(destination / "instructions.md", instructions.read_text())
    if (destination / "dispatch.json").exists():
        return read_json(destination / "dispatch.json")
    message = f"Read {destination / 'instructions.md'} and {destination / 'input.json'}. Execute the supplied role task and persist its artifacts."
    spec = sessions.dispatch(role, role + "_" + digest(str(destination))[:12], message, settings,
                             scope=scope, restart=restart, reason=reason)
    return immutable_json(destination / "dispatch.json", spec)


def usage(packet_dir, record):
    d = Path(packet_dir)
    payload, spec, session = (read_json(d / name) for name in ("input.json", "dispatch.json", "session.json"))
    return sessions.record_usage(d, record, payload.get("task_dir", payload.get("cost_directory", str(d))), spec["role"], session)
