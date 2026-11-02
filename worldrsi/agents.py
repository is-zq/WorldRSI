"""Role-specific packets for direct main-session dispatch."""
from pathlib import Path

from .common import ROOT, digest, immutable_json, read_json, sha256, write_text
from . import benchmark, learning, sessions, snapshots, workflow


def packet(role, destination, *, task_dir=None, source_dir=None, run_dir=None, index_dir=None,
           restart=False, reason=None):
    destination = Path(destination).resolve()
    runtime = read_json(ROOT / "config/runtime.json")
    scope = None
    instructions = ROOT / "prompts" / (role + ".md")
    settings = runtime["roles"][role]
    if role == "executor":
        c = read_json(Path(task_dir) / "task.json")
        snapshots.verify(c["snapshot_path"])
        instructions = Path(c["snapshot_path"]) / "source/prompts/executor.md"
        settings = c["roles"][role]
        if c.get("modeling_task"):
            raise ValueError("Executor modeling uses the primary seed task")
        payload = {k: c[k] for k in ("public", "phase", "snapshot_path", "policy_snapshot", "manual_snapshot", "template_snapshot", "intervention")}
        payload["task_dir"] = str(Path(task_dir).resolve())
        payload["tool_documentation"] = str(Path(c["snapshot_path"]) / "source/docs/tools.md")
    elif role == "scorer":
        payload = benchmark.scorer_payload(task_dir, index_dir)
    elif role == "reflector":
        scope = Path(source_dir).resolve()
        if (scope / "experience.json").exists():
            raise ValueError("Reflector deliveries belong to an open learning source")
        source = read_json(scope / "source.json")
        prior = read_json(source["history"]["path"])
        if digest(prior) != source["history"]["sha256"]:
            raise ValueError("Reflector history must match its batch-start binding")
        p = learning.progress(scope)
        directories = {"initial": scope / "initial", **{b["id"]: Path(b["directory"]) for b in source["branches"]}}
        summaries = {}
        for bid, d in directories.items():
            score = p["attempts"].get(bid)
            feedback = ({k: score[k] for k in ("score", "valid_output", "observations", "uncertainties") if k in score}
                        if score is not None else None)
            if score and score.get("components"):
                feedback["components"] = {k: score["components"][k] for k in ("path", "sha256")}
            summaries[bid] = {"score": feedback, "decisions": [
                {"before": read_json(f), "after": read_json(f.with_name("after.json")) if f.with_name("after.json").exists() else None}
                for f in sorted((d / "trace").glob("*/before.json"))], "task_dir": str(d)}
        payload = {"source_dir": str(scope), "progress": {k: v for k, v in p.items() if k != "attempts"},
                   "attempts": summaries, "history": prior,
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
                       "experiences": [b for s in current["sources"] if (b := learning.accepted_bundle(s["directory"]))],
                       "history": workflow.history(scope), "output_directory": str(Path(current["directory"]) / "editor")}
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
