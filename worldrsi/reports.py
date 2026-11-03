"""Main-session experiment progress and separated learning/evaluation costs."""
from pathlib import Path
from .common import read_json
from .records import total_cost
from .workflow import pending_test, stopping


def summarize(run_dir):
    root = Path(run_dir)
    state = read_json(root / "run.json")
    groups = {k: [] for k in ("learning", "transfer", "validation", "test", "reused_test", "coordination")}
    for costs in root.rglob("costs"):
        parent = costs.parent
        context = parent / "task.json"
        phase = read_json(context)["phase"] if context.exists() else "coordination"
        if (parent / "evaluation.json").exists():
            phase = read_json(parent / "evaluation.json")["phase"]
        if (parent / "reuse.json").exists():
            phase = "reused_test"
        groups[phase].append(parent)
    rounds = []
    for r in state["rounds"]:
        sources = []
        for s in r["sources"]:
            d = Path(s["directory"])
            closure = read_json(d / "experience.json") if (d / "experience.json").exists() else None
            gate = read_json(d / "transfer/decision.json") if (d / "transfer/decision.json").exists() else None
            sources.append({"scene_id": s["scene_id"], "closure": closure and closure["status"],
                            "gain": closure and closure["gain"], "transfer": gate and gate["passed"]})
        rounds.append({"round": r["round"], "epoch": r["epoch"], "sources": sources, "decision": r["decision"]})
    return {"run_id": state["run_id"], "current_snapshot": state["current_snapshot"], "rounds": rounds,
            "mode": state["mode"], "initial_test": state.get("initial_test"), "initial_test_plan": state.get("initial_test_plan"),
            "tests": state["tests"], "pending_test": pending_test(state), "stopping": stopping(state),
            "costs": {k: total_cost(v) for k, v in groups.items()}}
