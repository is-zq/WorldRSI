"""File-backed commands for the Physics-IQ learning protocol."""
import argparse
import json
from pathlib import Path
import sys

from .common import ROOT, read_json


def parser():
    p = argparse.ArgumentParser(prog="wrsi")
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    d = commands.add_parser("data")
    d.add_argument("operation", choices=["prepare", "verify"])
    d.add_argument("directory", nargs="?", default=str(ROOT / "prepared/physics-iq"))
    s = commands.add_parser("snapshot")
    s.add_argument("operation", choices=["freeze", "verify"])
    s.add_argument("path", nargs="?")
    s.add_argument("--templates")
    s.add_argument("--manuals")
    r = commands.add_parser("run")
    r.add_argument("operation", choices=["init", "initial-validation", "initial-test", "register-initial-test", "set-reference", "begin", "manual-inputs", "publish-manuals", "candidate", "decide", "test", "register-test", "status", "history", "report"])
    r.add_argument("path")
    r.add_argument("--index")
    r.add_argument("--snapshot")
    r.add_argument("--policy")
    r.add_argument("--templates")
    r.add_argument("--report")
    r.add_argument("--mode", choices=["quick", "full"])
    s = commands.add_parser("source")
    s.add_argument("operation", choices=["progress", "branch", "close"])
    s.add_argument("directory")
    s.add_argument("--record")
    t = commands.add_parser("transfer")
    t.add_argument("operation", choices=["prepare", "finalize"])
    t.add_argument("source")
    e = commands.add_parser("eval")
    e.add_argument("operation", choices=["submit", "collect", "finalize", "extend", "status", "validation-from-test", "prepare-learning"])
    e.add_argument("directory")
    e.add_argument("--task-id")
    e.add_argument("--destination")
    e.add_argument("--index")
    a = commands.add_parser("agent")
    a.add_argument("operation", choices=["packet", "register", "usage"])
    a.add_argument("destination")
    for option in ("role", "task-dir", "evaluation-dir", "source-dir", "run-dir", "index", "session-id", "model", "effort", "record", "reason"):
        a.add_argument("--" + option)
    a.add_argument("--restart", action="store_true")
    s = commands.add_parser("score")
    s.add_argument("operation", choices=["sample", "batch"])
    s.add_argument("task_dir")
    s.add_argument("--definition")
    s.add_argument("--source")
    s.add_argument("--frames", nargs="+", type=int)
    s.add_argument("--reason")
    s.add_argument("--crop", nargs=4, type=int)
    h = commands.add_parser("h3")
    h.add_argument("operation", choices=["package", "submit", "collect"])
    h.add_argument("task_dir")
    h.add_argument("--render")
    h.add_argument("--modalities", nargs="+")
    h.add_argument("--prompt")
    h.add_argument("--package")
    h.add_argument("--provenance")
    h.add_argument("--queue")
    o = commands.add_parser("output")
    o.add_argument("operation", choices=["fail", "verify"])
    o.add_argument("task_dir")
    o.add_argument("--reason")
    for name in ("trace", "manual", "cost"):
        c = commands.add_parser(name)
        c.add_argument("operation", choices={"trace": ["before", "after"], "manual": ["record"], "cost": ["record", "total"]}[name])
        c.add_argument("directory")
        c.add_argument("--id")
        c.add_argument("--record")
        c.add_argument("--tool")
        c.add_argument("--kind", default="executor")
    t = commands.add_parser("tool")
    t.add_argument("--gpu", action="store_true")
    t.add_argument("--retry", action="store_true")
    t.add_argument("task_dir")
    t.add_argument("step_id")
    t.add_argument("argv", nargs=argparse.REMAINDER)
    return p


def execute(a):
    from . import agents, benchmark, data, evaluation, h3, learning, media, records, sessions, snapshots, workflow
    cmd = a.command
    if cmd == "doctor":
        runtime = read_json(ROOT / "config/runtime.json")
        paths = [runtime["physics_iq"], runtime["first_frames"], runtime["h3"]["queue"], runtime["h3"]["skill"], "/data/bin/godot", media.FFMPEG, media.FFPROBE]
        return {"protocol": runtime["protocol"], "root": str(ROOT), "paths": {s: Path(s).exists() for s in paths}}
    if cmd == "data":
        return data.prepare(a.directory) if a.operation == "prepare" else data.verify(a.directory, check_inputs=True)
    if cmd == "snapshot":
        return snapshots.freeze(a.path, templates=a.templates, manuals=a.manuals) if a.operation == "freeze" else snapshots.verify(a.path)
    if cmd == "run":
        if a.operation == "init": return workflow.init_run(a.path, a.index, a.snapshot, mode=a.mode or "full")
        if a.operation == "candidate": return workflow.candidate(a.path, a.policy, a.templates)
        if a.operation == "register-test": return workflow.register_test(a.path, a.report)
        if a.operation == "register-initial-test": return workflow.register_initial_test(a.path, a.report)
        if a.operation == "test": return workflow.prepare_test(a.path, a.mode)
        if a.operation == "status": return read_json(Path(a.path) / "run.json")
        if a.operation == "report":
            from .reports import summarize
            return summarize(a.path)
        return {"initial-validation": workflow.initial_reference, "initial-test": workflow.prepare_initial_test, "set-reference": workflow.set_reference,
                "begin": workflow.begin_round, "manual-inputs": workflow.manual_inputs,
                "publish-manuals": workflow.publish_manuals, "decide": workflow.decide,
                "test": workflow.prepare_test, "history": workflow.history}[a.operation](a.path)
    if cmd == "source":
        if a.operation == "progress": return learning.progress(a.directory)
        return {"branch": learning.branch, "close": learning.close}[a.operation](a.directory, read_json(a.record))
    if cmd == "transfer":
        return {"prepare": learning.prepare_transfer, "finalize": learning.finalize_transfer}[a.operation](a.source)
    if cmd == "eval":
        if a.operation == "prepare-learning": return evaluation.prepare_learning(a.directory, a.index)
        if a.operation == "submit": return evaluation.submit(a.directory, a.task_id)
        if a.operation == "validation-from-test": return evaluation.validation_from_test(a.directory, a.destination)
        if a.operation == "extend": return evaluation.extend(a.directory, a.destination)
        if a.operation == "status": return evaluation.pending(a.directory)
        return {"collect": evaluation.collect, "finalize": evaluation.finalize}[a.operation](a.directory)
    if cmd == "agent":
        if a.operation == "packet":
            return agents.packet(a.role, a.destination, task_dir=a.task_dir, source_dir=a.source_dir, run_dir=a.run_dir,
                                 index_dir=a.index, evaluation_dir=a.evaluation_dir, restart=a.restart, reason=a.reason)
        if a.operation == "register": return sessions.register(a.destination, a.session_id, a.model, a.effort)
        return agents.usage(a.destination, read_json(a.record))
    if cmd == "score":
        if a.operation == "batch":
            from .scoring import batch
            return batch(a.task_dir, a.definition)
        from .scoring import evidence
        return evidence(a.task_dir, a.source, a.frames, a.reason, a.crop)
    if cmd == "h3":
        if a.operation == "package": return h3.prepare_package(a.task_dir, a.render, a.modalities, a.prompt)
        if a.operation == "collect": return h3.collect(a.task_dir)
        return h3.submit(a.task_dir, a.package or Path(a.task_dir) / "package",
                         read_json(a.provenance or Path(a.task_dir) / "provenance.json"), a.queue)
    if cmd == "output":
        if a.operation == "verify": return media.verify_locked(a.task_dir)
        if not a.reason: raise ValueError("Output failure requires an execution reason")
        return media.lock_failure(a.task_dir, read_json(Path(a.task_dir) / "task.json")["public"], a.reason)
    if cmd == "trace":
        return {"before": records.before, "after": records.after}[a.operation](a.directory, a.id, **read_json(a.record))
    if cmd == "manual": return records.update_manual(a.directory, a.tool, read_json(a.record))
    if cmd == "cost":
        if a.operation == "total": return records.total_cost([a.directory])
        r = read_json(a.record)
        return records.cost(a.directory, a.id, a.kind, r["values"], r.get("details"))
    if cmd == "tool":
        argv = a.argv[1:] if a.argv[:1] == ["--"] else a.argv
        return records.run_tool(a.task_dir, a.step_id, argv, a.gpu, a.retry)
    raise ValueError("Choose an installed command")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "tools":
        from tools_runtime.cli import main as tool_main
        return tool_main(argv[1:])
    try:
        print(json.dumps(execute(parser().parse_args(argv)), ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError, StopIteration) as error:
        print(f"WorldRSI: {error}", file=sys.stderr)
        raise SystemExit(2)
