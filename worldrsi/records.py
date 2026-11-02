"""Serialized decision traces, local resource accounting and tool experience writes."""
from __future__ import annotations

import os
import functools
from pathlib import Path
import resource
import subprocess
import time
import uuid

from .common import ROOT, digest, identifier, immutable_json, lock, now, read_json, sha256, write_json, finite_nonnegative


def validate_decision(decision):
    if not isinstance(decision, dict) or set(decision) != {"factors", "choice", "constraints"}:
        raise ValueError("Decision metadata requires factors, choice and constraints")
    factors = decision["factors"]
    if (not isinstance(factors, list) or not factors or
            any(factor not in ("P", "R", "M") for factor in factors) or len(set(factors)) != len(factors)):
        raise ValueError("Decision factors must be distinct names from P, R and M")
    if not isinstance(decision["choice"], str) or not decision["choice"].strip():
        raise ValueError("Record the actual P/R/M choice")
    if (not isinstance(decision["constraints"], list) or
            any(not isinstance(value, str) or not value.strip() for value in decision["constraints"])):
        raise ValueError("Decision constraints must be a list of explicit conditions")
    return decision


def before(task_dir, step_id, evidence, reason, local_expectation, downstream_expectation,
           *, decision=None, decision_refs=None):
    if not all((reason, local_expectation, downstream_expectation)):
        raise ValueError("Record the reason and both expectations before acting")
    record = {"step_id": identifier(step_id), "evidence": evidence, "reason": reason,
              "local_expectation": local_expectation, "downstream_expectation": downstream_expectation}
    if decision is not None:
        record["decision"] = validate_decision(decision)
    if decision_refs is not None:
        if (not isinstance(decision_refs, list) or
                any(not isinstance(ref, str) for ref in decision_refs) or len(set(decision_refs)) != len(decision_refs)):
            raise ValueError("decision_refs must be a list of distinct prior decision step IDs")
        bindings = {}
        for ref in decision_refs:
            if identifier(ref) == step_id:
                raise ValueError("A decision step is linked to itself automatically")
            prior = read_json(Path(task_dir) / "trace" / ref / "before.json")
            validate_decision(prior.get("decision"))
            bindings[ref] = digest(prior)
        record.update(decision_refs=decision_refs, decision_sha256=bindings)
    path = Path(task_dir) / "trace" / step_id / "before.json"
    if path.exists():
        existing = read_json(path)
        if {k: value for k, value in existing.items() if k != "recorded_at"} != record:
            raise ValueError("An existing decision cannot be rewritten")
        return existing
    return immutable_json(path, {**record, "recorded_at": now()})


def after(task_dir, step_id, action, artifacts, checks, local_goal_met, modifications=None):
    step = Path(task_dir) / "trace" / identifier(step_id)
    decision = read_json(step / "before.json")
    if not action or not isinstance(local_goal_met, bool):
        raise ValueError("Actual action and local goal status required")
    checked = []
    for check in checks:
        if not check.get("name") or not isinstance(check.get("passed"), bool) or not check.get("supports"):
            raise ValueError("Checks require name, passed and the limited claim they support")
        if not check.get("evidence"):
            raise ValueError("Process exit status alone is not a tool check")
        evidence = [{"path": str(Path(p).resolve()), "sha256": sha256(p)} for p in check["evidence"]]
        checked.append({**check, "evidence": evidence})
    if local_goal_met and (not checked or not all(c["passed"] for c in checked)):
        raise ValueError("A successful local stage needs successful artifact checks")
    payload = {"step_id": step_id, "before_sha256": digest(decision), "action": action,
               "artifacts": [{"path": str(Path(p).resolve()), "sha256": sha256(p)} for p in artifacts],
               "checks": checked, "local_goal_met": local_goal_met, "modifications": modifications or []}
    return immutable_json(step / "after.json", payload)


def cost(task_dir, event_id, kind, values, details=None):
    allowed = {"h3_generations", "local_gpu_seconds", "model_tokens", "local_cpu_seconds", "wall_seconds", "tool_retries", "input_tokens", "output_tokens", "cached_input_tokens"}
    if set(values) - allowed:
        raise ValueError("Unsupported cost field; H3 server resource cost is deliberately excluded")
    for name, value in values.items():
        if value is not None:
            finite_nonnegative(value, name)
    record = {"event_id": identifier(event_id), "kind": kind, "values": values, "details": details or {}}
    return immutable_json(Path(task_dir) / "costs" / f"{event_id}.json", record)


def total_cost(task_dirs):
    total = {k: 0 for k in ("h3_generations", "local_gpu_seconds", "model_tokens", "local_cpu_seconds", "wall_seconds", "tool_retries")}
    unknown, events, kinds = set(), {}, {}
    for directory in task_dirs:
        local_events = []
        for path in sorted((Path(directory) / "costs").glob("*.json")):
            record = read_json(path)
            local_events.append(record)
            # Reused completions are counted once globally, including across tasks.
            key = record["event_id"] if record["kind"] == "h3" else str(path.resolve())
            if key in events:
                if events[key] != record:
                    raise ValueError("Conflicting cost event")
                continue
            events[key] = record
            kinds[record["kind"]] = kinds.get(record["kind"], 0) + 1
            for field, value in record["values"].items():
                if value is None:
                    unknown.add(field)
                else:
                    total[field] = total.get(field, 0) + value
        if (Path(directory) / "task.json").exists():
            # An absent usage report is not a free model call.
            replica = read_json(Path(directory) / "task.json").get("modeling_task")
            if not replica and not any(r["kind"] == "executor" and "model_tokens" in r["values"] for r in local_events):
                unknown.add("model_tokens")
            if (Path(directory) / "scoring/score.json").exists() and not any(r["kind"] == "scorer" and "model_tokens" in r["values"] for r in local_events):
                unknown.add("model_tokens")
            if not any("local_cpu_seconds" in r["values"] for r in local_events):
                unknown.add("local_cpu_seconds")
            if not any("local_gpu_seconds" in r["values"] for r in local_events):
                unknown.add("local_gpu_seconds")
    return {"totals": total, "unknown_fields": sorted(unknown), "events": len(events), "events_by_kind": kinds,
            "excludes": "H3 service internal attempts and resource usage"}


def metered(kind="scorer"):
    """Record actual helper CPU/wall cost, including failed and repeated observations."""
    def decorate(function):
        @functools.wraps(function)
        def invoke(task_dir, *args, **kwargs):
            directory = Path(task_dir).resolve()
            if os.environ.get("WORLDRSI_METERED_TASK") == str(directory):
                return function(task_dir, *args, **kwargs)
            def cpu():
                values = [resource.getrusage(who) for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
                return sum(u.ru_utime + u.ru_stime for u in values)
            start, usage, failed = time.monotonic(), cpu(), True
            try:
                result = function(task_dir, *args, **kwargs)
                failed = False
                return result
            finally:
                cost(directory, "measure-" + uuid.uuid4().hex, kind,
                     {"local_cpu_seconds": max(0, cpu() - usage), "wall_seconds": time.monotonic() - start,
                      "local_gpu_seconds": 0}, {"helper": function.__module__ + "." + function.__name__, "failed": failed})
        return invoke
    return decorate


def run_tool(task_dir, step_id, argv, gpu=False, retry=False, *, measurement=None):
    """Bounded helper invocation; long jobs are launched by Codex in named tmux."""
    directory = Path(task_dir)
    if measurement is not None:
        from .learning import COST_SCOPES
        if (set(measurement) != {"measurement_scope", "cache_mode", "comparison_role", "comparison_attempt"}
                or measurement["measurement_scope"] not in COST_SCOPES
                or measurement["cache_mode"] not in ("cold", "warm")
                or measurement["comparison_role"] not in ("baseline", "candidate")):
            raise ValueError("Cost measurement needs scope, cache mode, comparison role and attempt")
        identifier(measurement["comparison_attempt"])
    read_json(directory / "trace" / identifier(step_id) / "before.json")
    with lock(directory / "trace" / step_id / ".command.lock"):
        existing = list((directory / "trace" / step_id).glob("command-*.json"))
        index = len(existing) + 1
        event = f"tool-{step_id}-{index:04d}"
        log_path = directory / "logs" / f"{event}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        with log_path.open("w") as log:
            try:
                result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=False,
                                        env={**os.environ, "WORLDRSI_METERED_TASK": str(directory.resolve())})
                code, error = result.returncode, None
            except OSError as e:
                code, error = 127, str(e)
                log.write(error + "\n")
        elapsed = time.monotonic() - started
        ended = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = ended.ru_utime + ended.ru_stime - usage.ru_utime - usage.ru_stime
        details = {"argv": argv, "returncode": code, "error": error, "log": str(log_path.resolve()),
                   "gpu_measurement": "local single-GPU allocation wall seconds" if gpu else "no GPU allocation"}
        if measurement is not None:
            details.update(measurement)
        cost(directory, event, "tool", {"local_cpu_seconds": cpu, "local_gpu_seconds": elapsed if gpu else 0,
                                       "wall_seconds": elapsed, "tool_retries": int(retry)}, details)
        write_json(directory / "trace" / step_id / f"command-{index:04d}.json", details)
        return details


def _manual_record(task_dir, tool, entry):
    task_dir = Path(task_dir).resolve()
    context = read_json(task_dir / "task.json")
    if context["phase"] != "learning":
        raise ValueError("Manual updates belong to learning execution")
    if context.get("role", "executor") != "executor":
        raise ValueError("Only Executor may update tool manuals")
    required = ("id", "scope", "operation", "evidence", "claim_review")
    if any(not entry.get(k) for k in required):
        raise ValueError("Manual entry requires applicability, operation, evidence and claim review")
    identifier(tool)
    identifier(entry["id"])
    refs = []
    for ref in entry["evidence"]:
        step = task_dir / "trace" / identifier(ref["step_id"]) / "after.json"
        result = read_json(step)
        if not result["local_goal_met"]:
            raise ValueError("Unconfirmed tool operation cannot become successful experience")
        checks = {c["name"]: c for c in result["checks"]}
        if not ref.get("checks") or any(name not in checks or not checks[name]["passed"] for name in ref["checks"]):
            raise ValueError("Entry must cite passed artifact checks")
        for name in ref["checks"]:
            for item in checks[name]["evidence"]:
                if sha256(item["path"]) != item["sha256"]:
                    raise ValueError("Manual evidence changed")
        refs.append({"record": str(step.resolve()), "sha256": sha256(step), "checks": ref["checks"]})
    return {"tool": tool, "entry": entry, "task_dir": str(task_dir),
            "task_id": context["public"]["task_id"], "snapshot_id": context["snapshot_id"],
            "task_sha256": sha256(task_dir / "task.json"), "evidence": refs}


def verify_manual_update(path):
    path = Path(path).resolve()
    record = read_json(path)
    directory = Path(record["task_dir"])
    expected = directory / "manual-updates" / identifier(record["tool"]) / (identifier(record["entry"]["id"]) + ".json")
    if path != expected or record != _manual_record(directory, record["tool"], record["entry"]):
        raise ValueError("Manual update or its execution evidence changed")
    return record


def update_manual(task_dir, tool, entry, root=ROOT):
    """Persist checked task-local experience; Manual Manager edits the next manuals."""
    task_dir = Path(task_dir).resolve()
    context = read_json(task_dir / "task.json")
    source_dir = Path(context["source_dir"]) if context.get("source_dir") else None
    with lock(source_dir / ".source.lock" if source_dir else task_dir / ".manuals.lock"):
        if source_dir:
            source = read_json(source_dir / "source.json")
            allowed = {source_dir / "initial", *(Path(item["directory"]) for item in source["branches"])}
            if task_dir not in allowed or (source_dir / "experience.json").exists():
                raise ValueError("Manual updates require an open learning source and its registered task")
        record = _manual_record(task_dir, tool, entry)
        path = task_dir / "manual-updates" / tool / (entry["id"] + ".json")
        immutable_json(path, record)
    refs = record["evidence"]
    return {"path": str(path), "sha256": sha256(path), "evidence": refs}
