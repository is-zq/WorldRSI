"""Persist role-session bindings; dispatch remains a main-Codex tool operation."""
from pathlib import Path

from .common import digest, finite_nonnegative, immutable_json, lock, read_json, sha256, write_json


def dispatch(role, task_name, message, settings, *, scope=None, restart=False, reason=None):
    if restart and (scope is None or not reason or not reason.strip()):
        raise ValueError("Restarting a persistent role requires its scope and a reason")
    record = {"role": role, "task_name": task_name, "model": settings["model"],
              "reasoning_effort": settings["model_reasoning_effort"]}
    existing = None
    if scope is not None:
        scope = Path(scope).resolve()
        path = scope / "sessions" / f"{role}.json"
        if path.exists():
            existing = read_json(path)
            if existing["model"] != record["model"] or existing["reasoning_effort"] != record["reasoning_effort"]:
                raise ValueError("Persistent session model/effort differs from the configured role")
        generation = existing["generation"] if existing else 1
        record["session_scope"] = {"directory": str(scope), "role": role,
            "generation": generation + int(bool(existing and restart)),
            "previous_session_id": existing["session_id"] if existing else None,
            "previous_generation": existing["generation"] if existing else None}
        if restart:
            task_name = f"{task_name}_g{record['session_scope']['generation']}"
            record["task_name"] = task_name
    if existing and not restart:
        record.update(method="followup_task", arguments={"target": existing["session_id"], "message": message})
    else:
        record.update(method="spawn_agent", arguments={"task_name": task_name, "fork_turns": "none",
                      "model": record["model"], "reasoning_effort": record["reasoning_effort"], "message": message})
    if restart:
        record["restart_reason"] = reason
    return record


def register(packet_dir, session_id, model, effort):
    packet_dir = Path(packet_dir).resolve()
    spec = read_json(packet_dir / "dispatch.json")
    if model != spec["model"] or effort != spec["reasoning_effort"]:
        raise ValueError("Actual subagent settings differ from the frozen role configuration")
    if not isinstance(session_id, str) or not session_id.strip() or any(ord(c) < 32 for c in session_id):
        raise ValueError("A real, nonempty subagent session identifier is required")
    base = {"session_id": session_id, "model": model, "model_reasoning_effort": effort,
            "dispatch_sha256": sha256(packet_dir / "dispatch.json")}
    scope = spec.get("session_scope")
    if not scope:
        return immutable_json(packet_dir / "session.json", base)
    directory = Path(scope["directory"]) / "sessions"
    registry = directory / f"{scope['role']}.json"
    with lock(directory / f".{scope['role']}.lock"):
        current = read_json(registry) if registry.exists() else None
        session_file = packet_dir / "session.json"
        if session_file.exists():
            saved = read_json(session_file)
            if any(saved.get(key) != value for key, value in base.items()):
                raise ValueError("Packet already registered with a different session")
            # Recover a crash after saving the packet binding but before updating the registry.
            if current and (current["generation"], current["delivery_index"]) >= (saved["generation"], saved["delivery_index"]):
                return saved
        else:
            actual_previous = (current["session_id"], current["generation"]) if current else (None, None)
            expected_previous = (scope["previous_session_id"], scope["previous_generation"])
            if actual_previous != expected_previous:
                raise ValueError("Role session changed after packet preparation; prepare a fresh packet")
            if spec["method"] == "followup_task" and session_id != spec["arguments"]["target"]:
                raise ValueError("Follow-up must use the registered role session")
            if spec["method"] == "spawn_agent" and current and session_id == current["session_id"]:
                raise ValueError("A replacement must have a new session identifier")
            index = current["delivery_index"] + 1 if spec["method"] == "followup_task" else 1
            saved = immutable_json(session_file, {**base, "scope": scope["directory"], "role": scope["role"],
                                   "generation": scope["generation"], "delivery_index": index})
        write_json(registry, {"session_id": session_id, "role": scope["role"], "model": model,
                             "reasoning_effort": effort, "generation": saved["generation"],
                             "delivery_index": saved["delivery_index"], "last_packet": str(packet_dir)})
        return saved


def record_usage(packet_dir, usage, destination, role, session):
    """Charge each delivery once, converting cumulative session counters to deltas."""
    from .records import cost
    packet_dir, destination = Path(packet_dir).resolve(), Path(destination).resolve()
    directory = Path(session["scope"]) / "sessions" if session.get("scope") else packet_dir
    role_lock = directory / f".{role}.lock"
    state_path = directory / "usage" / (digest(session["session_id"]) + ".json")
    with lock(role_lock):
        receipt = packet_dir / "usage.json"
        if receipt.exists():
            previous = read_json(receipt)
            if previous["event"]["details"]["reported_usage"] != usage or previous["cost_directory"] != str(destination):
                raise ValueError("Usage for this packet was already recorded differently")
            if previous.get("usage_state"):
                state = read_json(state_path) if state_path.exists() else None
                if not state or state["delivery_index"] < previous["usage_state"]["delivery_index"]:
                    write_json(state_path, previous["usage_state"])
            return previous["event"]
        mode = usage.get("scope", "packet")
        if mode not in ("packet", "cumulative"):
            raise ValueError("Usage scope must be packet or cumulative")
        incoming, outgoing = usage.get("input_tokens"), usage.get("output_tokens")
        total = usage.get("total_tokens")
        if total is None and incoming is not None and outgoing is not None:
            total = incoming + outgoing
        counters = {"input_tokens": incoming, "output_tokens": outgoing, "model_tokens": total,
                    "cached_input_tokens": usage.get("cached_input_tokens")}
        for key, value in counters.items():
            if value is not None:
                finite_nonnegative(value, key)
        index = session.get("delivery_index", 1)
        state = read_json(state_path) if state_path.exists() else None
        if state and state["delivery_index"] >= index:
            raise ValueError("Record usage in delivery order")
        baseline = {key: 0 for key in counters} if index == 1 else (
            state["cumulative"] if state and state["delivery_index"] == index - 1 else {key: None for key in counters})
        values, cumulative = {}, {}
        for key, current in counters.items():
            prior = baseline.get(key)
            if mode == "cumulative":
                if current is not None and prior is not None and current < prior:
                    raise ValueError("Cumulative session usage decreased")
                values[key] = current - prior if current is not None and prior is not None else None
                cumulative[key] = current
            else:
                values[key] = current
                cumulative[key] = prior + current if prior is not None and current is not None else None
        new_state = {"delivery_index": index, "cumulative": cumulative}
        event_id = "agent-" + digest({"session": session, "role": role})[:32]
        event = cost(destination, event_id, role, values, {"session": session, "packet": str(packet_dir),
                     "reported_usage": usage, "usage_scope": mode})
        immutable_json(receipt, {"cost_directory": str(destination), "event": event, "usage_state": new_state})
        write_json(state_path, new_state)
        return event
