"""CLI entry point also embeddable under ``worldrsi tools``."""
import argparse
import json
from pathlib import Path

from . import runtime
from ._vendor import simulators
from ._vendor.common import read_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tools_runtime")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("scaffold", help="Create runtime plumbing with optional learned files")
    command.add_argument("destination")
    command.add_argument("--profile", help="Profile name from the selected template catalog")
    selection = command.add_mutually_exclusive_group()
    selection.add_argument("--templates", help="Complete template tree with godot/ and physics/")
    selection.add_argument("--task-dir", help="Use the templates in this task's frozen Agent snapshot")
    command = commands.add_parser("physics-new", help="Create a solver profile project")
    command.add_argument("destination")
    command.add_argument("--profile", required=True, help="Profile name from the selected template catalog")
    selection = command.add_mutually_exclusive_group()
    selection.add_argument("--templates", help="Complete template tree with godot/ and physics/")
    selection.add_argument("--task-dir", help="Use the templates in this task's frozen Agent snapshot")
    command = commands.add_parser("simulators", help="List installed solver entries and learned profiles")
    command.add_argument("backend", nargs="?", choices=tuple(simulators.BACKENDS))
    selection = command.add_mutually_exclusive_group()
    selection.add_argument("--templates", help="Read this template tree's profile catalog")
    selection.add_argument("--task-dir", help="Read this task's frozen template catalog")
    for name in ("simulate", "render"):
        command = commands.add_parser(name)
        command.add_argument("--request", required=True, help="JSON: project, frames, optional action/state/trajectory/modes")
        command.add_argument("--output", required=True)
    command = commands.add_parser("validate-render", help="Check a saved render bundle")
    command.add_argument("rendered", help="Output directory or render.json")
    command = commands.add_parser("first-frame", help="Normalize the public first frame")
    command.add_argument("source")
    command.add_argument("--output", required=True)
    command.add_argument("--estimate-depth", action="store_true", help="Explicitly run VGGT-Omega on the first frame")
    command = commands.add_parser("depth", help="Explicitly estimate first-frame depth with VGGT-Omega")
    command.add_argument("source")
    command.add_argument("--output", required=True)
    command = commands.add_parser("restore", help="Restore one saved physical state without advancing it")
    command.add_argument("--project", required=True)
    command.add_argument("--state", required=True)
    command.add_argument("--action")
    command.add_argument("--output", required=True)
    command.add_argument("--modes", nargs="+", required=True)
    args = parser.parse_args(argv)
    templates, runtime_assets = None, None
    if args.command in ("scaffold", "physics-new", "simulators"):
        templates = args.templates
        if args.task_dir:
            from worldrsi.snapshots import verify
            context = read_json(Path(args.task_dir) / "task.json")
            snapshot = verify(context["snapshot_path"])
            if snapshot["snapshot_id"] != context["snapshot_id"]:
                raise ValueError("Task snapshot identity mismatch")
            templates = Path(snapshot["path"]) / "source/templates"
            runtime_assets = Path(snapshot["path"]) / "source/tools_runtime/assets/godot"
    if args.command in ("scaffold", "physics-new"):
        result = runtime.scaffold(args.destination, args.profile, templates=templates, runtime_assets=runtime_assets)
    elif args.command == "simulators":
        result = simulators.describe(args.backend, templates=templates)
    elif args.command in ("simulate", "render"):
        result = getattr(runtime, args.command)(read_json(args.request), args.output)
    elif args.command == "validate-render":
        result = runtime.validate_render(args.rendered)
    elif args.command in ("first-frame", "depth"):
        result = runtime.prepare_first_frame(args.source, args.output,
            estimate_depth=args.command == "depth" or args.estimate_depth)
    elif args.command == "restore":
        result = runtime.restore(args.project, args.state, args.output, modes=args.modes, action=args.action)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
