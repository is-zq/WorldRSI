#!/usr/bin/env python3
"""Build a portable, read-only review of a WorldRSI round and its validation."""
import argparse
import ast
import concurrent.futures
import difflib
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT, HEADER = 480, 280, 88
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SEEDS = [42, 43, 44, 45]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


class Report:
    def __init__(self, args):
        self.root = Path(args.project).resolve()
        self.run = Path(args.run).resolve()
        self.round = self.run / "rounds" / f"r{args.round:03d}"
        self.out = Path(args.output).resolve()
        self.ffmpeg = args.ffmpeg
        self.jobs = args.jobs
        self.summaries = read(args.summaries)
        self.media = self.out / "media"
        self.cache = self.out / "build-cache"
        for p in [self.out, self.media, self.cache, self.out / "evidence"]:
            p.mkdir(parents=True, exist_ok=True)
        self.private = {}
        for split in ["learning", "validation"]:
            for line in (self.root / f"prepared/physics-iq/private/{split}.jsonl").read_text().splitlines():
                item = json.loads(line)
                self.private[item["task_id"]] = item
        self.gt_jobs = {}

    def local(self, path):
        """Frozen records in a copied run still refer to the original checkout."""
        p = Path(path)
        for prefix in [str(self.root), "/data/WorldRSI"]:
            try:
                return self.root / p.relative_to(prefix)
            except ValueError:
                pass
        return p

    def evidence(self, path, name):
        dest = self.out / "evidence" / name
        shutil.copyfile(path, dest)
        return "evidence/" + name

    def arm(self, directory, task_id, label):
        directory = self.local(directory)
        report = read(directory / "report.json")
        videos, scores, bindings = {}, {}, {}
        for seed in SEEDS:
            k = str(seed)
            output = self.local(report["outputs"][k][task_id])
            lock = read(output / "output.lock.json")
            if lock["status"] != "valid":
                raise ValueError(f"Invalid locked output: {output}")
            video = self.local(lock["path"])
            if not video.is_file():
                raise FileNotFoundError(video)
            score = read(output / "scoring/score.json")
            value = report["scores"][k][task_id]
            if not math.isclose(score["score"], value, abs_tol=1e-12, rel_tol=0):
                raise ValueError(f"Score binding mismatch: {output}")
            lock_digest = hashlib.sha256(json.dumps(lock, sort_keys=True, ensure_ascii=False,
                                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()
            if score["output_lock_sha256"] != lock_digest:
                raise ValueError(f"Output lock changed: {output}")
            videos[k], scores[k] = str(video), value
            bindings[k] = {"video": str(video), "sha256": lock["sha256"],
                           "score_file": str(output / "scoring/score.json"),
                           "output_lock": str(output / "output.lock.json")}
        result = {"label": label, "scores": scores, "mean": sum(scores.values()) / 4,
                  "videos": videos, "bindings": bindings}
        h = directory / "hypothesis.json"
        if h.exists():
            result["hypothesis"] = read(h)
        return result

    def gt(self, task_id):
        item = self.private[task_id]
        source = self.local(item["gt_v1"])
        if not source.is_file():
            raise FileNotFoundError(source)
        target = self.cache / f"{task_id}-gt1.mp4"
        self.gt_jobs[str(target)] = (source, target)
        return [{"label": "GT take 1", "scores": None, "mean": None,
                 "videos": {str(s): str(target) for s in SEEDS},
                 "source": str(source), "sha256": item["sha256"]["gt_v1"]}]

    def collect_editor(self):
        directory = self.round / "editor"
        packet = self.run / "coordination" / f"editor-{self.round.name}" / "input.json"
        inputs = read(packet)
        manifest = read(directory / "manifest.json")
        if hashlib.sha256(packet.read_bytes()).hexdigest() != manifest["input_sha256"]:
            raise ValueError("Editor input does not match its final manifest")
        for component in ["policy", "templates"]:
            for relative, expected in manifest[component].items():
                if hashlib.sha256((directory / component / relative).read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Editor artifact changed: {component}/{relative}")
        for name in ["coverage", "validation"]:
            if hashlib.sha256((directory / f"{name}.json").read_bytes()).hexdigest() != manifest[f"{name}_sha256"]:
                raise ValueError(f"Editor evidence changed: {name}")
        relative = "skills/worldrsi-decision/SKILL.md"
        before = self.local(inputs["policy"]) / relative
        after = directory / "policy" / relative
        old_text, new_text = before.read_text(), after.read_text()
        helpers = directory / "templates/physics/decision_models.py"
        helper_text = helpers.read_text()
        functions = [n.name for n in ast.parse(helper_text).body
                     if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
        old_templates = self.local(inputs["templates"])
        template_changes = []
        for name, value in manifest["templates"].items():
            original = old_templates / name
            status = "added" if not original.exists() else (
                "unchanged" if hashlib.sha256(original.read_bytes()).hexdigest() == value else "modified")
            template_changes.append({"path": name, "status": status})
        diff = "".join(difflib.unified_diff(old_text.splitlines(True), new_text.splitlines(True),
                                           fromfile="before/skills/worldrsi-decision/SKILL.md",
                                           tofile="after/skills/worldrsi-decision/SKILL.md"))
        diff_name = "editor-policy.diff"
        (self.out / "evidence" / diff_name).write_text(diff)
        files = {name: self.evidence(directory / f"{name}.json", f"editor-{name}.json")
                 for name in ["result", "coverage", "validation", "manifest"]}
        files.update({"policy": self.evidence(after, "editor-final-decision-skill.md"),
                      "before_policy": self.evidence(before, "editor-original-decision-skill.md"),
                      "helpers": self.evidence(helpers, "editor-decision-models.py"),
                      "diff": f"evidence/{diff_name}"})
        return {"result": read(directory / "result.json"),
                "coverage": read(directory / "coverage.json"),
                "validation": read(directory / "validation.json"), "files": files,
                "policy_before_lines": len(old_text.splitlines()),
                "policy_after_lines": len(new_text.splitlines()),
                "policy_before_bytes": before.stat().st_size, "policy_after_bytes": after.stat().st_size,
                "policy_text": new_text, "helper_functions": functions,
                "template_changes": template_changes,
                "manual_files": sorted(manifest["staged_manuals"])}

    def collect(self):
        rows = []
        summaries = {}
        for directory in sorted((self.round / "sources").iterdir()):
            if not (directory / "source.json").is_file():
                continue
            source, experience = read(directory / "source.json"), read(directory / "experience.json")
            summary = self.summaries[source["scene_id"]]
            if summary["experience_sha256"] != hashlib.sha256((directory / "experience.json").read_bytes()).hexdigest():
                raise ValueError(f"Experience summary needs updating: {directory}")
            summaries[source["scene_id"]] = [summary["title"], summary["summary"]]
            tid = source["task_id"]
            public = read(directory / "initial/task.json")["public"]
            arms = self.gt(tid) + [self.arm(directory / "initial", tid, "Initial")]
            for branch in source["branches"]:
                arms.append(self.arm(branch["directory"], tid, branch["id"]))
            gate = directory / "transfer/decision.json"
            rows.append({"id": "learning-" + source["scene_id"], "section": "learning",
                         "scene": source["scene_id"], "task_id": tid,
                         "view": public["view"], "category": public["category"],
                         "experience": experience, "transfer": read(gate) if gate.exists() else None,
                         "experience_file": self.evidence(directory / "experience.json", source["scene_id"] + "-experience.json"),
                         "arms": arms})
        initial = self.run / "initial-validation"
        candidate = self.round / "validation"
        before, after = read(initial / "report.json"), read(candidate / "report.json")
        if before["task_ids"] != after["task_ids"] or before["seeds"] != after["seeds"]:
            raise ValueError("Validation plans do not match")
        for tid in after["task_ids"]:
            task = self.local(after["outputs"]["42"][tid])
            public = read(task / "task.json")["public"]
            rows.append({"id": "validation-" + tid, "section": "validation",
                         "scene": public["scene_id"], "task_id": tid,
                         "view": public["view"], "category": public["category"],
                         "arms": self.gt(tid) + [self.arm(initial, tid, "Initial"),
                                                   self.arm(candidate, tid, "R001 Candidate")]})
        branch_count = sum(sum("hypothesis" in a for a in r["arms"])
                           for r in rows if r["section"] == "learning")
        meta = {"run": self.run.name, "round": self.round.name, "seeds": SEEDS,
                "editor": self.collect_editor(),
                "summaries": summaries,
                "learning_count": sum(r["section"] == "learning" for r in rows),
                "branch_count": branch_count, "validation_count": after["count"],
                "initial_mean": before["mean"], "candidate_mean": after["mean"],
                "initial_per_seed": before["per_seed"], "candidate_per_seed": after["per_seed"],
                "decision": read(self.round / "decision.json"), "run_directory": str(self.run),
                "initial_report": self.evidence(initial / "report.json", "initial-validation-report.json"),
                "candidate_report": self.evidence(candidate / "report.json", "r001-validation-report.json")}
        for row in rows:
            row["comparison"] = {}
            for seed in SEEDS:
                stem = f"{row['id']}-seed-{seed}-gt1"
                row["comparison"][str(seed)] = {"video": f"media/{stem}.mp4", "poster": f"media/{stem}.jpg"}
        self.data = {"meta": meta, "rows": rows}
        return self.data

    def run_ffmpeg(self, argv, log):
        with open(log, "w") as f:
            done = subprocess.run([self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *argv],
                                  stdout=f, stderr=subprocess.STDOUT)
        if done.returncode:
            raise RuntimeError(f"ffmpeg failed; see {log}: {Path(log).read_text()[-1800:]}")

    def normalize_gt(self, job):
        source, target = job
        if target.exists():
            return
        temp = target.with_suffix(".partial.mp4")
        self.run_ffmpeg(["-threads", "2", "-i", str(source), "-an", "-vf",
                         f"fps=24,scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
                         f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=0x101923,setsar=1",
                         "-filter_threads", "1", "-frames:v", "120", "-c:v", "libx264", "-preset", "fast",
                         "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(temp)],
                        target.with_suffix(".log"))
        temp.replace(target)

    def stitch(self, job):
        row, seed = job
        media = row["comparison"][str(seed)]
        target, poster = self.out / media["video"], self.out / media["poster"]
        if not target.exists():
            args, filters = [], []
            header = Image.new("RGB", (WIDTH * len(row["arms"]), HEADER), "#101923")
            draw = ImageDraw.Draw(header)
            title_font, score_font = ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 18)
            for i, arm in enumerate(row["arms"]):
                args += ["-threads", "2", "-i", arm["videos"][str(seed)]]
                title = arm["label"]
                score = (f"seed {seed}: {arm['scores'][str(seed)]:.4f} | mean: {arm['mean']:.4f}"
                         if arm["scores"] else "Reference | score: N/A")
                draw.text((i * WIDTH + 16, 12), title, fill="white", font=title_font)
                draw.text((i * WIDTH + 16, 49), score, fill="#c4ddd1", font=score_font)
                if i:
                    draw.line((i * WIDTH, 8, i * WIDTH, HEADER - 8), fill="#344a50", width=1)
                # Letterbox without stretching or cropping; preserve the source motion speed.
                filters.append(f"[{i}:v]setpts=PTS-STARTPTS,fps=24,"
                               f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
                               f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=0x101923,setsar=1,"
                               f"pad={WIDTH}:{HEIGHT + HEADER}:0:{HEADER}:color=0x101923[v{i}]")
            inputs = "".join(f"[v{i}]" for i in range(len(row["arms"])))
            filters.append(f"{inputs}hstack=inputs={len(row['arms'])}:shortest=1[stack]")
            header_path = self.cache / f"{target.stem}-header.png"
            header.save(header_path)
            args += ["-threads", "1", "-i", str(header_path)]
            filters.append(f"[stack][{len(row['arms'])}:v]overlay=0:0[out]")
            temp = target.with_suffix(".partial.mp4")
            self.run_ffmpeg([*args, "-filter_complex_threads", "1", "-filter_complex", ";".join(filters),
                             "-map", "[out]", "-an", "-frames:v", "120", "-c:v", "libx264", "-preset", "fast",
                             "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(temp)],
                            self.cache / (target.stem + ".log"))
            temp.replace(target)
        if not poster.exists():
            self.run_ffmpeg(["-threads", "1", "-i", str(target), "-frames:v", "1", "-q:v", "3", str(poster)],
                            self.cache / (target.stem + "-poster.log"))
        return target.name

    def html(self):
        template = Path(__file__).with_name("round_report_template.html").read_text()
        data_json = json.dumps(self.data, ensure_ascii=False).replace("<", "\\u003c")
        (self.out / "index.html").write_text(template.replace("__REPORT_DATA__", data_json))
        write(self.out / "data.json", self.data)

    def build(self, only_html=False):
        self.collect()
        self.html()
        if only_html:
            return
        print(f"Preparing {len(self.gt_jobs)} GT references", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.jobs) as pool:
            for i, _ in enumerate(pool.map(self.normalize_gt, self.gt_jobs.values()), 1):
                if i % 10 == 0:
                    print(f"GT {i}/{len(self.gt_jobs)}", flush=True)
        jobs = [(row, seed) for seed in SEEDS for row in self.data["rows"]]
        print(f"Rendering {len(jobs)} horizontal comparisons", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.jobs) as pool:
            futures = [pool.submit(self.stitch, job) for job in jobs]
            for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
                print(f"Video {i}/{len(jobs)}: {f.result()}", flush=True)
        # Verify all encoded results, including timeline and browser-compatible codec.
        probe = str(Path(self.ffmpeg).with_name("ffprobe"))
        if not Path(probe).exists():
            probe = "ffprobe"
        checked = []
        for row, seed in jobs:
            file = self.out / row["comparison"][str(seed)]["video"]
            info = json.loads(subprocess.check_output([probe, "-v", "error", "-select_streams", "v:0",
                              "-show_entries", "stream=codec_name,width,height,nb_frames,r_frame_rate,pix_fmt:format=duration",
                              "-of", "json", str(file)]))
            s = info["streams"][0]
            assert s["codec_name"] == "h264" and s["pix_fmt"] == "yuv420p", file
            assert int(s["nb_frames"]) == 120 and s["r_frame_rate"] == "24/1", file
            assert s["width"] == WIDTH * len(row["arms"]) and s["height"] == HEIGHT + HEADER, file
            assert abs(float(info["format"]["duration"]) - 5) < .01, file
            checked.append({"video": str(file.relative_to(self.out)), **s, "bytes": file.stat().st_size})
        write(self.out / "verification.json", {"status": "passed", "comparisons": len(checked),
              "learning_tasks": self.data["meta"]["learning_count"], "branches": self.data["meta"]["branch_count"],
              "validation_tasks": self.data["meta"]["validation_count"], "videos": checked})
        print(f"DONE: {self.out / 'index.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--run", required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ffmpeg", default="/data/bin/ffmpeg")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--summaries", default=str(Path(__file__).with_name("round_report_summaries.json")),
                        help="Chinese summaries bound to the current source experiences")
    parser.add_argument("--html-only", action="store_true", help="Refresh the page after an existing media build")
    args = parser.parse_args()
    Report(args).build(args.html_only)


if __name__ == "__main__":
    main()
