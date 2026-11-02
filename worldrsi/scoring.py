"""Independent evidence sampling for official component observations."""
from pathlib import Path
from .common import digest, immutable_json, read_json, sha256
from .media import normalize, verify_locked
from .records import metered

@metered("scorer")
def evidence(task_dir, source, frames, reason, crop=None):
    """Scorer chooses arbitrary frame indices/crops; sampling itself is not a score."""
    directory = Path(task_dir)
    record = verify_locked(directory)
    if record["status"] != "valid":
        raise ValueError("Invalid outputs receive zero without sampling")
    task = record["task_config"]
    if not reason or not frames or any(type(f) is not int or f < 0 or f >= task["valid_frames"] for f in frames):
        raise ValueError("Sampling needs a reason and indices within the fixed valid window")
    source = Path(source).resolve()
    source_hash = sha256(source)
    selection = {"source": str(source), "source_sha256": source_hash, "frames": frames,
                 "reason": reason, "crop": crop, "valid_frames": task["valid_frames"]}
    key = digest(selection)
    dest = directory / "scoring/evidence" / key
    if (dest / "selection.json").exists():
        return read_json(dest / "selection.json")
    dest.mkdir(parents=True, exist_ok=True)
    normalized = dest / "normalized.mp4"
    if not normalized.exists():
        normalize(source, normalized, task["valid_frames"])
    import cv2
    cap = cv2.VideoCapture(str(normalized))
    images = []
    try:
        for frame in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok:
                raise ValueError("Cannot decode requested evidence frame")
            if crop is not None:
                x, y, width, height = crop
                if min(x, y) < 0 or min(width, height) <= 0 or x + width > image.shape[1] or y + height > image.shape[0]:
                    raise ValueError("Crop falls outside image")
                image = image[y:y + height, x:x + width]
            path = dest / f"frame-{frame:04d}.png"
            if not cv2.imwrite(str(path), image):
                raise ValueError("Cannot save evidence image")
            images.append({"frame": frame, "seconds": frame / 24, "path": str(path), "sha256": sha256(path)})
    finally:
        cap.release()
    return immutable_json(dest / "selection.json", {**selection, "images": images})


