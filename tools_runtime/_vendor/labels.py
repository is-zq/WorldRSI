"""Validate task-authored semantic IDs and RGB codes."""


def validate_palette(palette):
    if not isinstance(palette, dict) or not palette:
        raise ValueError("Semantic palette must be a nonempty ID mapping.")
    if palette.get("0", {}).get("color") != [0, 0, 0]:
        raise ValueError("Semantic ID 0 must encode black background.")
    names, colors = set(), set()
    for identifier, entry in palette.items():
        if not isinstance(identifier, str) or not identifier.isdecimal() or str(int(identifier)) != identifier:
            raise ValueError("Semantic palette IDs must be canonical nonnegative integers.")
        if not isinstance(entry, dict) or set(entry) != {"label", "color"}:
            raise ValueError("Semantic palette entries require label and color.")
        label, color = entry["label"], entry["color"]
        if not isinstance(label, str) or not label.strip() or label in names:
            raise ValueError("Semantic labels must be nonempty and unique.")
        if (not isinstance(color, list) or len(color) != 3 or
                any(type(v) is not int or not 0 <= v <= 255 for v in color) or tuple(color) in colors):
            raise ValueError("Semantic colors must be unique integer RGB triples.")
        names.add(label)
        colors.add(tuple(color))
    return palette
