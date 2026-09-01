# Acceptance check for the paper's \jsonpath tags: every tag resolves to a
# leaf of .results/all_results.json, and the number printed immediately
# before the tag literally equals the JSON value (the paper is a dumb pipe:
# no re-rounding downstream of collect_json).
#
#     uv run python -m replication.verify_paper_tags
#
# Value tokens recognized before a tag: plain number, thousands-separated
# number, sci string ``m \times 10^{e}``, or a bare power ``10^{x}``. A tag
# with no recognizable token (prose like "the ten calibrations") is listed
# as UNCHECKED, not a failure. Exit code 1 on any unresolved tag or literal
# mismatch.

from __future__ import annotations

import json
import pathlib
import re
import sys

import jsonargparse

TAG = re.compile(r"\\jsonpath\{([^{}]*)\}")

# Value tokens, tried in order against the tex immediately before the tag.
SCI = re.compile(r"\$?(-?\d+(?:\.\d+)?)\s*\\times\s*10\^\{(-?\d+)\}\$?\s*$")
POW10 = re.compile(r"\$?10\^\{(-?\d+(?:\.\d+)?)\}\$?\s*$")
NUMBER = re.compile(r"\$?(-?\d{1,3}(?:\{,\}\d{3})+|-?\d+(?:\.\d+)?)\$?\s*$")


def resolve(payload, path):
    node = payload
    for part in path.strip().strip("/").split("/"):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def value_token(tex, end):
    window = tex[max(0, end - 60) : end]
    m = SCI.search(window)
    if m:
        return ("sci", float(m.group(1)), int(m.group(2)))
    m = POW10.search(window)
    if m:
        return ("pow10", float(m.group(1)))
    m = NUMBER.search(window)
    if m:
        return ("number", float(m.group(1).replace("{,}", "")))
    return None


def matches(token, value):
    if token is None:
        return None  # unchecked
    kind = token[0]
    if isinstance(value, str):
        m = re.match(r"^(-?\d+(?:\.\d+)?) \\times 10\^\{(-?\d+)\}$", value)
        p = re.match(r"^10\^\{(-?\d+(?:\.\d+)?)\}$", value)
        if kind == "sci" and m:
            return float(m.group(1)) == token[1] and int(m.group(2)) == token[2]
        if kind == "pow10" and p:
            return float(p.group(1)) == token[1]
        return False
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if kind == "number":
            return float(value) == token[1]
        if kind == "pow10":
            # A clean power like 10^{-14} against the raw float 1e-14.
            return float(value) == token[1] or 10.0 ** token[1] == float(value)
        if kind == "sci":
            return float(value) == token[1] * 10 ** token[2]
    return None  # lists/dicts: prose references, not comparable


def check(
    json_path: str = ".results/all_results.json",
    paper_dir: str = "../kernels_dynamical_systems/QE",
):
    payload = json.loads(pathlib.Path(json_path).read_text())
    broken, mismatched, unchecked, ok = [], [], [], 0
    total = 0
    for tex_path in sorted(pathlib.Path(paper_dir).glob("*.tex")):
        src = tex_path.read_text()
        name = tex_path.name
        for m in TAG.finditer(src):
            # Skip the macro definition and commented lines.
            line_start = src.rfind("\n", 0, m.start()) + 1
            line = src[line_start : m.start()]
            if "newcommand" in line or line.lstrip().startswith("%"):
                continue
            total += 1
            found, value = resolve(payload, m.group(1))
            where = f"{name}:{src[: m.start()].count(chr(10)) + 1}"
            if not found:
                broken.append((where, m.group(1)))
                continue
            verdict = matches(value_token(src, m.start()), value)
            if verdict is True:
                ok += 1
            elif verdict is False:
                token = value_token(src, m.start())
                mismatched.append((where, m.group(1), token, value))
            else:
                unchecked.append((where, m.group(1)))

    print(f"tags checked: {total}")
    print(f"  resolved + literal match: {ok}")
    print(f"  resolved, value not comparable (prose): {len(unchecked)}")
    print(f"  UNRESOLVED: {len(broken)}")
    for where, path in broken:
        print(f"    {where}  {path}")
    print(f"  MISMATCHED: {len(mismatched)}")
    for where, path, token, value in mismatched:
        print(f"    {where}  {path}")
        print(f"      printed {token}  vs  json {value!r}")
    return 1 if (broken or mismatched) else 0


if __name__ == "__main__":
    sys.exit(jsonargparse.CLI(check))
