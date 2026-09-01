#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def discover(entry: Path):
    text = entry.read_text()
    tree = ast.parse(text)
    rows = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Attribute) or n.func.attr != "add_argument":
            continue
        opts = [a.value for a in n.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if not opts:
            continue
        kw = {}
        for k in n.keywords:
            if not k.arg:
                continue
            try:
                kw[k.arg] = ast.literal_eval(k.value)
            except Exception:
                kw[k.arg] = ast.unparse(k.value)
        rows.append({"options": opts, "kwargs": kw, "line": n.lineno})
    return rows


def best_option(rows, kind):
    ranked = []
    for r in rows:
        for opt in r["options"]:
            if not opt.startswith("--"):
                continue
            low = opt.lower().replace("-", "_")
            help_text = str(r["kwargs"].get("help", "")).lower()
            score = 0
            if kind == "stage1":
                if "stage1" in low: score += 10
                if "inference" in low: score += 6
                if "input" in low: score += 4
                if any(x in low for x in ("dir", "root", "manifest")): score += 4
                if "stage2" in low or "output" in low: score -= 8
                if "stage 1" in help_text or "stage1" in help_text: score += 4
            elif kind == "output":
                if "output" in low: score += 10
                if any(x in low for x in ("dir", "root")): score += 4
                if "stage1" in low or "input" in low: score -= 8
            elif kind == "session":
                if "session" in low or "group" in low: score += 8
                if "filter" in low: score += 4
            if score > 0:
                ranked.append((score, opt, r))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    return ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required=True)
    ap.add_argument("--stage1_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--session_filter", default="")
    ap.add_argument("--discover_only", type=int, default=0)
    ap.add_argument("--extra", action="append", default=[], help="extra CLI token; repeat for each token")
    args = ap.parse_args()

    entry = Path(args.entry)
    rows = discover(entry)
    s1 = best_option(rows, "stage1")
    out = best_option(rows, "output")
    sess = best_option(rows, "session")
    report = {
        "entry": str(entry),
        "stage1_candidates": [{"score": x[0], "option": x[1], "meta": x[2]} for x in s1[:10]],
        "output_candidates": [{"score": x[0], "option": x[1], "meta": x[2]} for x in out[:10]],
        "session_candidates": [{"score": x[0], "option": x[1], "meta": x[2]} for x in sess[:10]],
        "all_arguments": rows,
    }
    report_path = Path(args.output_dir).parent / "stage2_cli_discovery.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"stage2_cli_discovery={report_path}")
    if not s1 or not out:
        raise SystemExit("STAGE2_CLI_NOT_RESOLVED: inspect stage2_cli_discovery.json")
    if len(s1) > 1 and s1[1][0] >= s1[0][0]:
        raise SystemExit("STAGE2_CLI_AMBIGUOUS_STAGE1: inspect stage2_cli_discovery.json")
    if len(out) > 1 and out[1][0] >= out[0][0]:
        raise SystemExit("STAGE2_CLI_AMBIGUOUS_OUTPUT: inspect stage2_cli_discovery.json")

    stage1_opt = s1[0][1]
    output_opt = out[0][1]
    session_opt = sess[0][1] if sess else None
    print(f"resolved_stage1_option={stage1_opt}")
    print(f"resolved_output_option={output_opt}")
    print(f"resolved_session_option={session_opt}")
    if args.discover_only:
        print("STAGE2_CLI_DISCOVERY_ONLY_OK")
        return

    cmd = [sys.executable, str(entry), stage1_opt, args.stage1_dir, output_opt, args.output_dir]
    if args.session_filter:
        if not session_opt:
            raise SystemExit("A session_filter was requested but the Stage2 entrypoint has no discoverable session/group option")
        cmd += [session_opt, args.session_filter]
    cmd += args.extra
    print("EXEC:", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
