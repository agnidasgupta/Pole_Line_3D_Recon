#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    p = Path(args.entry)
    text = p.read_text()
    tree = ast.parse(text)
    rows = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        fn = n.func
        is_add = isinstance(fn, ast.Attribute) and fn.attr == "add_argument"
        if not is_add:
            continue
        opts = []
        for a in n.args:
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                opts.append(a.value)
        kw = {}
        for k in n.keywords:
            if k.arg:
                try:
                    kw[k.arg] = ast.literal_eval(k.value)
                except Exception:
                    kw[k.arg] = ast.unparse(k.value)
        rows.append({"options": opts, "kwargs": kw, "line": n.lineno})
    Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")
    print(f"STAGE2_CLI_DISCOVERY_OK args={len(rows)} output={args.output}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
