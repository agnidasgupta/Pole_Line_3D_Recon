#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
from pathlib import Path


def src(text: str, node: ast.AST) -> str:
    return ast.get_source_segment(text, node) or ""


def target_text(text: str, node: ast.AST) -> str:
    return src(text, node).strip()


def nameish(s: str, tokens) -> bool:
    q = s.lower()
    return any(t in q for t in tokens)


def collect_option_candidates(text: str):
    tree = ast.parse(text)
    rows = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        fn_text = src(text, fn)
        for node in ast.walk(fn):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if value is None:
                continue
            rhs = src(text, value)
            compares = [x for x in ast.walk(value) if isinstance(x, ast.Compare)]
            if not compares:
                continue
            for t in targets:
                lhs = target_text(text, t)
                score = 0
                if nameish(lhs, ("line", "wire", "conductor")): score += 4
                if nameish(lhs, ("mask", "cand", "keep", "active", "idx", "voxel")): score += 3
                if nameish(rhs, ("line", "wire", "conductor")): score += 4
                if nameish(rhs, ("thr", "threshold")): score += 2
                if nameish(fn_text, ("component", "refiner", "parameter")): score += 2
                if nameish(lhs, ("pole",)): score -= 4
                if score <= 0:
                    continue
                rows.append({
                    "score": score,
                    "function": fn.name,
                    "line": getattr(node, "lineno", -1),
                    "lhs": lhs,
                    "rhs": rhs,
                    "node": node,
                    "fn": fn,
                    "compares": compares,
                })
    return sorted(rows, key=lambda r: (-r["score"], r["line"]))


def expr_candidates(text: str, fn: ast.AST, kind: str):
    values = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Name):
            s = n.id
        elif isinstance(n, (ast.Subscript, ast.Attribute)):
            s = src(text, n)
        else:
            continue
        low = s.lower()
        score = 0
        if kind == "coords":
            if "coord" in low: score += 8
            if re.search(r"(^|[^a-z])xyz([^a-z]|$)", low): score += 7
            if "voxel" in low and ("xyz" in low or "coord" in low): score += 4
        elif kind == "pole":
            if "pole" in low:
                score += 6
                if "score" in low or "prob" in low or "logit" in low:
                    score += 3
            else:
                score = 0
        elif kind == "background":
            if "background" in low or re.search(r"(^|_)bg(_|$)", low):
                score += 6
                if "score" in low or "prob" in low or "logit" in low:
                    score += 3
            else:
                score = 0
        if score:
            values.append((score, s))
    out = []
    seen = set()
    for score, s in sorted(values, key=lambda x: (-x[0], len(x[1]))):
        if s in seen:
            continue
        seen.add(s)
        out.append((score, s))
    return out


def find_line_score_and_threshold(text: str, compares):
    ranked = []
    for c in compares:
        if not c.comparators:
            continue
        left = src(text, c.left).strip()
        right = src(text, c.comparators[0]).strip()
        ll, rr = left.lower(), right.lower()

        def is_thr(x: str) -> bool:
            return ("threshold" in x or "_thr" in x or x.endswith("thr")
                    or re.fullmatch(r"[0-9.eE+-]+", x) is not None)

        def is_line_score(x: str) -> bool:
            return (nameish(x, ("line", "wire", "conductor"))
                    and nameish(x, ("score", "prob", "logit", "evidence"))
                    and not is_thr(x))

        if is_line_score(left) and is_thr(right):
            ranked.append((12, left, right))
        if is_line_score(right) and is_thr(left):
            ranked.append((12, right, left))
        # Fallback for arrays simply named `line` / `line_pred`.
        if nameish(left, ("line", "wire", "conductor")) and not is_thr(left) and is_thr(right):
            ranked.append((7, left, right))
        if nameish(right, ("line", "wire", "conductor")) and not is_thr(right) and is_thr(left):
            ranked.append((7, right, left))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    return ranked[0][1], ranked[0][2]


def make_report(target: Path, rows, text: str):
    report = []
    for r in rows[:20]:
        report.append({k: r[k] for k in ("score", "function", "line", "lhs", "rhs")})
    report_path = target.with_suffix(target.suffix + ".line_recall_discovery.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    target = Path(args.target)
    text = target.read_text()
    if "v4_stage2_line_recall" in text:
        raise SystemExit("Target already contains the line-recall hook; refusing double patch")

    rows = collect_option_candidates(text)
    report_path = make_report(target, rows, text)
    print(f"discovery_report={report_path}")
    for r in rows[:10]:
        print(f"candidate score={r['score']} line={r['line']} fn={r['function']} lhs={r['lhs']} rhs={r['rhs']}")
    if not rows or rows[0]["score"] < 9:
        raise SystemExit("AUTO_PATCH_NOT_READY: no high-confidence line candidate assignment found")
    if len(rows) > 1 and rows[1]["score"] >= rows[0]["score"] - 1:
        raise SystemExit("AUTO_PATCH_NOT_READY: top line-candidate assignments are ambiguous; inspect discovery report")

    best = rows[0]
    lhs = best["lhs"]
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs):
        raise SystemExit(f"AUTO_PATCH_NOT_READY: candidate target is not a simple variable: {lhs!r}")
    score_thr = find_line_score_and_threshold(text, best["compares"])
    if not score_thr:
        raise SystemExit("AUTO_PATCH_NOT_READY: could not infer line-score and threshold expressions")
    line_expr, thr_expr = score_thr
    coords = expr_candidates(text, best["fn"], "coords")
    if not coords:
        raise SystemExit("AUTO_PATCH_NOT_READY: could not infer sparse coordinate expression")
    coords_expr = coords[0][1]
    pole = expr_candidates(text, best["fn"], "pole")
    bg = expr_candidates(text, best["fn"], "background")
    pole_expr = pole[0][1] if pole else "None"
    bg_expr = bg[0][1] if bg else "None"

    print("AUTO_PATCH_READY=true")
    print(f"hook_function={best['function']}")
    print(f"mask_var={lhs}")
    print(f"coords_expr={coords_expr}")
    print(f"line_score_expr={line_expr}")
    print(f"strong_threshold_expr={thr_expr}")
    print(f"pole_score_expr={pole_expr}")
    print(f"background_score_expr={bg_expr}")
    if not args.write:
        print("DRY_RUN_OK")
        return

    lines = text.splitlines(keepends=True)
    node = best["node"]
    end_line = getattr(node, "end_lineno", node.lineno)
    raw_line = lines[node.lineno - 1]
    indent = raw_line[: len(raw_line) - len(raw_line.lstrip())]
    hook = (
        f"{indent}{lhs}, _v4_line_recall_audit = recover_line_candidates(\n"
        f"{indent}    coords_xyz={coords_expr},\n"
        f"{indent}    line_score={line_expr},\n"
        f"{indent}    strong_mask={lhs},\n"
        f"{indent}    strong_threshold=float({thr_expr}),\n"
        f"{indent}    pole_score={pole_expr},\n"
        f"{indent}    background_score={bg_expr},\n"
        f"{indent})\n"
    )
    lines.insert(end_line, hook)
    patched = "".join(lines)

    # Insert import after future/import block, without relying on exact repository formatting.
    tree2 = ast.parse(patched)
    insert_line = 1
    body = tree2.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
        insert_line = body[0].end_lineno + 1
    for n in body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            insert_line = max(insert_line, n.end_lineno + 1)
        elif getattr(n, "lineno", 0) > insert_line:
            break
    plines = patched.splitlines(keepends=True)
    plines.insert(insert_line - 1, "from v4_stage2_line_recall import recover_line_candidates\n")
    patched = "".join(plines)
    ast.parse(patched)

    backup = target.with_suffix(target.suffix + ".pre_line_recall.bak")
    shutil.copy2(target, backup)
    target.write_text(patched)
    print(f"backup={backup}")
    print("PATCH_WRITTEN_OK")


if __name__ == "__main__":
    main()
