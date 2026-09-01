#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, shutil
from pathlib import Path

IMPORT_LINE = 'from v4_stage2_line_recall import recover_line_candidates_auto\n'
MARKER = 'recover_line_candidates_auto('


def function_span(text, name):
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    offs=[0]
    for ln in lines: offs.append(offs[-1]+len(ln))
    for n in tree.body:
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name:
            return offs[n.lineno-1], offs[n.end_lineno], n
    raise RuntimeError(f'function not found: {name}')


def find_hook(text, fn):
    hits=[]
    for n in ast.walk(fn):
        if not isinstance(n, ast.Assign) or len(n.targets)!=1:
            continue
        t=n.targets[0]
        if not isinstance(t, ast.Name) or t.id!='acc':
            continue
        rhs=ast.get_source_segment(text,n.value) or ''
        norm=''.join(rhs.split())
        if norm=='(prob>=thr)&physical' or norm=='prob>=thr&physical':
            hits.append(n)
    if len(hits)!=1:
        # Strong fallback: exact variable dependencies, independent of formatting.
        hits=[]
        for n in ast.walk(fn):
            if not isinstance(n, ast.Assign) or len(n.targets)!=1:
                continue
            t=n.targets[0]
            if not isinstance(t, ast.Name) or t.id!='acc':
                continue
            names={x.id for x in ast.walk(n.value) if isinstance(x,ast.Name)}
            if {'prob','thr','physical'}.issubset(names): hits.append(n)
    if len(hits)!=1:
        raise RuntimeError(f'expected exactly one apply_stage2 acc/prob/thr/physical assignment, found {len(hits)}')
    return hits[0]


def insert_import(text):
    if IMPORT_LINE.strip() in text:
        return text
    tree=ast.parse(text)
    insert_line=1
    body=tree.body
    if body and isinstance(body[0],ast.Expr) and isinstance(getattr(body[0],'value',None),ast.Constant) and isinstance(body[0].value.value,str):
        insert_line=body[0].end_lineno+1
    for n in body:
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            insert_line=max(insert_line,n.end_lineno+1)
        elif getattr(n,'lineno',0)>=insert_line:
            break
    lines=text.splitlines(keepends=True)
    lines.insert(insert_line-1,IMPORT_LINE)
    return ''.join(lines)


def patch(text):
    if MARKER in text:
        raise RuntimeError('target already contains recover_line_candidates_auto; refusing double patch')
    a,b,fn=function_span(text,'apply_stage2')
    hook=find_hook(text,fn)
    lines=text.splitlines(keepends=True)
    raw=lines[hook.lineno-1]
    indent=raw[:len(raw)-len(raw.lstrip())]
    block=(
        f"{indent}acc, _v4_line_recall_audit = recover_line_candidates_auto(\n"
        f"{indent}    local_vars=locals(),\n"
        f"{indent}    line_score=prob,\n"
        f"{indent}    strong_mask=acc,\n"
        f"{indent}    strong_threshold=float(thr),\n"
        f"{indent}    physical_mask=physical,\n"
        f"{indent})\n"
    )
    lines.insert(hook.end_lineno,block)
    out=''.join(lines)
    out=insert_import(out)
    ast.parse(out)
    return out, hook.lineno


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--target',required=True)
    ap.add_argument('--write',action='store_true')
    args=ap.parse_args()
    p=Path(args.target).expanduser().resolve()
    if not p.is_file(): raise SystemExit(f'target not found: {p}')
    old=p.read_text()
    new,line=patch(old)
    print('TARGETED_PATCH_READY=true')
    print(f'target={p}')
    print('hook_function=apply_stage2')
    print(f'hook_line={line}')
    print('hook_assignment=acc <- (prob >= thr) & physical')
    print('coordinate_resolution=runtime_strict')
    print('physical_mask_preserved=true')
    if not args.write:
        print('DRY_RUN_OK')
        return
    backup=p.with_suffix(p.suffix+'.pre_line_recall_targeted.bak')
    if backup.exists(): raise SystemExit(f'backup already exists: {backup}')
    shutil.copy2(p,backup)
    p.write_text(new)
    print(f'backup={backup}')
    print('TARGETED_PATCH_WRITTEN_OK')

if __name__=='__main__': main()
