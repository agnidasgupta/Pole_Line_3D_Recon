#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError

INT_RE=re.compile(r'-?\d+')

def read_csv(path):
    try:return pd.read_csv(path)
    except EmptyDataError:return pd.DataFrame()

def ints_from_text(x):
    if x is None:return []
    return [int(v) for v in INT_RE.findall(str(x))]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--replay_dir',required=True)
    ap.add_argument('--max_span_slices',type=int,default=9)
    ap.add_argument('--max_span_length_ft',type=float,default=450.0)
    ap.add_argument('--require_all_stage3_snapshots',type=int,default=1)
    a=ap.parse_args()
    root=Path(a.replay_dir)
    manifest_path=root/'inference_manifest.csv'
    timing_path=root/'realtime_slice_timing.csv'
    complete_path=root/'COMPLETED.json'
    if not manifest_path.is_file(): raise SystemExit(f'ERROR missing {manifest_path}')
    if not timing_path.is_file(): raise SystemExit(f'ERROR missing {timing_path}')
    if not complete_path.is_file(): raise SystemExit(f'ERROR missing {complete_path}')
    man=read_csv(manifest_path)
    tim=read_csv(timing_path)
    comp=json.load(open(complete_path))
    errs=[]; warns=[]
    if not comp.get('completed'): errs.append('replay COMPLETED.json does not report completed=true')
    for key in ('stage1_slice_local','stage2_slice_local','stage3_multi_slice_only'):
        if comp.get(key) is not True: errs.append(f'{key} is not true')
    if int(comp.get('stage3_max_span_slices',-1))!=a.max_span_slices: errs.append('max span slice contract mismatch')
    if float(comp.get('stage3_max_span_length_ft',-1))!=a.max_span_length_ft: errs.append('max span length contract mismatch')
    if man.empty: errs.append('manifest empty')
    else:
        seq=pd.to_numeric(man['slice_seq'],errors='coerce').dropna().astype(int)
        if len(seq)!=len(man): errs.append('manifest contains invalid slice_seq')
        if seq.duplicated().any(): errs.append(f'duplicate slice_seq values: {sorted(seq[seq.duplicated()].unique())[:20]}')
        if not seq.is_monotonic_increasing: errs.append('manifest slice_seq is not increasing')
        for col in ['pole_csv','line_csv','line_vertices_csv']:
            if col not in man.columns: errs.append(f'manifest missing {col}'); continue
            for v in man[col].astype(str):
                p=Path(v)
                if not p.is_file() or p.stat().st_size==0:
                    errs.append(f'missing/empty Stage2 output: {p}')
                    if len(errs)>50: break
    if not tim.empty:
        tseq=pd.to_numeric(tim['slice_seq'],errors='coerce').dropna().astype(int)
        if tseq.duplicated().any(): errs.append('timing CSV has duplicate slice_seq rows')
        missing=set(pd.to_numeric(man['slice_seq'],errors='coerce').dropna().astype(int))-set(tseq)
        if missing: warns.append(f'timing rows missing for {len(missing)} manifest slices; first={sorted(missing)[:10]}')
    stage3_root=root/'stage3_incremental'
    latest=list(stage3_root.rglob('LATEST.json')) if stage3_root.exists() else []
    if len(latest)!=1: errs.append(f'expected one LATEST.json for one replay session, found {len(latest)}')
    snapshots=[]
    for p in stage3_root.rglob('slice_*') if stage3_root.exists() else []:
        if p.is_dir(): snapshots.append(p)
    expected_seqs=set(pd.to_numeric(man['slice_seq'],errors='coerce').dropna().astype(int)) if not man.empty else set()
    found_seqs=set()
    for snap in snapshots:
        m=re.search(r'slice_(\d+)$',snap.name)
        if not m: continue
        latest_seq=int(m.group(1)); found_seqs.add(latest_seq)
        cp=snap/'COMPLETED.json'
        if not cp.is_file():
            errs.append(f'missing Stage3 COMPLETED.json: {snap}')
            continue
        c=json.load(open(cp)); rules=c.get('rules',{})
        if float(rules.get('max_span_length_ft',-1))!=a.max_span_length_ft: errs.append(f'{snap}: max_span_length_ft mismatch')
        if int(rules.get('max_span_slices',-1))!=a.max_span_slices: errs.append(f'{snap}: max_span_slices mismatch')
        lo=latest_seq-a.max_span_slices
        gid=str(comp.get('group_id',''))
        session_dir=snap/'sessions'/Path(gid)
        ch=read_csv(session_dir/'conductor_chains.csv') if session_dir.exists() else pd.DataFrame()
        if not ch.empty:
            for col in ('slice_min','slice_max'):
                if col in ch.columns:
                    vals=pd.to_numeric(ch[col],errors='coerce').dropna().astype(int)
                    if col=='slice_min' and (vals<lo).any(): errs.append(f'{snap}: conductor uses slice older than {lo}')
                    if col=='slice_max' and (vals>latest_seq).any(): errs.append(f'{snap}: conductor uses future slice > {latest_seq}')
        wp=read_csv(session_dir/'world_poles.csv') if session_dir.exists() else pd.DataFrame()
        if not wp.empty and 'source_slices' in wp.columns:
            for x in wp['source_slices'].dropna():
                ss=ints_from_text(x)
                if any(v<lo for v in ss): errs.append(f'{snap}: pole source slice older than {lo}')
                if any(v>latest_seq for v in ss): errs.append(f'{snap}: pole source slice in future of {latest_seq}')
    if a.require_all_stage3_snapshots and expected_seqs-found_seqs:
        errs.append(f'missing Stage3 snapshots for {len(expected_seqs-found_seqs)} slices; first={sorted(expected_seqs-found_seqs)[:10]}')
    report={
        'replay_dir':str(root),'manifest_rows':len(man),'timing_rows':len(tim),'stage3_snapshots':len(found_seqs),
        'max_span_slices':a.max_span_slices,'max_span_length_ft':a.max_span_length_ft,'errors':errs,'warnings':warns,'ok':not errs,
    }
    print(json.dumps(report,indent=2))
    (root/'REALTIME_REPLAY_VERIFICATION.json').write_text(json.dumps(report,indent=2))
    if errs: raise SystemExit('V4_REALTIME_REPLAY_VERIFICATION_FAILED')
    print('V4_REALTIME_REPLAY_VERIFICATION_OK')
if __name__=='__main__':main()
