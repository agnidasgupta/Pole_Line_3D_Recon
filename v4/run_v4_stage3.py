#!/usr/bin/env python3
"""Independent rolling V4 Stage-3 runner from durable Stage-2 artifacts."""
from __future__ import annotations
import argparse,contextlib,io,json,sys,time,traceback
from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError
import reconstruct_v4_stage3 as stage3
from v4_stage_contracts import CONTRACT_VERSION,atomic_json,safe_id


def pa():
    p=argparse.ArgumentParser(); p.add_argument("--stage2_dir",required=True); p.add_argument("--output_dir",required=True); p.add_argument("--session_filter",required=True)
    p.add_argument("--max_sequence_gap",type=int,default=9); p.add_argument("--max_span_length_ft",type=float,default=450.0); p.add_argument("--slice_length_ft",type=float,default=50.0)
    p.add_argument("--latest_only",type=int,default=0); p.add_argument("--resume",type=int,default=1); return p.parse_args()


def main():
    a=pa()
    if a.max_sequence_gap!=9 or abs(a.max_span_length_ft-450.0)>1e-9 or abs(a.slice_length_ft-50.0)>1e-9: raise ValueError("Production Stage3 contract is 9 x 50 ft = 450 ft")
    root=Path(a.stage2_dir).resolve(); out=Path(a.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True); mp=root/"inference_manifest.csv"
    if not mp.is_file(): raise FileNotFoundError(mp)
    try: man=pd.read_csv(mp)
    except EmptyDataError as e: raise RuntimeError(f"Stage2 manifest is empty: {mp}") from e
    q=man[(man.group_id.astype(str)==a.session_filter)&(man.status.astype(str)=="completed")].copy(); q=q.sort_values("slice_seq")
    if q.empty: raise RuntimeError(f"No completed Stage2 rows for {a.session_filter}")
    seqs=[int(x) for x in q.slice_seq.tolist()]; seqs=[seqs[-1]] if a.latest_only else seqs
    for i,seq in enumerate(seqs,1):
        dest=out/"stage3_incremental"/safe_id(a.session_filter)/f"slice_{seq:06d}"
        if a.resume and (dest/"COMPLETED.json").is_file(): print(f"[stage3] {i}/{len(seqs)} reuse seq={seq}",flush=True); continue
        dest.mkdir(parents=True,exist_ok=True); argv=["reconstruct_v4_stage3.py","--inference_dir",str(root),"--output_dir",str(dest),"--session_filter",a.session_filter,"--latest_slice",str(seq),"--max_sequence_gap",str(a.max_sequence_gap),"--max_span_length_ft",str(a.max_span_length_ft),"--slice_length_ft",str(a.slice_length_ft),"--disable_plots","--realtime_inmemory_cache"]
        old=sys.argv[:]; so=io.StringIO(); se=io.StringIO(); t0=time.perf_counter()
        try:
            sys.argv=argv
            with contextlib.redirect_stdout(so),contextlib.redirect_stderr(se): stage3.main()
        except Exception:
            se.write("\n--- PYTHON TRACEBACK ---\n"+traceback.format_exc()); raise
        finally:
            sys.argv=old; (dest/"stage3_stdout.log").write_text(so.getvalue()+"\n--- STDERR ---\n"+se.getvalue())
        print(f"[stage3] {i}/{len(seqs)} seq={seq} elapsed={(time.perf_counter()-t0)*1000:.1f}ms",flush=True)
    atomic_json({"completed":True,"contract_version":CONTRACT_VERSION,"stage":3,"group_id":a.session_filter,"latest_slice":seqs[-1],"max_sequence_gap":9,"max_observed_slice_centers":10,"slice_length_ft":50.0,"max_span_length_ft":450.0},out/"STAGE3_COMPLETED.json")


if __name__=="__main__": main()
