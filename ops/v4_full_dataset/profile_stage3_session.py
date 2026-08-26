#!/usr/bin/env python3
"""Run accepted V4 Stage 3 algorithm from saved Stage 2 artifacts with per-slice timings."""
from __future__ import annotations
import argparse,contextlib,io,json,sys,time,traceback
from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError
import reconstruct_v4_stage3 as stage3
from v4_stage_contracts import CONTRACT_VERSION,atomic_json,safe_id

def pa():
 p=argparse.ArgumentParser(); p.add_argument('--stage2_dir',required=True); p.add_argument('--output_dir',required=True); p.add_argument('--session_filter',required=True); p.add_argument('--timing_csv',required=True); p.add_argument('--max_sequence_gap',type=int,default=9); p.add_argument('--max_span_length_ft',type=float,default=450.0); p.add_argument('--slice_length_ft',type=float,default=50.0); p.add_argument('--resume',type=int,default=1); return p.parse_args()

def rj(p):
 try:return json.loads(Path(p).read_text())
 except Exception:return {}

def main():
 a=pa()
 if a.max_sequence_gap!=9 or abs(a.max_span_length_ft-450.0)>1e-9 or abs(a.slice_length_ft-50.0)>1e-9:raise ValueError('Production Stage3 contract is 9 x 50 ft = 450 ft')
 root=Path(a.stage2_dir).resolve(); out=Path(a.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True); tp=Path(a.timing_csv).resolve(); tp.parent.mkdir(parents=True,exist_ok=True); mp=root/'inference_manifest.csv'
 if not mp.is_file():raise FileNotFoundError(mp)
 try:man=pd.read_csv(mp)
 except EmptyDataError as e:raise RuntimeError(f'Stage2 manifest is empty: {mp}') from e
 q=man[(man.group_id.astype(str)==a.session_filter)&(man.status.astype(str)=='completed')].copy().sort_values('slice_seq')
 if q.empty:raise RuntimeError(f'No completed Stage2 rows for {a.session_filter}')
 rows=[]; seqs=[int(x) for x in q.slice_seq.tolist()]
 for i,seq in enumerate(seqs,1):
  dest=out/'stage3_incremental'/Path(a.session_filter)/f'slice_{seq:06d}'; marker=dest/'COMPLETED.json'; t0=time.perf_counter(); resumed=False
  if a.resume and marker.is_file():resumed=True; print(f'[stage3] {i}/{len(seqs)} reuse seq={seq}',flush=True)
  else:
   dest.mkdir(parents=True,exist_ok=True); argv=['reconstruct_v4_stage3.py','--inference_dir',str(root),'--output_dir',str(dest),'--session_filter',a.session_filter,'--latest_slice',str(seq),'--max_sequence_gap',str(a.max_sequence_gap),'--max_span_length_ft',str(a.max_span_length_ft),'--slice_length_ft',str(a.slice_length_ft),'--disable_plots','--realtime_inmemory_cache']; old=sys.argv[:]; so=io.StringIO(); se=io.StringIO()
   try:
    sys.argv=argv
    with contextlib.redirect_stdout(so),contextlib.redirect_stderr(se):stage3.main()
   except Exception:
    se.write('\n--- PYTHON TRACEBACK ---\n'+traceback.format_exc()); raise
   finally:
    sys.argv=old; (dest/'stage3_stdout.log').write_text(so.getvalue()+'\n--- STDERR ---\n'+se.getvalue())
  wall=(time.perf_counter()-t0)*1000.0; summ=rj(dest/'sessions'/Path(a.session_filter)/'summary.json')
  row={'group_id':a.session_filter,'slice_seq':seq,'stage3_wall_ms':0.0 if resumed else wall,'stage3_elapsed_ms':float(summ.get('elapsed_seconds',0))*1000.0,'stage3_fragment_join_ms':float(summ.get('fragment_join_seconds',0))*1000.0,'stage3_span_completion_pre_ms':float(summ.get('span_completion_pre_seconds',0))*1000.0,'stage3_hidden_pole_ms':float(summ.get('hidden_pole_seconds',0))*1000.0,'stage3_span_completion_post_ms':float(summ.get('span_completion_post_seconds',0))*1000.0,'stage3_chain_build_attachment_ms':float(summ.get('chain_build_and_attachment_seconds',0))*1000.0,'stage3_output_write_ms':float(summ.get('output_write_seconds',0))*1000.0,'merged_poles':summ.get('merged_poles'),'conductor_chains':summ.get('conductor_chains'),'spans':summ.get('spans'),'resume_from':'stage3' if resumed else 'none'}; rows.append(row); print(f'[stage3] {i}/{len(seqs)} seq={seq} algorithm={row["stage3_elapsed_ms"]:.1f}ms wall={row["stage3_wall_ms"]:.1f}ms',flush=True)
 pd.DataFrame(rows).to_csv(tp,index=False); atomic_json({'completed':True,'contract_version':CONTRACT_VERSION,'stage':3,'group_id':a.session_filter,'latest_slice':seqs[-1],'max_sequence_gap':9,'max_observed_slice_centers':10,'slice_length_ft':50.0,'max_span_length_ft':450.0,'timing_csv':str(tp)},out/'STAGE3_COMPLETED.json')
if __name__=='__main__':main()
