#!/usr/bin/env python3
"""Replay one recorded session through the realtime V4 Stage1->Stage2->rolling Stage3 architecture.

Stage 1: current slice only.
Stage 2: current slice only.
Stage 3: after each new slice, current + prior sequence indices within latest-9
         (450 ft at nominal 50-ft slices). Missing sequence numbers are allowed.
"""
from __future__ import annotations
import argparse,contextlib,io,json,os,re,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np,pandas as pd
from pandas.errors import EmptyDataError

from v4_realtime_pipeline import V4RealtimePipeline
from v4_stage2_runtime import atomic_csv,POLE_OUTPUT_COLUMNS,LINE_OUTPUT_COLUMNS,VERTEX_OUTPUT_COLUMNS

SESSION_RE=re.compile(r'^(session\d+)_slice(\d+)$',re.I)
MANIFEST_COLUMNS=['id','source','relative_path','geography','session','slice_seq','group_id','center_x','center_y','center_z','output_csv','pole_csv','line_csv','line_vertices_csv','rows','accepted_poles','accepted_line_segments','status']


def args():
 p=argparse.ArgumentParser(); p.add_argument('--input_dir',required=True); p.add_argument('--session_filter',required=True,help='geography/sessionN')
 p.add_argument('--output_dir',required=True); p.add_argument('--model_path',required=True); p.add_argument('--calibration_json',required=True); p.add_argument('--stage2_bundle',required=True)
 p.add_argument('--stage3_script',default=str(Path(__file__).with_name('reconstruct_v4_stage3.py')))
 p.add_argument('--grid_size',type=int,nargs=3,default=[400,400,200]); p.add_argument('--voxel_size_ft',type=float,default=.5); p.add_argument('--core_size',type=int,default=48); p.add_argument('--batch_size',type=int,default=12)
 p.add_argument('--amp',choices=['fp16','bf16','none'],default='bf16'); p.add_argument('--compile_model',type=int,default=0); p.add_argument('--evaluate_all_cores',type=int,default=1,help='1 = reference full V4 tiling, 0 = occupied-output-core scheduling'); p.add_argument('--gpu_coord_channels',type=int,default=0,help='0 = original CPU coordinate-channel math; 1 = GPU-generated coordinate channels')
 p.add_argument('--pole_candidate_threshold',type=float,default=.15); p.add_argument('--line_candidate_threshold',type=float,default=.08); p.add_argument('--line_weak_threshold',type=float,default=.04); p.add_argument('--line_competition_ratio',type=float,default=.55)
 p.add_argument('--pole_min_voxels',type=int,default=4); p.add_argument('--line_min_voxels',type=int,default=3); p.add_argument('--edge_width_vox',type=int,default=10)
 p.add_argument('--max_span_slices',type=int,default=9); p.add_argument('--max_span_length_ft',type=float,default=450.); p.add_argument('--stage3_every_slice',type=int,default=1); p.add_argument('--stage3_execution',choices=['inprocess','subprocess'],default='inprocess',help='inprocess avoids Python/import startup on every arriving slice')
 p.add_argument('--write_row_csv',type=int,default=0); p.add_argument('--finalize_full_session',type=int,default=0,help='Compatibility flag only; production contract requires 0 and rejects all-session reconstruction.'); p.add_argument('--resume',type=int,default=1); p.add_argument('--max_slices',type=int,default=0)
 return p.parse_args()


def safe_id(s): return re.sub(r'[^A-Za-z0-9_.-]+','__',str(s))

def discover(root,gid):
 root=Path(root).resolve(); geo,sess=gid.split('/',1); base=root/geo
 if not base.exists(): raise FileNotFoundError(base)
 rows=[]
 for p in sorted(base.rglob('*.csv'))+sorted(base.rglob('*.csv.gz')):
  m=SESSION_RE.match(p.parent.name)
  if not m or m.group(1).lower()!=sess.lower(): continue
  seq=int(m.group(2)); rel=str(p.relative_to(root)); rows.append((seq,p,rel))
 rows.sort(key=lambda x:(x[0],x[2]))
 if not rows: raise RuntimeError(f'No slices found for {gid} under {root}')
 # one source file per numeric slice is the safe realtime contract
 seqs=[x[0] for x in rows]; dup=sorted({x for x in seqs if seqs.count(x)>1})
 if dup: raise RuntimeError(f'Multiple CSV sources for slice sequence(s) {dup[:20]}; resolve before realtime replay')
 return rows


def atomic_json(x,p):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(x,indent=2)); os.replace(q,p)


_STAGE3_MODULE = None

def _stage3_argv(a,out,gid,seq,dest):
 cmd=[a.stage3_script,'--inference_dir',str(out),'--output_dir',str(dest),'--session_filter',gid,
      '--max_span_slices',str(a.max_span_slices),'--max_span_length_ft',str(a.max_span_length_ft)]
 if seq is not None: cmd += ['--latest_slice',str(seq),'--disable_plots']
 return cmd

def run_stage3(a,out,gid,seq,dest):
 """Run Stage 3 for the acquired rolling window only.

 The default in-process mode imports the deterministic reconstruction module once
 and reuses that Python process for later slices.  The reconstruction itself is
 still recomputed from only [latest-9, latest] sequence indices, so no future or
 older-than-450-ft slice can influence the current update.
 """
 global _STAGE3_MODULE
 dest=Path(dest); dest.mkdir(parents=True,exist_ok=True); log=dest/'stage3_stdout.log'; argv=_stage3_argv(a,out,gid,seq,dest)
 t=time.perf_counter()
 if a.stage3_execution=='subprocess':
  proc=subprocess.run([sys.executable,*argv],text=True,capture_output=True)
  ms=(time.perf_counter()-t)*1000
  log.write_text(proc.stdout+'\n--- STDERR ---\n'+proc.stderr)
  if proc.returncode!=0: raise RuntimeError(f'Stage3 failed seq={seq} code={proc.returncode}; see {log}')
  return ms
 if _STAGE3_MODULE is None:
  import importlib.util
  spec=importlib.util.spec_from_file_location('v4_realtime_stage3_module',a.stage3_script)
  if spec is None or spec.loader is None: raise RuntimeError(f'Cannot import Stage3 module: {a.stage3_script}')
  _STAGE3_MODULE=importlib.util.module_from_spec(spec); spec.loader.exec_module(_STAGE3_MODULE)
 old_argv=sys.argv[:]; so=io.StringIO(); se=io.StringIO()
 try:
  sys.argv=argv
  with contextlib.redirect_stdout(so),contextlib.redirect_stderr(se):
   _STAGE3_MODULE.main()
 except Exception:
  se.write('\n--- PYTHON TRACEBACK ---\n'+traceback.format_exc())
  raise
 finally:
  sys.argv=old_argv
  log.write_text(so.getvalue()+'\n--- STDERR ---\n'+se.getvalue())
 return (time.perf_counter()-t)*1000


def stage3_breakdown(dest,gid):
 p=Path(dest)/'sessions'/Path(gid)/'summary.json'
 if not p.is_file(): return {}
 try: d=json.load(open(p))
 except Exception: return {}
 mapping={
  'elapsed_seconds':'stage3_algorithm_ms',
  'fragment_join_seconds':'stage3_fragment_join_ms',
  'span_completion_pre_seconds':'stage3_span_completion_pre_ms',
  'hidden_pole_seconds':'stage3_hidden_pole_ms',
  'span_completion_post_seconds':'stage3_span_completion_post_ms',
  'chain_build_and_attachment_seconds':'stage3_chain_build_attachment_ms',
  'output_write_seconds':'stage3_output_write_ms',
 }
 out={}
 for src,dst in mapping.items():
  try: out[dst]=float(d[src])*1000.0
  except Exception: pass
 return out


def main():
 a=args()
 if int(a.finalize_full_session) != 0:
  raise ValueError('--finalize_full_session must remain 0: production Stage3 is rolling past-only per arriving slice')
 if int(a.stage3_every_slice) != 1:
  raise ValueError('--stage3_every_slice must remain 1 for the production realtime contract')
 if int(a.max_span_slices) != 9 or abs(float(a.max_span_length_ft)-450.0) > 1e-9:
  raise ValueError('production Stage3 requires max_span_slices=9 and max_span_length_ft=450')
 inp=Path(a.input_dir).resolve(); out=Path(a.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
 rows=discover(inp,a.session_filter); rows=rows[:a.max_slices] if a.max_slices>0 else rows
 pipe=V4RealtimePipeline(a.model_path,a.calibration_json,a.stage2_bundle,a.grid_size,a.voxel_size_ft,a.core_size,a.batch_size,a.amp,bool(a.compile_model),bool(a.evaluate_all_cores),bool(a.gpu_coord_channels),
     a.pole_candidate_threshold,a.line_candidate_threshold,a.line_weak_threshold,a.line_competition_ratio,a.pole_min_voxels,a.line_min_voxels,a.edge_width_vox)
 manifest_path=out/'inference_manifest.csv'; timing_path=out/'realtime_slice_timing.csv'; timings=[]; manifest=[]
 if a.resume and manifest_path.is_file():
  try: manifest=pd.read_csv(manifest_path).to_dict('records')
  except EmptyDataError: manifest=[]
 if a.resume and timing_path.is_file():
  try: timings=pd.read_csv(timing_path).to_dict('records')
  except EmptyDataError: timings=[]
 completed={int(r['slice_seq']) for r in manifest if str(r.get('status',''))=='completed'}
 geo,sess=a.session_filter.split('/',1); prev=None
 for i,(seq,src,rel) in enumerate(rows,1):
  if prev is not None and seq<=prev: raise RuntimeError('Slice sequence must be strictly increasing')
  prev=seq; sid=safe_id(rel.replace('/','__')); objdir=out/'stage2_objects'/Path(rel).parent; objdir.mkdir(parents=True,exist_ok=True)
  stem=src.name[:-7] if src.name.endswith('.csv.gz') else src.stem
  polecsv=objdir/f'{stem}_poles.csv'; linecsv=objdir/f'{stem}_line_segments.csv'; vertcsv=objdir/f'{stem}_line_vertices.csv'; rowcsv=out/'csv'/Path(rel).parent/f'{stem}_v4_realtime.csv.gz'
  if seq in completed and all(x.is_file() for x in (polecsv,linecsv,vertcsv)):
   # If the process previously committed Stage 1/2 but died before rolling Stage 3,
   # repair only the missing Stage-3 snapshot. The latest_slice filter guarantees
   # that later manifest rows cannot leak into this historical reconstruction.
   if a.stage3_every_slice:
    stage3_dest=out/'stage3_incremental'/safe_id(a.session_filter)/f'slice_{seq:06d}'
    if not (stage3_dest/'COMPLETED.json').is_file():
     stage3_dest.mkdir(parents=True,exist_ok=True)
     stage3_ms=run_stage3(a,out,a.session_filter,seq,stage3_dest)
     stage3_detail=stage3_breakdown(stage3_dest,a.session_filter)
     acquired=[int(r['slice_seq']) for r in manifest if int(r['slice_seq'])<=seq and int(r['slice_seq'])>=seq-a.max_span_slices]
     atomic_json({'group_id':a.session_filter,'latest_slice':seq,'window_first_seq':seq-a.max_span_slices,'window_observed_slices':acquired,'window_observed_slice_count':len(acquired),'max_span_slices':a.max_span_slices,'max_span_length_ft':a.max_span_length_ft,'output_dir':str(stage3_dest)},out/'stage3_incremental'/safe_id(a.session_filter)/'LATEST.json')
     tm={'slice_order':i,'slice_seq':seq,'relative_path':rel,'csv_read_ms':0.0,'sparse_item_prep_ms':0.0,'stage1_wall_ms':0.0,'stage2_component_ms':0.0,'stage2_refiner_parametric_ms':0.0,'stage12_total_ms':0.0,'stage12_wall_ms':0.0,'stage3_incremental_ms':stage3_ms,**stage3_detail,
         'stage3_wrapper_overhead_ms':max(0.0,stage3_ms-stage3_detail.get('stage3_algorithm_ms',stage3_ms)),'stage3_window_observed_slices':len(acquired),'stage3_window_first_seq':seq-a.max_span_slices,'end_to_end_update_ms':stage3_ms,'stage3_output':str(stage3_dest),'resume_repair':'stage3_only'}
     timings=[r for r in timings if int(r.get('slice_seq',-1))!=seq]+[tm]; timings=sorted(timings,key=lambda r:int(r['slice_seq'])); pd.DataFrame(timings).to_csv(timing_path,index=False)
     print(f'[v4-realtime] {i}/{len(rows)} repaired missing Stage3 seq={seq} stage3={stage3_ms:.1f}ms')
    else:
     print(f'[v4-realtime] {i}/{len(rows)} reuse seq={seq}')
   else:
    print(f'[v4-realtime] {i}/{len(rows)} reuse seq={seq}')
   continue
  t0=time.perf_counter(); df=pd.read_csv(src); read_ms=(time.perf_counter()-t0)*1000
  t0=time.perf_counter(); result=pipe.process_dataframe(df,sid,seq,bool(a.write_row_csv)); stage12_ms=(time.perf_counter()-t0)*1000
  atomic_csv(result['poles'],polecsv,POLE_OUTPUT_COLUMNS); atomic_csv(result['lines'],linecsv,LINE_OUTPUT_COLUMNS); atomic_csv(result['vertices'],vertcsv,VERTEX_OUTPUT_COLUMNS)
  if a.write_row_csv:
   rowcsv.parent.mkdir(parents=True,exist_ok=True); atomic_csv(result['row_frame'],rowcsv,compression={'method':'gzip','compresslevel':1})
  c=result['center_metadata']
  if not np.isfinite([c['center_x'],c['center_y'],c['center_z']]).all():
   raise RuntimeError(f'Stage3 center metadata missing/nonfinite in {rel}; Stage1/2 completed but realtime Stage3 cannot transform local to world')
  rec={'id':sid,'source':str(src),'relative_path':rel,'geography':geo,'session':sess,'slice_seq':seq,'group_id':a.session_filter,
       **c,'output_csv':str(rowcsv) if a.write_row_csv else '','pole_csv':str(polecsv),'line_csv':str(linecsv),'line_vertices_csv':str(vertcsv),
       'rows':len(df),'accepted_poles':len(result['poles']),'accepted_line_segments':len(result['lines']),'status':'completed'}
  manifest=[r for r in manifest if int(r.get('slice_seq',-1))!=seq]+[rec]; manifest=sorted(manifest,key=lambda r:int(r['slice_seq'])); atomic_csv(pd.DataFrame(manifest),manifest_path,MANIFEST_COLUMNS)
  stage3_ms=0.; stage3_dest=''; stage3_detail={}; stage3_window_count=0; stage3_window_first_seq=seq-a.max_span_slices
  if a.stage3_every_slice:
   stage3_dest=out/'stage3_incremental'/safe_id(a.session_filter)/f'slice_{seq:06d}'; stage3_dest.mkdir(parents=True,exist_ok=True)
   stage3_ms=run_stage3(a,out,a.session_filter,seq,stage3_dest)
   stage3_detail=stage3_breakdown(stage3_dest,a.session_filter)
   acquired=[int(r['slice_seq']) for r in manifest if int(r['slice_seq'])<=seq and int(r['slice_seq'])>=seq-a.max_span_slices]
   stage3_window_count=len(acquired)
   atomic_json({'group_id':a.session_filter,'latest_slice':seq,'window_first_seq':seq-a.max_span_slices,'window_observed_slices':acquired,'window_observed_slice_count':len(acquired),'max_span_slices':a.max_span_slices,'max_span_length_ft':a.max_span_length_ft,'output_dir':str(stage3_dest)},out/'stage3_incremental'/safe_id(a.session_filter)/'LATEST.json')
  tm={'slice_order':i,'slice_seq':seq,'relative_path':rel,'csv_read_ms':read_ms,**result['timing'],'stage12_wall_ms':stage12_ms,'stage3_incremental_ms':stage3_ms,**stage3_detail,
      'stage3_wrapper_overhead_ms':max(0.0,stage3_ms-stage3_detail.get('stage3_algorithm_ms',stage3_ms)) if stage3_ms else 0.0,
      'stage3_window_observed_slices':stage3_window_count,'stage3_window_first_seq':stage3_window_first_seq,
      'end_to_end_update_ms':read_ms+stage12_ms+stage3_ms,'stage3_output':str(stage3_dest)}; timings=[r for r in timings if int(r.get('slice_seq',-1))!=seq]+[tm]; timings=sorted(timings,key=lambda r:int(r['slice_seq'])); pd.DataFrame(timings).to_csv(timing_path,index=False)
  print(f'[v4-realtime] {i}/{len(rows)} seq={seq} poles={len(result["poles"])} lines={len(result["lines"])} stage12={stage12_ms:.1f}ms stage3={stage3_ms:.1f}ms')
 report={'completed':True,'group_id':a.session_filter,'slices':len(rows),'first_slice_seq':rows[0][0],'last_slice_seq':rows[-1][0],
         'stage1_slice_local':True,'stage2_slice_local':True,'stage3_multi_slice_only':True,'stage3_rolling_past_only':True,'stage3_max_span_slices':a.max_span_slices,'stage3_max_span_length_ft':a.max_span_length_ft,
         'full_session_reconstruction_disabled':True,'model_compiled':pipe.compiled,'amp':a.amp,'evaluate_all_cores':bool(a.evaluate_all_cores),'gpu_coord_channels':bool(a.gpu_coord_channels),'stage3_execution':a.stage3_execution}
 if timings:
  x=np.array([r['end_to_end_update_ms'] for r in timings]); report['incremental_latency_ms']={'mean':float(x.mean()),'p50':float(np.quantile(x,.5)),'p95':float(np.quantile(x,.95)),'max':float(x.max())}
 atomic_json(report,out/'COMPLETED.json'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
