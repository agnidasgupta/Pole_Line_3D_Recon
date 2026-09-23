#!/usr/bin/env python3
"""Aggregate all-session V4 inference, reconstruction and timing results."""
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd


def pa():
    p=argparse.ArgumentParser(); p.add_argument('--run_root',required=True); p.add_argument('--session_table',required=True); return p.parse_args()


def qstats(s):
    x=pd.to_numeric(pd.Series(s),errors='coerce').dropna().to_numpy(float)
    if not len(x): return {"count":0,"mean":None,"p50":None,"p95":None,"max":None,"total":0.0}
    return {"count":int(len(x)),"mean":float(np.mean(x)),"p50":float(np.percentile(x,50)),"p95":float(np.percentile(x,95)),"max":float(np.max(x)),"total":float(np.sum(x))}


def safe(s): return re.sub(r'[^A-Za-z0-9_.-]+','__',str(s))


def read_json(p):
    try:return json.loads(Path(p).read_text())
    except Exception:return {}


def main():
    a=pa(); root=Path(a.run_root).resolve(); sess=pd.read_csv(a.session_table,sep='\t')
    timing_rows=[]; session_rows=[]; cms=np.zeros((3,3),dtype=np.int64); evaluated=0
    for r in sess.itertuples(index=False):
        gid=str(r.group_id); sid=safe(gid)
        s1=root/'stage1'/sid; exp=root/'stage1_inference'/sid; s2=root/'stage2'/sid; s3=root/'stage3'/sid
        # Stage 1 detailed timings from metadata JSONs.
        s1rows=[]
        if (s1/'stage1_manifest.csv').is_file():
            m=pd.read_csv(s1/'stage1_manifest.csv')
            for z in m.itertuples(index=False):
                md=read_json(z.stage1_meta_json); t=dict(md.get('timing',{})); row={'group_id':gid,'slice_seq':int(z.slice_seq),**t}; s1rows.append(row)
        # Parse artifact-write timing from Stage1 log (accepted runner prints it).
        writes={}
        log=root/'logs'/'stage1'/f'{sid}.log'
        if log.is_file():
            pat=re.compile(r'seq=(\d+).*?artifact_write=([0-9.]+)ms')
            for line in log.read_text(errors='replace').splitlines():
                m=pat.search(line)
                if m:writes[int(m.group(1))]=float(m.group(2))
        for row in s1rows:
            row['stage1_artifact_write_ms']=writes.get(row['slice_seq'],math.nan)
            timing_rows.append(row)
        s2tim=root/'timings'/'stage2'/f'{sid}.csv'
        s3tim=root/'timings'/'stage3'/f'{sid}.csv'
        d2=pd.read_csv(s2tim) if s2tim.is_file() else pd.DataFrame()
        d3=pd.read_csv(s3tim) if s3tim.is_file() else pd.DataFrame()
        # Merge Stage2/3 timing fields into matching all-slice timing rows.
        index={(x['group_id'],int(x['slice_seq'])):x for x in timing_rows}
        for d in (d2,d3):
            for z in d.to_dict('records'):
                key=(gid,int(z['slice_seq'])); index.setdefault(key,{'group_id':gid,'slice_seq':int(z['slice_seq'])}).update(z)
        # Stage 1 metrics.
        ms=read_json(exp/'stage1_metrics_summary.json')
        cm=np.asarray(ms.get('confusion_matrix_rows_true_cols_pred',np.zeros((3,3))),dtype=np.int64)
        if cm.shape==(3,3): cms+=cm; evaluated+=int(cm.sum())
        # Stage2 counts.
        poles=lines=0
        if (s2/'inference_manifest.csv').is_file():
            m2=pd.read_csv(s2/'inference_manifest.csv'); poles=int(pd.to_numeric(m2.get('accepted_poles',0),errors='coerce').fillna(0).sum()); lines=int(pd.to_numeric(m2.get('accepted_line_segments',0),errors='coerce').fillna(0).sum())
        # Final Stage3 summary for session.
        final_summary={}; final_seq=None
        cand=sorted((s3/'stage3_incremental'/Path(gid)).glob('slice_*')) if (s3/'stage3_incremental'/Path(gid)).exists() else []
        if cand:
            final=cand[-1]; final_seq=int(final.name.split('_')[-1]); final_summary=read_json(final/'sessions'/Path(gid)/'summary.json')
        session_rows.append({
            'group_id':gid,'source_slices':int(r.slice_count),'min_seq':int(r.min_seq),'max_seq':int(r.max_seq),'missing_sequence_count':int(r.missing_sequence_count),
            'stage1_evaluated_rows':int(ms.get('evaluated_rows',0) or 0),'stage1_accuracy':ms.get('accuracy'),
            'stage2_accepted_poles_total':poles,'stage2_accepted_line_segments_total':lines,'stage3_final_slice_seq':final_seq,
            'stage3_final_merged_poles':final_summary.get('merged_poles'),'stage3_final_conductor_chains':final_summary.get('conductor_chains'),
            'stage3_final_spans':final_summary.get('spans'),'stage3_final_partial_spans':final_summary.get('partial_spans'),
            'stage3_final_open_conductors':final_summary.get('open_conductors'),'stage3_final_hidden_poles':final_summary.get('inferred_hidden_poles'),
        })
    t=pd.DataFrame(timing_rows)
    tdir=root/'timings'; mdir=root/'metrics'; tdir.mkdir(exist_ok=True); mdir.mkdir(exist_ok=True)
    if not t.empty: t.sort_values(['group_id','slice_seq']).to_csv(tdir/'all_slice_timings.csv',index=False)
    sr=pd.DataFrame(session_rows); sr.to_csv(mdir/'reconstruction_metrics_by_session.csv',index=False)
    # Aggregate confusion metrics.
    classes=[]
    for c,name in [(0,'background'),(1,'pole'),(2,'line')]:
        tp=int(cms[c,c]); fp=int(cms[:,c].sum()-tp); fn=int(cms[c,:].sum()-tp); precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.; f1=2*precision*recall/(precision+recall) if precision+recall else 0.; iou=tp/(tp+fp+fn) if tp+fp+fn else 0.; classes.append({'class':c,'name':name,'support':int(cms[c,:].sum()),'precision':precision,'recall':recall,'f1':f1,'iou':iou})
    metric_summary={'metric_scope':'occupied valid voxels only','evaluated_rows':int(evaluated),'accuracy':float(np.trace(cms)/evaluated) if evaluated else None,'confusion_matrix_rows_true_cols_pred':cms.tolist(),'classes':classes}
    (mdir/'stage1_metrics_overall.json').write_text(json.dumps(metric_summary,indent=2,sort_keys=True))
    # Timing summary for all numeric timing columns.
    timing_summary={}
    if not t.empty:
        for c in t.columns:
            if c in ('group_id','slice_seq','resume_from'):continue
            vals=pd.to_numeric(t[c],errors='coerce')
            if vals.notna().any(): timing_summary[c]=qstats(vals)
    (tdir/'timing_overall.json').write_text(json.dumps(timing_summary,indent=2,sort_keys=True))
    by=[]
    if not t.empty:
        for gid,g in t.groupby('group_id'):
            row={'group_id':gid,'slices':int(g.slice_seq.nunique())}
            for c in ('stage1_wall_ms','stage2_total_ms','stage3_elapsed_ms','stage3_fragment_join_ms','stage3_span_completion_pre_ms','stage3_chain_build_attachment_ms'):
                if c in g:
                    st=qstats(g[c]); row.update({f'{c}_mean':st['mean'],f'{c}_p50':st['p50'],f'{c}_p95':st['p95'],f'{c}_max':st['max']})
            by.append(row)
    pd.DataFrame(by).to_csv(tdir/'timing_by_session.csv',index=False)
    summary={'completed':True,'sessions':int(len(sess)),'source_slices':int(sess.slice_count.sum()),'evaluated_rows':int(evaluated),'stage1_accuracy':metric_summary['accuracy'],'timing_columns':timing_summary,'final_reconstruction_totals':{
        'stage2_accepted_poles':int(pd.to_numeric(sr.stage2_accepted_poles_total,errors='coerce').fillna(0).sum()) if not sr.empty else 0,
        'stage2_accepted_line_segments':int(pd.to_numeric(sr.stage2_accepted_line_segments_total,errors='coerce').fillna(0).sum()) if not sr.empty else 0,
    }}
    (root/'FULL_DATASET_SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True))
    lines=[f"sessions={summary['sessions']}",f"source_slices={summary['source_slices']}",f"evaluated_rows={summary['evaluated_rows']}",f"stage1_accuracy={summary['stage1_accuracy']}"]
    for k in ('stage1_wall_ms','stage2_total_ms','stage3_elapsed_ms'):
        if k in timing_summary:
            st=timing_summary[k]; lines.append(f"{k}: mean={st['mean']:.3f} p50={st['p50']:.3f} p95={st['p95']:.3f} max={st['max']:.3f}")
    (root/'FULL_DATASET_SUMMARY.txt').write_text('\n'.join(lines)+'\n')
    print('V4_ALL_DATA_SUMMARY_OK')

if __name__=='__main__': main()
