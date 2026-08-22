#!/usr/bin/env python3
"""V6.2 Stage 1 + Stage 2 inference.

Requires only x,y,z. Optional label and dist_center_ft are used for metrics/features.
center_* and world_* are deliberately ignored and never required. Stage 2 emits
slice-local parametric poles and conductor line segments; line segments are NOT
required to touch a visible pole.
"""
from __future__ import annotations
import argparse,json,os,re,time
from pathlib import Path
import joblib,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
from pandas.errors import EmptyDataError
from sklearn.metrics import confusion_matrix
from v6_components import atomic_json,extract_all_components
from v6_predict import load_v6_model,predict_dense_scores,setup_torch
from v62_local import LOCAL_FEATURE_COLUMNS,add_edge_features,pole_local_physical_ok,line_local_physical_ok
SESSION_RE=re.compile(r"^(session\d+)_slice(\d+)$",re.I)

def pa():
 p=argparse.ArgumentParser(); p.add_argument('--input_dir',required=True); p.add_argument('--output_dir',required=True); p.add_argument('--model_path',required=True); p.add_argument('--calibration_json',required=True); p.add_argument('--local_refiner_bundle',required=True); p.add_argument('--manifest_json',default=None); p.add_argument('--grid_size',type=int,nargs=3,default=[400,400,200]); p.add_argument('--voxel_size_ft',type=float,default=.5); p.add_argument('--pole_candidate_threshold',type=float,default=.15); p.add_argument('--line_candidate_threshold',type=float,default=.08); p.add_argument('--line_weak_threshold',type=float,default=.04); p.add_argument('--line_competition_ratio',type=float,default=.55); p.add_argument('--line_min_voxels',type=int,default=3); p.add_argument('--edge_width_vox',type=int,default=10); p.add_argument('--core_size',type=int,default=48); p.add_argument('--batch_size',type=int,default=5); p.add_argument('--build_workers',type=int,default=6); p.add_argument('--amp',default='bf16'); p.add_argument('--compile_model',type=int,default=1); p.add_argument('--resume',type=int,default=1); p.add_argument('--missing_source_policy',choices=['skip','error'],default='skip'); p.add_argument('--compression_level',type=int,default=1); p.add_argument('--max_files',type=int,default=0); return p.parse_args()
def safe_id(rel): return re.sub(r'\.csv(?:\.gz)?$','',str(rel).replace('/','__').replace('\\','__'),flags=re.I)
def parse_group(rel):
 parts=Path(rel).parts; geo=parts[0] if len(parts)>=3 else 'ungrouped'; parent=parts[-2] if len(parts)>=2 else 'session0_slice0'; m=SESSION_RE.match(parent); return (geo,m.group(1).lower(),int(m.group(2)),f'{geo}/{m.group(1).lower()}') if m else (geo,'session0',0,f'{geo}/session0')
def discover(a):
 root=Path(a.input_dir).resolve(); rows=[]; skipped=[]
 if a.manifest_json:
  for r in json.load(open(a.manifest_json)):
   rel=r.get('source_relpath',''); src=root/rel if rel else Path(r.get('source_csv',''))
   if not src.exists():
    skipped.append({'id':r.get('id',safe_id(rel)),'relative_path':rel,'reason':'missing_manifest_source'});
    if a.missing_source_policy=='error': raise FileNotFoundError(src)
    continue
   geo,sess,seq,gid=parse_group(rel); rows.append({'source':str(src),'relative_path':rel,'id':r.get('id',safe_id(rel)),'geography':geo,'session':sess,'slice_seq':seq,'group_id':r.get('group_id',gid)})
 else:
  for src in sorted(root.rglob('*.csv'))+sorted(root.rglob('*.csv.gz')):
   rel=str(src.relative_to(root)); geo,sess,seq,gid=parse_group(rel); rows.append({'source':str(src),'relative_path':rel,'id':safe_id(rel),'geography':geo,'session':sess,'slice_seq':seq,'group_id':gid})
 return (rows[:a.max_files] if a.max_files>0 else rows),skipped
def build_item(df,grid):
 gx,gy,gz=grid
 if not {'x','y','z'}.issubset(df.columns): raise ValueError('Stage 1 requires only x,y,z; one or more are missing')
 arr=df[['x','y','z']].apply(pd.to_numeric,errors='coerce').to_numpy(float); valid=np.isfinite(arr).all(1); coords=np.zeros((len(df),3),np.int32); coords[valid]=np.rint(arr[valid]).astype(np.int32); valid&=(coords[:,0]>=0)&(coords[:,0]<gx)&(coords[:,1]>=0)&(coords[:,1]<gy)&(coords[:,2]>=0)&(coords[:,2]<gz)
 occ=np.zeros((gz,gy,gx),np.float32); lab=np.full((gz,gy,gx),-100,np.int16); dist=np.zeros((gz,gy,gx),np.float32); c=coords[valid]; occ[c[:,2],c[:,1],c[:,0]]=1
 has_gt='label' in df.columns; rowlab=np.full(len(df),-100,np.int16)
 if has_gt:
  y=pd.to_numeric(df.label,errors='coerce').to_numpy(float); good=valid&np.isfinite(y)&np.isin(y,[0,1,2]); rowlab[good]=y[good].astype(np.int16); cg=coords[good]; lab[cg[:,2],cg[:,1],cg[:,0]]=rowlab[good]
 else: lab[c[:,2],c[:,1],c[:,0]]=0
 if 'dist_center_ft' in df.columns:
  d=pd.to_numeric(df.dist_center_ft,errors='coerce').to_numpy(float); finite=np.isfinite(d); den=max(float(np.max(np.abs(d[finite]))) if finite.any() else 1.,1.); good=valid&finite; cg=coords[good]; dist[cg[:,2],cg[:,1],cg[:,0]]=(d[good]/den).astype(np.float32)
 return {'occ':occ,'labels':lab,'hardneg':np.zeros_like(occ),'dist':dist,'coords':c,'row_coords':coords,'valid_rows':valid,'row_labels':rowlab,'has_gt':has_gt}
def label_scores(ps,ls,pt,lt):
 o=np.zeros(len(ps),np.int8); p=ps>=pt; l=ls>=lt; o[p&~l]=1; o[l&~p]=2; b=p&l; o[b&(ps>=ls)]=1; o[b&(ls>ps)]=2; return o
def local_X(d): return d.reindex(columns=LOCAL_FEATURE_COLUMNS,fill_value=0).replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float)
def order_line(pts):
 p=np.asarray(pts,float)
 if len(p)<2:return p
 xy=p[:,:2]; c=np.median(xy,0)
 try: _,_,vh=np.linalg.svd(xy-c,full_matrices=False); d=vh[0]
 except np.linalg.LinAlgError: d=np.array([1.,0.])
 s=(xy-c)@(d/max(np.linalg.norm(d),1e-9)); return p[np.argsort(s)]
def pole_param(pts,voxel=.5):
 p=np.asarray(pts,float); z=p[:,2]; lo=np.quantile(z,.15); hi=np.quantile(z,.85); b=np.median(p[z<=lo],0); t=np.median(p[z>=hi],0); return {'base_x':b[0],'base_y':b[1],'base_z':p[:,2].min(),'top_x':t[0],'top_y':t[1],'top_z':p[:,2].max(),'height_ft':(p[:,2].max()-p[:,2].min()+1)*voxel,'tilt_ft':np.linalg.norm(t[:2]-b[:2])*voxel}
def line_vertices(pts,bin_vox=2.):
 p=order_line(pts)
 if len(p)<2:return p
 xy=p[:,:2]; c=np.median(xy,0); _,_,vh=np.linalg.svd(xy-c,full_matrices=False); d=vh[0]/max(np.linalg.norm(vh[0]),1e-9); s=(xy-c)@d; bins=np.floor((s-s.min())/bin_vox).astype(int); rows=[]
 for b in np.unique(bins): rows.append(np.median(p[bins==b],0))
 return np.asarray(rows,float)
def cm_metrics(cm):
 out={}
 for c,n in [(1,'pole'),(2,'line')]:
  tp=cm[c,c]; fp=cm[:,c].sum()-tp; fn=cm[c,:].sum()-tp; out[n]={'precision':float(tp/max(tp+fp,1)),'recall':float(tp/max(tp+fn,1)),'iou':float(tp/max(tp+fp+fn,1)),'tp':int(tp),'fp':int(fp),'fn':int(fn)}
 return out
def plot_cm(cm,path,title):
 fig,ax=plt.subplots(figsize=(6,5)); im=ax.imshow(cm); fig.colorbar(im,ax=ax); names=['other','pole','line']; ax.set_xticks(range(3),names); ax.set_yticks(range(3),names); ax.set_xlabel('Predicted'); ax.set_ylabel('GT'); ax.set_title(title)
 for i in range(3):
  for j in range(3): ax.text(j,i,f'{int(cm[i,j]):,}',ha='center',va='center')
 fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)

POLE_OUTPUT_COLUMNS=['file_id','component_id','slice_seq','refiner_probability','touches_xy_edge','radius_p90_ft','verticality','base_x','base_y','base_z','top_x','top_y','top_z','height_ft','tilt_ft']
LINE_OUTPUT_COLUMNS=['file_id','component_id','slice_seq','refiner_probability','horizontal_span_ft','vertical_span_ft','verticality','tortuosity','vertex_count']
VERTEX_OUTPUT_COLUMNS=['file_id','component_id','slice_seq','vertex_index','x','y','z']
SKIPPED_OUTPUT_COLUMNS=['id','relative_path','reason']
COMPONENT_EMPTY_COLUMNS=['file_id','slice_seq','relative_path','component_id','refiner_probability','physical_ok','component_accept','accept_mode']+list(LOCAL_FEATURE_COLUMNS)

def write_df(obj,path,columns=None,**kwargs):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 d=obj.copy() if isinstance(obj,pd.DataFrame) else pd.DataFrame(obj)
 if d.empty and columns is not None: d=pd.DataFrame(columns=columns)
 d.to_csv(path,index=False,**kwargs)

def progress(out,i,total,last,started):
 atomic_json({'completed_files':i,'total_files':total,'last_file':last,'elapsed_seconds':time.time()-started},Path(out)/'progress.json')
def main():
 a=pa(); setup_torch(); out=Path(a.output_dir); csvroot=out/'csv'; objroot=out/'stage2_objects'; pointroot=out/'stage2_points'; [x.mkdir(parents=True,exist_ok=True) for x in (csvroot,objroot,pointroot)]
 records,skipped=discover(a); write_df(skipped,out/'skipped_missing_sources.csv',SKIPPED_OUTPUT_COLUMNS); cal=json.load(open(a.calibration_json)); bundle=joblib.load(a.local_refiner_bundle); model,cfg,compiled=load_v6_model(a.model_path,compile_model=bool(a.compile_model)); print('files',len(records),'compile',compiled)
 cms={'stage1':np.zeros((3,3),np.int64),'stage2':np.zeros((3,3),np.int64)}; gtfiles=0; manifest=[]; started=time.time(); pt=float(cal['pole_threshold']); lt=float(cal['line_threshold'])
 for i,r in enumerate(records,1):
  rel=Path(r['relative_path']); stem=rel.name[:-7] if rel.name.endswith('.csv.gz') else rel.stem; op=csvroot/rel.parent/(stem+'_v62_inference.csv.gz'); od=objroot/rel.parent; pp=pointroot/rel.parent/(stem+'_points.npz'); op.parent.mkdir(parents=True,exist_ok=True); od.mkdir(parents=True,exist_ok=True); pp.parent.mkdir(parents=True,exist_ok=True); polecsv=od/(stem+'_poles.csv'); linecsv=od/(stem+'_line_segments.csv'); vertcsv=od/(stem+'_line_vertices.csv')
  if a.resume and all(x.exists() for x in (op,polecsv,linecsv,vertcsv,pp)):
   manifest.append({**r,'output_csv':str(op),'pole_csv':str(polecsv),'line_csv':str(linecsv),'line_vertices_csv':str(vertcsv),'points_npz':str(pp),'status':'resumed'}); print('[infer-v62]',i,'resume',r['relative_path']); continue
  try:
   raw=pd.read_csv(r['source'])
  except EmptyDataError:
   skipped.append({'id':r['id'],'relative_path':r['relative_path'],'reason':'empty_source_csv'}); progress(out,i,len(records),r['relative_path'],started); print('[infer-v62]',i,'skip empty',r['relative_path']); continue
  if raw.empty:
   skipped.append({'id':r['id'],'relative_path':r['relative_path'],'reason':'header_only_source_csv'}); progress(out,i,len(records),r['relative_path'],started); print('[infer-v62]',i,'skip header-only',r['relative_path']); continue
  # world/center columns may exist but are never referenced below.
  item=build_item(raw,tuple(a.grid_size)); scores=predict_dense_scores(item,model,cfg,tuple(a.grid_size),a.core_size,a.batch_size,a.build_workers,a.amp,bool(int(cfg.get('channels_last',1))),float(cfg.get('score_sem_weight',.55)),float(cfg.get('score_binary_weight',.35)),float(cfg.get('score_object_weight',.10)))
  comps=extract_all_components(item['occ'],scores,a.pole_candidate_threshold,a.line_candidate_threshold,a.voxel_size_ft,item['labels'] if item['has_gt'] else None,line_min_voxels=a.line_min_voxels,attachment_radius_ft=9999.,line_weak_threshold=a.line_weak_threshold,line_competition_ratio=a.line_competition_ratio)
  accepted={}; objframes=[]; polerows=[]; linerows=[]; vertices=[]
  for name,df,points in [('pole',comps['poles'],comps['pole_points']),('line',comps['lines'],comps['line_points'])]:
   d=add_edge_features(df,points,a.grid_size,a.edge_width_vox) if not df.empty else df.copy()
   if not d.empty:
    clf=bundle[name+'_model']; thr=float(bundle[name+'_threshold']); prob=clf.predict_proba(local_X(d))[:,1]; physical=np.array([pole_local_physical_ok(x) if name=='pole' else line_local_physical_ok(x) for _,x in d.iterrows()]); acc=(prob>=thr)&physical; d['refiner_probability']=prob; d['physical_ok']=physical; d['component_accept']=acc; d['accept_mode']=np.where(acc,'refiner_plus_loose_hard_gate','rejected'); d['file_id']=r['id']; d['slice_seq']=r['slice_seq']; d['relative_path']=r['relative_path']; objframes.append(d)
    for _,x in d[d.component_accept].iterrows():
     key=str(x.component_id); accepted[(name,int(key[1:]))]=float(x.refiner_probability); pts=points[key]
     if name=='pole': polerows.append({'file_id':r['id'],'component_id':key,'slice_seq':r['slice_seq'],'refiner_probability':x.refiner_probability,'touches_xy_edge':bool(x.touches_xy_edge),'radius_p90_ft':x.radius_p90_ft,'verticality':x.principal_verticality,**pole_param(pts,a.voxel_size_ft)})
     else:
      vv=line_vertices(pts); lid=key; linerows.append({'file_id':r['id'],'component_id':lid,'slice_seq':r['slice_seq'],'refiner_probability':x.refiner_probability,'horizontal_span_ft':x.horizontal_span_ft,'vertical_span_ft':x.z_span_ft,'verticality':x.principal_verticality,'tortuosity':x.xy_tortuosity,'vertex_count':len(vv)})
      for vi,q in enumerate(vv): vertices.append({'file_id':r['id'],'component_id':lid,'slice_seq':r['slice_seq'],'vertex_index':vi,'x':q[0],'y':q[1],'z':q[2]})
  write_df(polerows,polecsv,POLE_OUTPUT_COLUMNS); write_df(linerows,linecsv,LINE_OUTPUT_COLUMNS); write_df(vertices,vertcsv,VERTEX_OUTPUT_COLUMNS)
  allobjs=pd.concat(objframes,ignore_index=True) if objframes else pd.DataFrame(columns=COMPONENT_EMPTY_COLUMNS); write_df(allobjs,od/(stem+'_components.csv.gz'),COMPONENT_EMPTY_COLUMNS,compression='gzip')
  # Preserve local points only; Stage 3 transforms them later when center metadata is supplied.
  pcoords=[];pids=[];lcoords=[];lids=[]
  for k,v in comps['pole_points'].items(): pcoords.append(v); pids.append(np.full(len(v),int(k[1:]),np.int32))
  for k,v in comps['line_points'].items(): lcoords.append(v); lids.append(np.full(len(v),int(k[1:]),np.int32))
  np.savez_compressed(pp,pole_coords=np.vstack(pcoords).astype(np.int16) if pcoords else np.zeros((0,3),np.int16),pole_component=np.concatenate(pids) if pids else np.zeros(0,np.int32),line_coords=np.vstack(lcoords).astype(np.int16) if lcoords else np.zeros((0,3),np.int16),line_component=np.concatenate(lids) if lids else np.zeros(0,np.int32))
  rowc=item['row_coords']; valid=item['valid_rows']; idx=np.flatnonzero(valid); c=rowc[valid]; ps=np.zeros(len(raw),np.float32);ls=np.zeros(len(raw),np.float32);sem=np.full(len(raw),-1,np.int8);pc=np.zeros(len(raw),np.int32);lc=np.zeros(len(raw),np.int32)
  ps[idx]=scores['pole'][c[:,2],c[:,1],c[:,0]]; ls[idx]=scores['line'][c[:,2],c[:,1],c[:,0]]; sem[idx]=scores['semantic'][c[:,2],c[:,1],c[:,0]]; pc[idx]=comps['pole_labels'][c[:,2],c[:,1],c[:,0]]; lc[idx]=comps['line_labels'][c[:,2],c[:,1],c[:,0]]; s1=label_scores(ps,ls,pt,lt); s1[~valid]=-1; s2=np.zeros(len(raw),np.int8)
  for j in idx:
   kp=('pole',int(pc[j])); kl=('line',int(lc[j])); pprob=accepted.get(kp,0.); lprob=accepted.get(kl,0.); s2[j]=1 if pprob>0 and pprob>=lprob else (2 if lprob>0 else 0)
  s2[~valid]=-1; raw['v62_pole_score']=ps.astype(np.float16); raw['v62_line_score']=ls.astype(np.float16); raw['v62_stage1_label']=s1; raw['v62_pole_component']=pc; raw['v62_line_component']=lc; raw['v62_stage2_label']=s2; op.parent.mkdir(parents=True,exist_ok=True); raw.to_csv(op,index=False,compression={'method':'gzip','compresslevel':a.compression_level})
  if item['has_gt']:
   y=item['row_labels']; good=np.isin(y,[0,1,2])&np.isin(s1,[0,1,2])&np.isin(s2,[0,1,2])
   if np.any(good):
    cms['stage1']+=confusion_matrix(y[good],s1[good],labels=[0,1,2]); cms['stage2']+=confusion_matrix(y[good],s2[good],labels=[0,1,2]); gtfiles+=1
   else:
    skipped.append({'id':r['id'],'relative_path':r['relative_path'],'reason':'no_valid_gt_rows_for_metrics'})
  manifest.append({**r,'output_csv':str(op),'pole_csv':str(polecsv),'line_csv':str(linecsv),'line_vertices_csv':str(vertcsv),'points_npz':str(pp),'rows':len(raw),'accepted_poles':len(polerows),'accepted_line_segments':len(linerows),'status':'completed'}); progress(out,i,len(records),r['relative_path'],started); print(f"[infer-v62] {i}/{len(records)} {r['relative_path']} poles={len(polerows)} lines={len(linerows)}")
 write_df(skipped,out/'skipped_missing_sources.csv',SKIPPED_OUTPUT_COLUMNS); write_df(manifest,out/'inference_manifest.csv',['id','source','relative_path','geography','session','slice_seq','group_id','output_csv','pole_csv','line_csv','line_vertices_csv','points_npz','rows','accepted_poles','accepted_line_segments','status']); report={'completed':True,'files':len(manifest),'files_with_gt':gtfiles,'note':'Strict metrics use supplied GT and can understate precision where labels are incomplete/incorrect. Stage 1 and Stage 2 never use center/world columns. Stage 2 uses strong/weak hysteresis for line candidates and only a loose impossible-geometry hard gate.','line_candidate_hysteresis':{'strong_threshold':a.line_candidate_threshold,'weak_threshold':a.line_weak_threshold,'competition_ratio':a.line_competition_ratio,'min_voxels':a.line_min_voxels},'stages':{}}
 for k,cm in cms.items(): report['stages'][k]={'cm':cm.tolist(),**cm_metrics(cm)}; plot_cm(cm,out/f'confusion_matrix_{k}.png',f'V6.2 {k} strict GT')
 atomic_json(report,out/'inference_metrics.json'); (out/'inference_metrics.txt').write_text(json.dumps(report,indent=2)); atomic_json({'completed':True,'files':len(manifest)},out/'COMPLETED.json'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
