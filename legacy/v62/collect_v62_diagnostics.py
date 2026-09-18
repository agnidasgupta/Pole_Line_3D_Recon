#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v6_components import atomic_json
p=argparse.ArgumentParser();p.add_argument('--stage1_test',required=True);p.add_argument('--stage2_metrics',required=True);p.add_argument('--inference_metrics',required=True);p.add_argument('--stage3_completed',required=True);p.add_argument('--output_dir',required=True);a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
def rd(x):return json.load(open(x))
s1=rd(a.stage1_test);s2=rd(a.stage2_metrics);inf=rd(a.inference_metrics);s3=rd(a.stage3_completed)
report={'stage1_strict_gt':s1,'stage2_local_refiner':s2,'all_inference_strict_gt':inf,'stage3_reconstruction':s3,'interpretation':'Strict GT metrics can understate precision because reviewed catenary/pole labels contain omissions/mistakes. Stage-2 plausible GT disagreements were excluded from negative training.'};atomic_json(report,out/'v62_diagnostics.json');(out/'v62_diagnostics.txt').write_text(json.dumps(report,indent=2))
vals=[];labs=[]
for stage in ('stage1','stage2'):
 d=inf.get('stages',{}).get(stage,{})
 for cls in ('pole','line'):
  m=d.get(cls,{})
  for metric in ('precision','recall','iou'):labs.append(f'{stage}\n{cls} {metric}');vals.append(float(m.get(metric,0)))
plt.figure(figsize=(14,6));plt.bar(range(len(vals)),vals);plt.xticks(range(len(vals)),labs,rotation=60,ha='right');plt.ylim(0,1);plt.ylabel('strict GT metric');plt.tight_layout();plt.savefig(out/'strict_gt_metric_summary.png',dpi=180);plt.close()
print(json.dumps({'completed':True,'output_dir':str(out)},indent=2))
