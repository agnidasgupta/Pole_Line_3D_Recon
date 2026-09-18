#!/usr/bin/env python3
"""Create a compact V4-vs-teacher-recall V6.2 strict-GT comparison."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('--v4_metrics',required=True)
p.add_argument('--v62_metrics',required=True)
p.add_argument('--output_dir',required=True)
a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)

def load(path): return json.load(open(path))
v4=load(a.v4_metrics); v6=load(a.v62_metrics)
v4c=v4.get('calibrated',v4.get('validation_metrics',v4))
v6c=v6.get('calibrated',v6.get('validation_metrics',v6))
rows=[]
for model,d in [('V4',v4c),('V6.2 teacher-recall',v6c)]:
    for cls in ('pole','line'):
        rows.append({'model':model,'class':cls,
                     'precision':float(d.get(f'{cls}_precision',0)),
                     'recall':float(d.get(f'{cls}_recall',0)),
                     'iou':float(d.get(f'{cls}_iou',0))})
df=pd.DataFrame(rows); df.to_csv(out/'v4_vs_v62_strict_gt_metrics.csv',index=False)
plot=df[df['class']=='line'].set_index('model')[['precision','recall','iou']]
ax=plot.plot(kind='bar',figsize=(9,5),rot=0); ax.set_ylim(0,1); ax.set_ylabel('strict GT metric'); ax.set_title('V4 vs V6.2 teacher-recall: powerline voxels'); ax.grid(axis='y',alpha=.3); plt.tight_layout(); plt.savefig(out/'v4_vs_v62_line_metrics.png',dpi=180); plt.close()
summary={'completed':True,'v4_metrics':a.v4_metrics,'v62_metrics':a.v62_metrics,'rows':rows,
         'interpretation':'Strict GT can understate precision where reviewed labels omit or misalign legitimate conductors. Line recall is the primary teacher-recall comparison metric.'}
(out/'v4_vs_v62_line_metrics.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
