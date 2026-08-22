#!/usr/bin/env python3
import argparse, glob, json, os, shutil
from precision_common import write_json_atomic

ap=argparse.ArgumentParser()
ap.add_argument('--train_dir',required=True)
ap.add_argument('--candidate_eval_root',required=True)
ap.add_argument('--output_dir',required=True)
args=ap.parse_args()
os.makedirs(args.output_dir,exist_ok=True)
rows=[]
for p in sorted(glob.glob(os.path.join(args.candidate_eval_root,'epoch_*','full_scene_metrics.json'))):
    with open(p) as f: m=json.load(f)
    c=m['calibrated']; epoch=int(os.path.basename(os.path.dirname(p)).split('_')[-1])
    ck=os.path.join(args.train_dir,'candidates',f'epoch_{epoch:03d}.pt')
    if not os.path.exists(ck): continue
    rows.append({'epoch':epoch,'score':float(c.get('score',-1e18)),'target_gap':float(c.get('total_target_gap',1e18)),
                 'pole_precision':c['pole_precision'],'pole_recall':c['pole_recall'],'pole_iou':c['pole_iou'],
                 'line_precision':c['line_precision'],'line_recall':c['line_recall'],'line_iou':c['line_iou'],
                 'checkpoint':ck,'metrics':p,'calibration':os.path.join(os.path.dirname(p),'calibration.json')})
if not rows: raise SystemExit('No completed candidate full-validation metrics found')
rows.sort(key=lambda r:(r['score'],-r['target_gap']),reverse=True)
best=rows[0]
shutil.copy2(best['checkpoint'],os.path.join(args.output_dir,'v6_stage1_selected.pt'))
shutil.copy2(best['calibration'],os.path.join(args.output_dir,'calibration.json'))
write_json_atomic({'selected':best,'all_candidates':rows},os.path.join(args.output_dir,'candidate_selection.json'))
print(json.dumps({'selected':best},indent=2))
