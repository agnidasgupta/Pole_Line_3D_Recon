"""Compare independent pre/post-layout processes on real and synthetic inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(x):
    import numpy as np
    import pandas as pd
    if isinstance(x, pd.DataFrame):
        return {'csv_sha256': hashlib.sha256(x.to_csv(index=False).encode()).hexdigest()}
    if isinstance(x, np.ndarray):
        return {'dtype':str(x.dtype),'shape':list(x.shape),'sha256':hashlib.sha256(x.tobytes()).hexdigest()}
    if isinstance(x, dict):
        return {k:digest(v) for k,v in x.items() if k!='timing'}
    if isinstance(x, (tuple,list)):
        return [digest(v) for v in x]
    if isinstance(x, (float,np.floating)):
        return {'float_hex':float(x).hex()}
    if isinstance(x, np.integer): return int(x)
    return x


def run_side(args):
    import numpy as np
    import pandas as pd
    sys.path[:0]=[str(args.repo/'v4'),str(args.repo/'ops/v4_stage2_stage1_electrical_v10_opt')]
    import v4_stage2_stage1_electrical_tracks_opt as stage2
    from v4_realtime_core import build_sparse_item_from_dataframe
    results={};last_profile=None
    for session in sorted(args.data_root.iterdir()):
        if not session.is_dir():continue
        for path in sorted(session.rglob('*_stage1_electrical_track_audit.json')):
            stem=path.name.removesuffix('_stage1_electrical_track_audit.json')
            poles=pd.read_csv(path.with_name(stem+'_poles.csv'))
            vox=pd.read_csv(path.with_name(stem+'_stage1_line_voxels.csv'))
            if len(poles) or len(vox)<20:continue
            audit=json.loads(path.read_text());last_profile=audit['profile']
            out=stage2.build_electrical_track_outputs(vox[['x','y','z']].to_numpy(np.int32),
                vox.v4_line_score.to_numpy(np.float32),vox.v4_deployed_label.to_numpy(np.int8),
                poles,audit['id'],audit['slice_seq'],(400,400,200),.5,last_profile)
            results[audit['id']]=digest(out)
            break
    assert len(results)==30, f'Expected one real slice in each of 30 sessions, got {len(results)}'
    processor=stage2.Stage1ElectricalTrackStage2Processor(str(args.fixtures/'MOCK_stage2_bundle.joblib'),
        str(args.fixtures/'MOCK_calibration.json'),last_profile)
    coords=[(x,y,z) for x in (40,260) for y in (49,50,51) for z in range(5,66)]
    coords += [(x,y,60) for y in (50,70) for x in range(41,260)]
    coords += [(x,160,80) for x in range(100,170)]
    item=build_sparse_item_from_dataframe(pd.DataFrame(coords,columns=['x','y','z']))
    pole=np.isin(item['coords'][:,0],[40,260]);labels=np.where(pole,1,2).astype(np.uint8)
    pred=dict(pole=np.where(pole,.95,.01).astype(np.float32),line=np.where(pole,.01,.95).astype(np.float32),
              semantic=labels,objectness=np.full(len(pole),.98,np.float32),deployed_labels=labels)
    out=processor.process(item,pred,'MOCK_SLICE',0)
    results['synthetic_full_stage2']=digest(out)
    args.output.write_text(json.dumps(results,sort_keys=True,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path)
    p.add_argument('--baseline',type=Path)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.repo:
        run_side(args);return
    if not args.baseline:p.error('--baseline is required for comparison')
    args.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    for name,repo in [('before',args.baseline),('after',root)]:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--repo',str(repo.resolve()),
            '--data-root',str(args.data_root.resolve()),'--fixtures',str(args.fixtures.resolve()),
            '--output',str((args.output/(name+'.json')).resolve())],check=True,cwd=repo)
    before=json.loads((args.output/'before.json').read_text());after=json.loads((args.output/'after.json').read_text())
    assert before==after,'Pre/post-layout outputs differ'
    report=dict(exact=True,real_line_slices=30,synthetic_full_stage2_cases=1,
                comparison='All non-timing return fields; array dtype/shape/bytes, CSV serialization, float hex, IDs/order',
                production_stage1_checkpoint_tested=False)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
