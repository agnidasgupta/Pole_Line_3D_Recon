#!/usr/bin/env python3
"""Small learned-bundle/runtime schema smoke test for Stage 2."""
from __future__ import annotations
import tempfile
import numpy as np,pandas as pd,joblib
from sklearn.ensemble import ExtraTreesClassifier
from v4_stage2_local import LOCAL_FEATURE_COLUMNS
from v4_stage2_runtime import load_bundle,apply_stage2

def main():
    rng=np.random.default_rng(1); X=rng.normal(size=(20,len(LOCAL_FEATURE_COLUMNS))); y=np.array([0,1]*10)
    clf=ExtraTreesClassifier(n_estimators=4,random_state=1,n_jobs=1).fit(X,y)
    with tempfile.NamedTemporaryFile(suffix='.joblib') as f:
        joblib.dump({'pole_model':clf,'line_model':clf,'pole_threshold':0.0,'line_threshold':0.0,'feature_columns':LOCAL_FEATURE_COLUMNS},f.name)
        b=load_bundle(f.name)
        base={c:0.0 for c in LOCAL_FEATURE_COLUMNS}; base.update(n_voxels=20,score_mean=.9,score_p90=.95,score_max=.99,z_span_ft=20.,horizontal_span_ft=1.,radius_p90_ft=1.,principal_verticality=.95,xy_tortuosity=1.,touches_xy_edge=False)
        pole=pd.DataFrame([{**base,'component_id':'P00001','class_name':'pole'}]); line=pd.DataFrame([{**base,'component_id':'L00001','class_name':'line','z_span_ft':1.,'horizontal_span_ft':20.,'principal_verticality':.05,'xy_tortuosity':1.05}])
        r=apply_stage2({'poles':pole,'lines':line,'pole_points':{'P00001':np.array([[10,10,z] for z in range(20)],int)},'line_points':{'L00001':np.array([[x,20,40] for x in range(20)],int)}},b,'f',1,.5)
        assert len(r['poles'])==1 and len(r['lines'])==1 and len(r['vertices'])>=2
        print(f'V4_STAGE2_RUNTIME_SMOKE_OK poles={len(r["poles"])} lines={len(r["lines"])} vertices={len(r["vertices"])}')
if __name__=='__main__': main()
