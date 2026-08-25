#!/usr/bin/env python3
"""Durable Stage-1/2 boundary smoke test; runs only inside the V4 Docker image."""
from __future__ import annotations
import json,tempfile
from pathlib import Path
import numpy as np,pandas as pd
from v4_stage_contracts import CONTRACT_VERSION,STAGE1_MANIFEST_COLUMNS,atomic_csv,load_stage1_artifact,save_stage1_artifact,stage1_paths,stage2_paths,upsert_manifest_row

def main():
    with tempfile.TemporaryDirectory(prefix='v4_contract_') as td:
        root=Path(td); rel='Geo/session1_slice7/example.csv'; npz,meta=stage1_paths(root,rel)
        item={'coords':np.array([[1,2,3],[4,5,6]],np.int32),'dist_values':np.array([.1,.2],np.float32),'source_rows':np.array([3,9],np.int64),'raw_labels':np.array([1,2],np.int16)}
        pred={'pole':np.array([.9,.1],np.float32),'line':np.array([.2,.8],np.float32),'semantic':np.array([1,2],np.uint8),'objectness':np.array([.7,.6],np.float32)}
        metadata={'id':'example','relative_path':rel,'geography':'Geo','session':'session1','slice_seq':7,'group_id':'Geo/session1','rows':10,'occupied_rows':2,'center_metadata':{'center_x':1.0,'center_y':2.0,'center_z':3.0}}
        save_stage1_artifact(npz,meta,item,pred,metadata); item2,pred2,meta2=load_stage1_artifact(npz,meta)
        assert meta2['contract_version']==CONTRACT_VERSION
        assert np.array_equal(item2['coords'],item['coords']); assert np.allclose(pred2['pole'],pred['pole']); assert np.allclose(pred2['line'],pred['line'])
        man=root/'stage1_manifest.csv'; row={'contract_version':CONTRACT_VERSION,'id':'example','source':'raw','relative_path':rel,'geography':'Geo','session':'session1','slice_seq':7,'group_id':'Geo/session1','center_x':1.0,'center_y':2.0,'center_z':3.0,'stage1_npz':str(npz),'stage1_meta_json':str(meta),'rows':10,'occupied_rows':2,'status':'completed'}; upsert_manifest_row(man,row,STAGE1_MANIFEST_COLUMNS)
        d=pd.read_csv(man); assert len(d)==1 and int(d.iloc[0].slice_seq)==7
        p,l,v=stage2_paths(root,rel); atomic_csv(pd.DataFrame(columns=['component_id']),p,['component_id']); atomic_csv(pd.DataFrame(columns=['component_id']),l,['component_id']); atomic_csv(pd.DataFrame(columns=['component_id']),v,['component_id'])
        assert p.is_file() and l.is_file() and v.is_file()
    print('V4_STAGE_CONTRACT_SMOKE_OK stage1_npz_roundtrip=1 stage2_paths=1 atomic_boundaries=1')
if __name__=='__main__':main()
