#!/usr/bin/env python3
"""Synthetic contract test for rolling Stage-3: past-only, 450-ft/9-increment window, missing seq allowed."""
from __future__ import annotations
import json,tempfile
from pathlib import Path
import pandas as pd
import reconstruct_v4_stage3 as s3
import sys

POLE_COLS=['file_id','component_id','slice_seq','refiner_probability','touches_xy_edge','radius_p90_ft','verticality','base_x','base_y','base_z','top_x','top_y','top_z','height_ft','tilt_ft']
LINE_COLS=['file_id','component_id','slice_seq','refiner_probability','horizontal_span_ft','vertical_span_ft','verticality','tortuosity','vertex_count']
VERT_COLS=['file_id','component_id','slice_seq','vertex_index','x','y','z']

def main():
    with tempfile.TemporaryDirectory(prefix='v4_s3_inc_') as td:
        root=Path(td); inf=root/'inf'; inf.mkdir(); rows=[]
        for seq,cx in [(10,0.0),(12,200.0),(30,1000.0)]:
            d=inf/'stage2_objects'/'Geo'/f'session1_slice{seq}'; d.mkdir(parents=True)
            stem=f's{seq}'
            pd.DataFrame([[stem,'P00001',seq,.99,False,1.0,.99,10,10,0,10,10,60,30,0]],columns=POLE_COLS).to_csv(d/f'{stem}_poles.csv',index=False)
            pd.DataFrame(columns=LINE_COLS).to_csv(d/f'{stem}_line_segments.csv',index=False)
            pd.DataFrame(columns=VERT_COLS).to_csv(d/f'{stem}_line_vertices.csv',index=False)
            rows.append(dict(id=stem,relative_path=f'Geo/session1_slice{seq}/{stem}.csv',geography='Geo',session='session1',slice_seq=seq,group_id='Geo/session1',center_x=cx,center_y=0,center_z=0,
                             pole_csv=str(d/f'{stem}_poles.csv'),line_csv=str(d/f'{stem}_line_segments.csv'),line_vertices_csv=str(d/f'{stem}_line_vertices.csv')))
        pd.DataFrame(rows).to_csv(inf/'inference_manifest.csv',index=False)
        out=root/'out'; old=sys.argv[:]
        try:
            sys.argv=['reconstruct_v4_stage3.py','--inference_dir',str(inf),'--output_dir',str(out),'--session_filter','Geo/session1','--latest_slice','12','--max_span_slices','9','--max_span_length_ft','450','--disable_plots']
            s3.main()
        finally: sys.argv=old
        c=json.load(open(out/'COMPLETED.json')); assert c['completed'] is True; assert c['rules']['max_span_length_ft']==450.0; assert c['rules']['max_span_slices']==9; assert c['rules']['missing_intermediate_slices_allowed'] is True
        p=pd.read_csv(out/'all_world_poles.csv'); src=';'.join(p.get('source_slices',pd.Series(dtype=str)).astype(str).tolist())
        assert '30' not in src, f'future slice leaked into latest=12 reconstruction: {src}'
        assert ('10' in src) and ('12' in src), f'expected acquired seq10/12 missing: {src}'
        print('V4_STAGE3_INCREMENTAL_SMOKE_OK latest=12 included=10,12 excluded_future=30 missing_seq_11_allowed=1')

if __name__=='__main__': main()
