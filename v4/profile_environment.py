#!/usr/bin/env python3
from __future__ import annotations
import json,platform,subprocess,sys
from pathlib import Path
import torch

def cmd(args):
 try:return subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=False).stdout.strip()
 except Exception as e:return f'ERROR: {e!r}'

def main():
 out=Path(sys.argv[1] if len(sys.argv)>1 else 'environment_profile.json'); out.parent.mkdir(parents=True,exist_ok=True)
 d={'python':sys.version,'platform':platform.platform(),'torch_version':torch.__version__,'torch_cuda_version':torch.version.cuda,'torch_cudnn_version':torch.backends.cudnn.version(),'cuda_available':torch.cuda.is_available()}
 d['nvcc_version']=cmd(['bash','-lc','nvcc --version 2>/dev/null || true'])
 if torch.cuda.is_available():
  pr=torch.cuda.get_device_properties(0); d.update({'gpu_name':pr.name,'gpu_compute_capability':list(torch.cuda.get_device_capability(0)),'gpu_memory_gb':pr.total_memory/2**30,'gpu_multiprocessor_count':pr.multi_processor_count})
 d['spconv_used_by_production_model']=False
 d['nvidia_smi_query']=cmd(['nvidia-smi','--query-gpu=name,pci.bus_id,memory.total,power.limit,driver_version','--format=csv,noheader'])
 d['nvidia_smi_q']=cmd(['nvidia-smi','-q'])
 q=d['nvidia_smi_q']
 def qfield(label):
  import re
  m=re.search(r'^\s*'+re.escape(label)+r'\s*:\s*(.+)$',q,re.M)
  return m.group(1).strip() if m else None
 d['gpu_product_name_nvidia_smi']=qfield('Product Name')
 d['gpu_board_part_number']=qfield('GPU Part Number') or qfield('Board Part Number')
 d['gpu_max_power_limit']=qfield('Max Power Limit')
 d['nvidia_smi_topo']=cmd(['nvidia-smi','topo','-m'])
 d['nsys_version']=cmd(['nsys','--version']) if cmd(['bash','-lc','command -v nsys']) else 'not_installed'
 ncu_path=cmd(['bash','-lc','command -v ncu || find /usr/local/cuda* /opt/nvidia/nsight-compute -type f -name ncu -perm -111 2>/dev/null | head -n 1'])
 d['ncu_path']=ncu_path or 'not_installed'
 d['ncu_version']=cmd([ncu_path,'--version']) if ncu_path else 'not_installed'
 out.write_text(json.dumps(d,indent=2)); print(json.dumps(d,indent=2))
if __name__=='__main__':main()
