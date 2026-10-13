import argparse,json,hashlib,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();root=a.experiment
rows=[json.loads(x) for x in (root/'gradient_records.jsonl').read_text().splitlines() if x.strip()]
if not rows:raise RuntimeError('No measured gradients to plot')
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
out=root/'curves';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
keys=list(rows[0]['gradient_norms'])
for k in keys:axes[0].plot([r['index'] for r in rows],[r['weighted_gradient_norms'][k] for r in rows],marker='.',label=k)
axes[0].set(yscale='log',xlabel='实测样本序号（不是训练步）',ylabel='加权梯度范数',title='各目标的梯度贡献');axes[0].legend();axes[0].grid(alpha=.2)
matrix=np.eye(len(keys))
for i in range(len(keys)):
 for j in range(i+1,len(keys)):
  values=[r['pairwise_cosine'][keys[i]+'__'+keys[j]] for r in rows if r['role']=='calibration']
  matrix[i,j]=matrix[j,i]=np.mean(values)
im=axes[1].imshow(matrix,vmin=-1,vmax=1,cmap='RdBu')
axes[1].set(xticks=range(len(keys)),yticks=range(len(keys)),xticklabels=keys,yticklabels=keys,title='训练样本的目标梯度余弦均值')
for i in range(len(keys)):
 for j in range(len(keys)):axes[1].text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',color='white' if abs(matrix[i,j])>.55 else 'black')
fig.colorbar(im,ax=axes[1]);fig.suptitle('梯度诊断：仅局部代理，未更新模型，不代表轨迹 ADE')
files=[]
for suffix in ['png','svg']:
 path=out/('gradient_diagnostic.'+suffix);fig.savefig(path,dpi=180);files.append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
plt.close(fig)
with (out/'gradient_cosines.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['角色','片段','探针','目标对','梯度余弦'])
 for r in rows:
  for k,v in r['pairwise_cosine'].items():w.writerow([r['role'],r['clip_id'],r['expert_probe_index'],k,v])
(root/'curves/gradient_manifest.json').write_text(json.dumps({'language':'zh','files':files,'input_sha256':hashlib.sha256((root/'gradient_records.jsonl').read_bytes()).hexdigest(),'actual_sampling':'12 deterministic training clips and 8 independent monitoring clips; one balanced probe per sample','semantics':'Local adapter gradient geometry; no trajectory evaluation; no training updates'},indent=2))
