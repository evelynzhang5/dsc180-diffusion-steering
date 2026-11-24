"""Round 40, POST HOC (written after the stage 3 results; not part of the frozen protocol).
Does BokehDiff's delivered blur follow K x (depth gap between the behind tiles and the focus)?
Fit r = a*K*gap on the development responses (K >= 12), then predict the held-out test outputs.
usage (repo root): python quarter2/compare40/post40_gap.py
"""
import json, sys, numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view as sw
def _f(d,fn):
    return fn(sw(np.pad(d,1,mode='edge'),(3,3)),axis=(-2,-1))
grey_erosion=lambda d,size: _f(d,np.min)
grey_dilation=lambda d,size: _f(d,np.max)
sys.path.insert(0,'quarter2/compare40'); import common40 as C
D='outputs/q2_compare40/data'; scenes,regions,review=C.load(D)
def gap(i):
    d=np.load(f'{D}/depth/{i}_pred.npy').astype(np.float32); d=(d-d.min())/(d.max()-d.min())
    dm=grey_dilation(grey_erosion(d,size=(3,3)),size=(3,3)); d=np.clip((d-dm.min())/(dm.max()-dm.min()),0,1)
    m=np.array(Image.open(f'{D}/mask/{i}.png').resize((d.shape[1],d.shape[0]),resample=Image.LANCZOS).convert('L'))/255.
    f=np.median(d[m>0.5]) if m.max()>0 else np.nan
    H,W=regions[i]['size']; d=np.asarray(Image.fromarray(d.astype(np.float32)).resize((W,H),resample=Image.BILINEAR))
    return float(np.median([np.median(np.abs(d[t['y']:t['y']+64,t['x']:t['x']+64]-f)) for t in regions[i]['tiles']]))
mp=json.load(open('quarter2/compare40/results/stage3/mapping.json'))
X=[];Y=[]
for i,rs in mp['bokehdiff']['responses'].items():
    g=gap(i)
    for K,v in rs.items():
        if np.isfinite(v['r']) and float(K)>=12: X.append(float(K)*g); Y.append(v['r'])
    print('dev',i,round(g,3))
X,Y=np.array(X),np.array(Y); a=(X@Y)/(X@X); print('dev fit r = a*K*gap, a=%.3f  R2=%.2f  corr=%.2f n=%d'%(a,1-((Y-a*X)**2).sum()/((Y-Y.mean())**2).sum(),np.corrcoef(X,Y)[0,1],len(X)))
s=json.load(open('quarter2/compare40/results/stage3/score40.json'))
xs=[];ys=[];zs=[];bk=[]
for r in s['rows']:
    if r['category']=='existing_blur' or not np.isfinite(r.get('blur',np.nan)) or r['flag']: continue
    if r['method']=='bokehdiff':
        g=gap(r['id']); xs.append(a*r['setting']*g); ys.append(r['blur']); zs.append(r['target'])
xs,ys,zs=np.array(xs),np.array(ys),np.array(zs)
print('test: predicted vs delivered corr %.2f; |pred-deliv| median %.2f; within TOL of delivered %d/%d'%(np.corrcoef(xs,ys)[0,1],np.median(abs(xs-ys)),sum(abs(xs-ys)<=np.maximum(2,.2*ys)),len(xs)))
print('delivered/target spread per target:',{t:(round(np.min(ys[zs==t]),1),round(np.median(ys[zs==t]),1),round(np.max(ys[zs==t]),1)) for t in [4,8,12]})
# if K were set per scene as target/(a*gap): hypothetical, report predicted residual on test via the linear fit only
