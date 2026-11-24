"""Round 40 stage 3 (GPU): BokehDiff's preparation latency (Depth Anything V2 Base + BiRefNet, as prepare_data.py),
per image, models already loaded and warmed up, synchronised. Added to BokehDiff's rendering time for its total.
usage: python lat40.py <data root> <out root>   -> <out>/lat40.json
"""
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation, pipeline

DATA, OUT = sys.argv[1], sys.argv[2]
ids = sorted(f for f in os.listdir(f"{DATA}/input") if f.endswith(".png"))[:11]
dpipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf", device=0)
bir = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).eval().to("cuda")
tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
dt, st = [], []
with torch.no_grad():
    for f in ids:
        im = Image.open(f"{DATA}/input/{f}").convert("RGB")
        torch.cuda.synchronize()
        t = time.time()
        d = dpipe(im)["predicted_depth"].squeeze().cpu().numpy()
        torch.cuda.synchronize()
        dt.append(time.time() - t)
        t = time.time()
        bir(tf(im).unsqueeze(0).to("cuda"))[-1].sigmoid().cpu()
        torch.cuda.synchronize()
        st.append(time.time() - t)
dt, st = dt[1:], st[1:]                         # first image = warm-up
res = dict(depth_median_s=float(np.median(dt)), seg_median_s=float(np.median(st)),
           prep_median_s=float(np.median(np.array(dt) + np.array(st))), n=len(dt))
json.dump(res, open(f"{OUT}/lat40.json", "w"), indent=1)
print("prep latency", res, flush=True)
