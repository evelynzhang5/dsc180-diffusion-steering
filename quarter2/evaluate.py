"""Score generated images against each prompt's checklist (raw scores; analyze.py turns them into pass/fail).

    python evaluate.py --run RUN --prompts prompts.json

For every image in RUN/baseline.jsonl and RUN/branches.jsonl:
- Objects: OWLv2 boxes for "a photo of a <object>". The two objects get the highest-scoring pair of boxes
  that overlap little (IoU < 0.5), so one box cannot count as both objects.
- Colors: CLIP ViT-L/14 picks among 11 colors for each object's crop ("a photo of a <color> <object>").
- Relation: the horizontal centers of the two boxes.
Writes RUN/scores.csv (one row per image; already-scored images are skipped). Manual audit labels go in a
separate file, so the automated scores here can be checked against them.
"""
import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image

DETECTOR = "google/owlv2-base-patch16-ensemble"
CLIP = "openai/clip-vit-large-patch14"
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray"]
FIELDS = ["key", "det_1", "det_2", "box_1", "box_2", "cx_1", "cx_2", "color_pred_1", "color_pred_2",
          "color_prob_1", "color_prob_2"]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Scorer:
    def __init__(self, device):
        from transformers import CLIPModel, CLIPProcessor, Owlv2ForObjectDetection, Owlv2Processor
        self.device = device
        self.det_proc = Owlv2Processor.from_pretrained(DETECTOR)
        self.det = Owlv2ForObjectDetection.from_pretrained(DETECTOR).to(device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(CLIP)
        self.clip = CLIPModel.from_pretrained(CLIP).to(device).eval()

    @torch.no_grad()
    def boxes(self, img, objects):
        inputs = self.det_proc(text=[[f"a photo of a {o}" for o in objects]], images=img,
                               return_tensors="pt").to(self.device)
        out = self.det(**inputs)
        side = max(img.size)  # OWLv2 pads to a square, so boxes are in padded-square coordinates
        res = self.det_proc.post_process_object_detection(out, threshold=0.0,
                                                          target_sizes=torch.tensor([[side, side]]))[0]
        top = []
        for label in range(len(objects)):
            m = res["labels"] == label
            s, b = res["scores"][m], res["boxes"][m]
            idx = s.argsort(descending=True)[:10]
            top.append([(float(s[i]), [float(v) for v in b[i]]) for i in idx])
        best = max(((s1 + s2, (s1, b1), (s2, b2)) for s1, b1 in top[0] for s2, b2 in top[1]
                    if iou(b1, b2) < 0.5), default=None)
        if best is None:  # every candidate pair overlaps: keep each object's own best box
            return top[0][0], top[1][0]
        return best[1], best[2]

    @torch.no_grad()
    def colors(self, img, box, obj):
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(img.width, max(x1, x0 + 8)), min(img.height, max(y1, y0 + 8))
        crop = img.crop((x0, y0, x1, y1))
        if min(crop.size) < 32:  # tiny crops make the processor misread the channel axis
            f = 32 / min(crop.size)
            crop = crop.resize((round(crop.width * f), round(crop.height * f)))
        inputs = self.clip_proc(text=[f"a photo of a {c} {obj}" for c in COLORS], images=crop,
                                return_tensors="pt", padding=True).to(self.device)
        return self.clip(**inputs).logits_per_image.softmax(-1)[0].cpu()

    def score(self, img, p):
        (d1, b1), (d2, b2) = self.boxes(img, [p["object_1"], p["object_2"]])
        c1, c2 = self.colors(img, b1, p["object_1"]), self.colors(img, b2, p["object_2"])
        return {"det_1": round(d1, 4), "det_2": round(d2, 4),
                "box_1": " ".join(f"{v:.0f}" for v in b1), "box_2": " ".join(f"{v:.0f}" for v in b2),
                "cx_1": round((b1[0] + b1[2]) / 2, 1), "cx_2": round((b2[0] + b2[2]) / 2, 1),
                "color_pred_1": COLORS[int(c1.argmax())], "color_pred_2": COLORS[int(c2.argmax())],
                "color_prob_1": round(float(c1[COLORS.index(p["color_1"])]), 4),
                "color_prob_2": round(float(c2[COLORS.index(p["color_2"])]), 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    run = Path(args.run)
    prompts = {p["id"]: p for p in json.load(open(args.prompts))}
    rows = []
    for name in ("baseline.jsonl", "branches.jsonl"):
        if (run / name).exists():
            rows += [json.loads(l) for l in open(run / name)]
    out = run / "scores.csv"
    done = {r["key"] for r in csv.DictReader(open(out))} if out.exists() else set()
    todo = [r for r in rows if r["key"] not in done]
    print(f"scoring {len(todo)} images ({len(done)} already scored)", flush=True)
    if not todo:
        return
    scorer = Scorer(args.device)
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, r in enumerate(todo):
            img = Image.open(run / r["image"]).convert("RGB")
            w.writerow({"key": r["key"], **scorer.score(img, prompts[r["prompt_id"]])})
            if i % 100 == 99:
                f.flush()
                print(f"  {i + 1}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
