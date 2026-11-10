"""Blinded hand audit of the automated scorer (pilots 2-3).

    python quarter2/sd_pilots/make_audit.py build --out outputs/q2_audit [--tasks counting] [--random 0] [--pairs 12]
    python quarter2/sd_pilots/make_audit.py score --out outputs/q2_audit --labels ~/Downloads/audit_labels.csv

Two separate sets per task, never pooled:
- random: images drawn at random from the originals, real-nudge branches and placebo branches (equal thirds).
  Gives the scorer's agreement with a person on typical images.
- flips: pairs (original, branch) whose automated verdict changed, half from real nudges and half from placebo.
  Both images of a pair are shown separately and shuffled, so the labeler never sees a pair or its source.
  Gives the share of automated repairs and harms a person confirms, for real vs. placebo.
The page shows only an id, the question and the image. key.csv (which image is which) is for scoring only;
don't open it before labeling.
"""
import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scorer_check import load  # noqa: E402

TASKS = {"two_object": "two_object_prompts.json", "counting": "counting_countable.json"}



def build(out, tasks, n_random, n_pairs):
    rng = random.Random(0)
    (out / "images").mkdir(parents=True, exist_ok=True)
    items = []
    for task in tasks:
        pf = TASKS[task]
        run = ROOT / "outputs" / "q2_pilot2" / task / "main"
        prompts = {p["id"]: p for p in json.load(open(Path(__file__).parent / pf))}
        base, br = load(run, prompts)
        real = [r for r in br if r["action"] != "placebo"]
        plac = [r for r in br if r["action"] == "placebo"]
        pools = {"original": list(base.values()), "real": real, "placebo": plac}
        for src, pool in pools.items():
            for r in rng.sample(pool, n_random // 3):
                items.append(dict(set="random", task=task, source=src, pair="", r=r, p=prompts[r["prompt_id"]]))
        for src, pool in (("real", real), ("placebo", plac)):
            flips = [r for r in pool if (r["m"] >= 0) != (base[r["base_key"]]["m"] >= 0)]
            for j, r in enumerate(rng.sample(flips, min(n_pairs // 2, len(flips)))):
                pair = f"{task}-{src}-{j}"
                kind = "repair" if r["m"] >= 0 else "harm"
                items.append(dict(set="flips", task=task, source=src, pair=pair, kind=kind, side="branch", r=r,
                                  p=prompts[r["prompt_id"]]))
                b = base[r["base_key"]]
                items.append(dict(set="flips", task=task, source="original", pair=pair, kind=kind, side="original",
                                  r=b, p=prompts[b["prompt_id"]]))
    rng.shuffle(items)
    key_rows, cards = [], []
    for i, it in enumerate(items):
        aid = f"a{i:03d}"
        run = ROOT / "outputs" / "q2_pilot2" / it["task"] / "main"
        shutil.copy(run / it["r"]["image"], out / "images" / f"{aid}.jpg")
        p = it["p"]
        if it["task"] == "counting":
            q = f"How many {p['object_1']}s are clearly visible? (number; ? if unsure)"
        else:
            q = f"Is there a {p['object_1']}? Is there a {p['object_2']}? (y/n/? for each)"
        key_rows.append(dict(id=aid, set=it["set"], task=it["task"], source=it["source"], pair=it["pair"],
                             kind=it.get("kind", ""), side=it.get("side", ""), key=it["r"]["key"],
                             auto_margin=round(it["r"]["m"], 4), auto_pass=int(it["r"]["m"] >= 0),
                             count=p.get("count", ""), object_1=p["object_1"], object_2=p.get("object_2", "")))
        cards.append((aid, it["task"], q, p))
    with open(out / "key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key_rows[0]))
        w.writeheader()
        w.writerows(key_rows)
    html(out, cards)
    print(f"{len(items)} images -> {out / 'audit.html'}")


def html(out, cards):
    rows = []
    for aid, task, q, p in cards:
        if task == "counting":
            inp = f'<input name="{aid}_n" size="4" placeholder="count">'
        else:
            inp = "".join(f'<label>{o}: <select name="{aid}_{k}"><option></option><option>y</option><option>n'
                          f'</option><option>?</option></select></label> '
                          for k, o in (("o1", p["object_1"]), ("o2", p["object_2"])))
        rows.append(f'<div class="card"><img src="images/{aid}.jpg" loading="lazy"><div><b>{aid}</b><br>{q}'
                    f'<br>{inp}</div></div>')
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Scorer audit</title><style>
body{{font-family:system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}}
.card{{display:flex;gap:12px;align-items:center;border-bottom:1px solid #ddd;padding:8px 0}}
img{{width:320px;height:320px;object-fit:contain;background:#eee}}
button{{font-size:16px;padding:8px 16px;position:sticky;top:8px}}
</style></head><body>
<h2>Scorer audit ({len(cards)} images)</h2>
<p>Answer from the image only. Count an object if a person would say it is there; use ? when you can't tell.
Your answers are saved in this browser as you go. Press the button to download them as CSV.</p>
<button onclick="save()">Download labels (CSV)</button> <span id="done"></span>
<form id="f">{''.join(rows)}</form>
<script>
const f=document.getElementById('f');
function load(){{try{{const s=JSON.parse(localStorage.getItem('audit_v3')||'{{}}');
 for(const [k,v] of Object.entries(s)){{if(f.elements[k])f.elements[k].value=v;}}}}catch(e){{}}count();}}
function count(){{let n=0;for(const e of f.elements)if(e.value)n++;document.getElementById('done').textContent=n+' answers';}}
f.addEventListener('change',()=>{{const s={{}};for(const e of f.elements)if(e.value)s[e.name]=e.value;
 try{{localStorage.setItem('audit_v3',JSON.stringify(s));}}catch(e){{}}count();}});
function save(){{let t='field,value\\n';for(const e of f.elements)if(e.value)t+=e.name+','+e.value+'\\n';
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([t],{{type:'text/csv'}}));
 a.download='audit_labels.csv';a.click();}}
load();
</script></body></html>"""
    (out / "audit.html").write_text(page)


def score(out, labels_path):
    lab = {r["field"]: r["value"].strip() for r in csv.DictReader(open(labels_path))}
    key = list(csv.DictReader(open(out / "key.csv")))
    def human(k):
        if k["task"] == "counting":
            v = lab.get(f"{k['id']}_n", "")
            return None if not v.isdigit() else int(v) == int(k["count"])
        a, b = lab.get(f"{k['id']}_o1", ""), lab.get(f"{k['id']}_o2", "")
        if a not in ("y", "n") or b not in ("y", "n"):
            return None
        return a == "y" and b == "y"
    for k in key:
        k["human"] = human(k)
    L = ["# Scorer audit results\n"]
    for task in TASKS:
        rs = [k for k in key if k["task"] == task and k["set"] == "random" and k["human"] is not None]
        if rs:
            agree = sum(k["human"] == bool(int(k["auto_pass"])) for k in rs)
            fp = sum(bool(int(k["auto_pass"])) and not k["human"] for k in rs)
            fn = sum(not int(k["auto_pass"]) and k["human"] for k in rs)
            L.append(f"**{task}, random set:** scorer agrees with the person on {agree} of {len(rs)} "
                     f"({agree / len(rs):.0%}); scorer says pass but person says fail {fp}, the reverse {fn}. "
                     f"Person's pass rate {sum(k['human'] for k in rs) / len(rs):.0%}.\n")
        pairs = defaultdict(dict)
        for k in key:
            if k["task"] == task and k["set"] == "flips":
                pairs[k["pair"]][k["side"]] = k
        tab = defaultdict(lambda: [0, 0])
        for pr in pairs.values():
            b, o = pr.get("branch"), pr.get("original")
            if not b or not o or b["human"] is None or o["human"] is None:
                continue
            want = (False, True) if b["kind"] == "repair" else (True, False)
            tab[(b["source"], b["kind"])][0] += (o["human"], b["human"]) == want
            tab[(b["source"], b["kind"])][1] += 1
        if tab:
            L += [f"**{task}, automated flips:**\n", "| source | change | confirmed by the person |", "| --- | --- | --- |"]
            L += [f"| {s} | {kd} | {a} of {n} ({a / n:.0%}) |" for (s, kd), (a, n) in sorted(tab.items())]
            L.append("")
    (out / "audit_results.md").write_text("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "score"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels")
    ap.add_argument("--tasks", default="counting", help="comma-separated: two_object,counting")
    ap.add_argument("--random", type=int, default=0, help="random images per task (split over 3 sources)")
    ap.add_argument("--pairs", type=int, default=12, help="flip pairs per task, half real and half placebo")
    a = ap.parse_args()
    out = Path(a.out)
    build(out, a.tasks.split(","), a.random, a.pairs) if a.cmd == "build" else score(out, a.labels)


if __name__ == "__main__":
    main()
