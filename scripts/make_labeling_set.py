"""Draw a balanced validation set and build a local labelling page.

The pilot sample is proportional, so two thirds of it sits in the upper
deciles. Measuring accuracy on that would flatter the model exactly where the
streets are widest and best built. This draws an equal number of images per
socioeconomic decile from what has already been downloaded, so no refetching
is needed, and writes a self contained HTML page for labelling them by hand.

Nothing is uploaded. The page reads the images off your disk and exports a
CSV you keep.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from banquetas.config import load_city

PER_DECILE = 25

PAGE = """<!doctype html><meta charset=utf-8><title>banquetas labelling</title>
<style>
 body{font:14px system-ui;margin:0;background:#111;color:#eee}
 header{position:fixed;top:0;left:0;right:0;background:#000;padding:8px 14px;
        display:flex;gap:18px;align-items:center;border-bottom:1px solid #333}
 #img{margin-top:52px;width:100vw;object-fit:contain;max-height:calc(100vh - 110px);display:block}
 .k{background:#222;border:1px solid #444;border-radius:4px;padding:1px 6px;font-family:ui-monospace}
 footer{position:fixed;bottom:0;left:0;right:0;background:#000;padding:8px 14px;
        border-top:1px solid #333;display:flex;gap:18px}
 b.on{color:#4ade80} b.off{color:#666}
 button{background:#222;color:#eee;border:1px solid #555;padding:5px 12px;border-radius:4px;cursor:pointer}
</style>
<header>
 <span id=pos></span>
 <span>near side (right of travel):
  <span class=k>a</span> absent
  <span class=k>p</span> present
  <span class=k>o</span> occluded
  <span class=k>n</span> not applicable</span>
 <span><span class=k>b</span> blocked: <b id=blk class=off>no</b></span>
 <span><span class=k>&larr;</span> <span class=k>&rarr;</span> move</span>
 <button onclick=save()>download CSV</button>
</header>
<img id=img>
<footer><span id=cur></span><span id=done></span></footer>
<script>
const ROWS = __ROWS__;
const KEY = "banquetas_labels";
let i = 0, L = {};
try { L = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { L = {}; }

function show(){
  const r = ROWS[i];
  document.getElementById("img").src = "__IMGDIR__/" + r.id + ".jpg";
  document.getElementById("pos").textContent = (i+1) + " / " + ROWS.length;
  const l = L[r.id] || {};
  document.getElementById("blk").textContent = l.blocked ? "yes" : "no";
  document.getElementById("blk").className = l.blocked ? "on" : "off";
  document.getElementById("cyc").textContent = l.cycleway ? "yes" : "no";
  document.getElementById("cyc").className = l.cycleway ? "on" : "off";
  document.getElementById("flt").textContent = FILTERS[fi];
  document.getElementById("cur").textContent =
    "decile " + r.decile + " | " + r.mun + " | " + r.tier + " | label: " + (l.presence || "none");
  document.getElementById("done").textContent = Object.keys(L).length + " labelled";
}
const FILTERS = ["all", "unlabelled", "no right roadside", "occluded"];
let fi = 0;
function matches(idx){
  const l = L[ROWS[idx].id] || {};
  if (fi === 0) return true;
  if (fi === 1) return !l.presence;
  if (fi === 2) return l.presence === "na";
  return l.presence === "occluded";
}
function step(dir){
  let j = i;
  for (let n = 0; n < ROWS.length; n++){
    j += dir;
    if (j < 0 || j >= ROWS.length) return;
    if (matches(j)){ i = j; show(); return; }
  }
}
function set(p){ const r=ROWS[i]; L[r.id]=Object.assign(L[r.id]||{},{presence:p});
  try{localStorage.setItem(KEY,JSON.stringify(L));}catch(e){}
  step(1); show(); }
function flag(k){ const r=ROWS[i]; const l=L[r.id]||{}; l[k]=!l[k]; L[r.id]=l;
  try{localStorage.setItem(KEY,JSON.stringify(L));}catch(e){} show(); }
document.addEventListener("keydown", e => {
  const m = {a:"absent", p:"present", o:"occluded", n:"na"};
  if (m[e.key]) set(m[e.key]);
  else if (e.key === "b") flag("blocked");
  else if (e.key === "c") flag("cycleway");
  else if (e.key === "f"){ fi = (fi + 1) % FILTERS.length; show(); }
  else if (e.key === "ArrowRight") step(1);
  else if (e.key === "ArrowLeft") step(-1);
});
function save(){
  let out = "id,chunk_id,decile,mun,tier,presence,blocked,cycleway\\n";
  for (const r of ROWS){ const l = L[r.id]; if(!l||!l.presence) continue;
    out += [r.id,r.chunk_id,r.decile,JSON.stringify(r.mun),r.tier,l.presence,
            l.blocked?1:0,l.cycleway?1:0].join(",") + "\\n"; }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([out], {type:"text/csv"}));
  a.download = "labels.csv"; a.click();
}
show();
</script>
"""


def main(city_name="monterrey", per_decile=PER_DECILE, seed=1):
    city = load_city(city_name)
    rng = np.random.default_rng(seed)

    meta = pd.read_csv(city.out_dir / "image_metadata.csv")
    meta = meta[meta["path"].notna()][["id", "chunk_id"]]
    chunks = gpd.read_file(city.out_dir / "sample_chunks.gpkg", layer="sample")
    ctx = pd.DataFrame(chunks.drop(columns="geometry"))[
        ["chunk_id", "ses_decile", "mun_name", "tier"]]
    df = meta.merge(ctx, on="chunk_id", how="left")
    df = df[df["ses_decile"].notna()]

    picks = []
    for dec, g in df.groupby("ses_decile", observed=True):
        take = min(per_decile, len(g))
        picks.append(g.iloc[rng.choice(len(g), size=take, replace=False)])
    sample = pd.concat(picks).sample(frac=1, random_state=seed)   # shuffle
    print(f"  validation set: {len(sample)} images, "
          f"{sample['ses_decile'].nunique()} deciles, "
          f"{sample.groupby('ses_decile', observed=True).size().min()} to "
          f"{sample.groupby('ses_decile', observed=True).size().max()} per decile")

    rows = [{"id": str(r.id), "chunk_id": int(r.chunk_id), "decile": str(r.ses_decile),
             "mun": str(r.mun_name), "tier": str(r.tier)} for r in sample.itertuples(index=False)]

    out_dir = city.out_dir / "labeling"
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    rel = Path("../../../raw") / city.name / "images"
    html = PAGE.replace("__ROWS__", json.dumps(rows)).replace("__IMGDIR__", str(rel))
    (out_dir / "index.html").write_text(html)
    sample.to_csv(out_dir / "validation_sample.csv", index=False)
    print(f"  open  {out_dir / 'index.html'}  in your browser")
    print("  label with a / p / o / n, b toggles blocked, then download the CSV")
    print(f"  save it as {out_dir / 'labels.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
