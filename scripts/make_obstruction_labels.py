"""A focused second pass on obstruction only.

Three measures of obstruction all scored at chance against the first labels.
Either obstruction needs depth, or those labels were noise: they were applied
with a keystroke while attention was on presence, against no written rule.

This separates the two. It reserves the frames labelled in the first pass and
mixes them, unmarked, with an equal number of fresh ones. So the same run
yields both a reliability check, how often you agree with yourself, and more
power for the model comparison.

The rule is deliberately behavioural rather than visual:

    BLOCKED means a pedestrian would have to step off the walking surface,
    onto the road or the dirt, to get past.

Not "there is a pole in shot". A pole at the kerb that leaves a metre of clear
walk is passable. That distinction is the entire experiment.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from banquetas.config import load_city

N_FRESH = 130

PAGE = """<!doctype html><meta charset=utf-8><title>banquetas · obstruction</title>
<style>
 body{font:14px system-ui;margin:0;background:#111;color:#eee}
 header{position:fixed;top:0;left:0;right:0;background:#000;padding:9px 14px;
        border-bottom:1px solid #333;z-index:2}
 .rule{background:#1d1d1d;border-left:3px solid #d03b3b;padding:7px 11px;
       margin:7px 0 0;font-size:13px;max-width:1100px}
 .keys{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-top:8px}
 .k{background:#222;border:1px solid #444;border-radius:4px;padding:1px 7px;
    font-family:ui-monospace}
 #img{margin-top:150px;width:100vw;object-fit:contain;
      max-height:calc(100vh - 200px);display:block}
 footer{position:fixed;bottom:0;left:0;right:0;background:#000;padding:8px 14px;
        border-top:1px solid #333;display:flex;gap:20px;font-size:13px}
 b.on{color:#4ade80} b.off{color:#666}
 button{background:#222;color:#eee;border:1px solid #555;padding:5px 12px;
        border-radius:4px;cursor:pointer}
</style>
<header>
 <div style="display:flex;gap:18px;align-items:center">
   <span id=pos></span><span id=done></span>
   <button onclick=save()>download CSV</button>
 </div>
 <div class=rule><b>Blocked</b> means a pedestrian would have to step off the
   walking surface, onto the road or the dirt, to get past. A pole at the kerb
   that leaves the walk usable is <b>passable</b>, not blocked.</div>
 <div class=keys>
   <span><span class=k>b</span> blocked</span>
   <span><span class=k>p</span> passable</span>
   <span><span class=k>t</span> tight, passable single file</span>
   <span><span class=k>x</span> cannot tell</span>
   <span style="color:#888">what is in the way:
     <span class=k>1</span>pole <span class=k>2</span>vehicle
     <span class=k>3</span>plants <span class=k>4</span>vendor or goods
     <span class=k>5</span>works <span class=k>6</span>other
     &rarr; <b id=cause class=off>none</b></span>
   <span><span class=k>&larr;</span><span class=k>&rarr;</span> move</span>
 </div>
</header>
<img id=img>
<footer><span id=cur></span></footer>
<script>
const ROWS = __ROWS__;
const KEY = "banquetas_obstruction";
let i = 0, L = {};
try { L = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { L = {}; }
const CAUSES = {1:"pole",2:"vehicle",3:"plants",4:"vendor",5:"works",6:"other"};

function show(){
  const r = ROWS[i], l = L[r.id] || {};
  document.getElementById("img").src = "__IMGDIR__/" + r.id + ".jpg";
  document.getElementById("pos").textContent = (i+1) + " / " + ROWS.length;
  document.getElementById("done").textContent = Object.keys(L).length + " labelled";
  document.getElementById("cause").textContent = l.cause || "none";
  document.getElementById("cause").className = l.cause ? "on" : "off";
  document.getElementById("cur").textContent = "current label: " + (l.verdict || "none");
}
function put(k,v){ const r=ROWS[i]; L[r.id]=Object.assign(L[r.id]||{},{[k]:v});
  try{localStorage.setItem(KEY,JSON.stringify(L));}catch(e){} }
document.addEventListener("keydown", e => {
  const m = {b:"blocked", p:"passable", t:"tight", x:"unknown"};
  if (m[e.key]){ put("verdict", m[e.key]); if(i<ROWS.length-1) i++; show(); }
  else if (CAUSES[e.key]){ put("cause", CAUSES[e.key]); show(); }
  else if (e.key === "ArrowRight" && i < ROWS.length-1){ i++; show(); }
  else if (e.key === "ArrowLeft" && i > 0){ i--; show(); }
});
function save(){
  let out = "id,verdict,cause\\n";
  for (const r of ROWS){ const l = L[r.id]; if(!l || !l.verdict) continue;
    out += [r.id, l.verdict, l.cause || ""].join(",") + "\\n"; }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([out], {type:"text/csv"}));
  a.download = "obstruction_labels.csv"; a.click();
}
show();
</script>
"""


def main(city_name="monterrey", n_fresh=N_FRESH, seed=5):
    city = load_city(city_name)
    rng = np.random.default_rng(seed)

    meas = pd.read_csv(city.out_dir / "image_measurements.csv", dtype={"id": str})
    meas = meas[meas["present"].astype(str).str.lower().isin(["true", "1"])]

    old_path = city.out_dir / "labeling" / "labels.csv"
    repeats = []
    if old_path.exists():
        old = pd.read_csv(old_path, dtype={"id": str})
        repeats = old.loc[old["presence"] == "present", "id"].tolist()
        repeats = [i for i in repeats if i in set(meas["id"])]

    pool = meas.loc[~meas["id"].isin(repeats), "id"].tolist()
    fresh = list(rng.choice(pool, size=min(int(n_fresh), len(pool)), replace=False))

    ids = repeats + fresh
    rng.shuffle(ids)                       # unmarked, so the repeats are blind
    rows = [{"id": str(i)} for i in ids]

    out = city.out_dir / "labeling_obstruction"
    out.mkdir(parents=True, exist_ok=True)
    rel = Path("../../../raw") / city.name / "images"
    (out / "index.html").write_text(
        PAGE.replace("__ROWS__", json.dumps(rows)).replace("__IMGDIR__", str(rel)))
    pd.DataFrame({"id": ids, "repeat": [i in set(repeats) for i in ids]}).to_csv(
        out / "sample.csv", index=False)

    print(f"  {len(ids)} frames: {len(repeats)} repeated from the first pass "
          f"(blind, mixed in) and {len(fresh)} fresh")
    print(f"  open {out / 'index.html'}")
    print(f"  save the export as {out / 'obstruction_labels.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
