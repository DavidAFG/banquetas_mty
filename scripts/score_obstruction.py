"""Score the focused obstruction labels, and check the labels against themselves.

Two questions, in this order, because the second is only meaningful if the
first has an answer:

1. How consistent are you with yourself? The run mixes in every frame you
   labelled in the first pass. If the two passes disagree badly, the label is
   noise and no model can be scored against it.

2. Does any obstruction measure separate blocked from passable? Three are
   recomputed here straight from the cached masks, so no remeasure is needed.
"""
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from banquetas import geometry as G
from banquetas.config import load_city

MEASURES = ("obstructed_frac", "gap_frac", "touching_frac")


def auc(pos, neg):
    c = d = 0
    for a, b in itertools.product(pos, neg):
        if a > b: c += 1
        elif a < b: d += 1
    n = c + d
    return (c / n if n else 0.5), n


class Row: pass


def main(city_name="monterrey"):
    city = load_city(city_name)
    d = city.out_dir / "labeling_obstruction"
    path = d / "obstruction_labels.csv"
    if not path.exists():
        print(f"{path} not found. Label first, then save the export there.")
        return 1
    new = {r["id"]: r for r in csv.DictReader(open(path))}
    print(f"  {len(new)} frames labelled")
    from collections import Counter
    print("  " + ", ".join(f"{k} {v}" for k, v in
                           Counter(r["verdict"] for r in new.values()).most_common()))
    causes = Counter(r["cause"] for r in new.values() if r["cause"])
    if causes:
        print("  what was in the way: " + ", ".join(f"{k} {v}" for k, v in causes.most_common()))

    # 1. agreement with the first pass
    old_path = city.out_dir / "labeling" / "labels.csv"
    if old_path.exists():
        old = {r["id"]: r for r in csv.DictReader(open(old_path))}
        both = [(old[i], new[i]) for i in new if i in old
                and old[i].get("blocked") not in (None, "")
                and new[i]["verdict"] in ("blocked", "passable", "tight")]
        if both:
            agree = sum(1 for o, n in both
                        if (o["blocked"] == "1") == (n["verdict"] == "blocked"))
            p = agree / len(both)
            # chance agreement, for kappa
            a1 = sum(1 for o, _ in both if o["blocked"] == "1") / len(both)
            a2 = sum(1 for _, n in both if n["verdict"] == "blocked") / len(both)
            pe = a1 * a2 + (1 - a1) * (1 - a2)
            kappa = (p - pe) / (1 - pe) if pe < 1 else 0.0
            print(f"\n  SELF AGREEMENT on {len(both)} repeated frames: {100*p:.0f}%"
                  f"   Cohen's kappa {kappa:.2f}")
            print("    kappa under about 0.4 means the label itself is unreliable,")
            print("    and no model can be scored against it.")

    # 2. the measures
    groups = {k: list(map(int, v)) for k, v in
              json.loads((city.interim_dir / "label_map.json").read_text())["groups"].items()}
    meta = {r["id"]: r for r in csv.DictReader(open(city.out_dir / "image_metadata.csv"))}
    mask_dir = city.interim_dir / "masks"

    scores = {m: ([], []) for m in MEASURES}
    for i, r in new.items():
        if r["verdict"] not in ("blocked", "passable"):
            continue
        mp = mask_dir / f"{i}.png"
        if not mp.exists() or i not in meta:
            continue
        o = Row()
        o.camera_type = meta[i].get("camera_type")
        o.camera_parameters = meta[i].get("camera_parameters")
        res = G.measure_image(np.array(Image.open(mp)), groups, G.parse_camera(o),
                              city.camera_height_m, city.camera_pitch_deg, "right")
        for m in MEASURES:
            v = res.get(m)
            if v is None or v != v:
                continue
            scores[m][0 if r["verdict"] == "blocked" else 1].append(v)

    print(f"\n  {'measure':22s} {'AUC':>6} {'blocked':>9} {'passable':>9}")
    for m, (pos, neg) in scores.items():
        if not pos or not neg:
            continue
        a, _ = auc(pos, neg)
        print(f"  {m:22s} {a:6.3f} {np.mean(pos):9.3f} {np.mean(neg):9.3f}"
              f"   n={len(pos)}/{len(neg)}")
    n = min(len(v[0]) for v in scores.values() if v[0]) if any(v[0] for v in scores.values()) else 0
    if n:
        se = 0.5 / max(np.sqrt(n), 1)
        print(f"\n  rough standard error on AUC at this sample size: {se:.3f}")
        print(f"  anything under about {0.5 + 2*se:.2f} is not distinguishable from chance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
