"""Score the pipeline against hand labels.

Reads data/out/<city>/labeling/labels.csv as exported by the labelling page
and compares it with the per image measurements.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from banquetas.config import load_city


def main(city_name="monterrey"):
    city = load_city(city_name)
    lab_path = city.out_dir / "labeling" / "labels.csv"
    if not lab_path.exists():
        print(f"{lab_path} not found. Label first, then save the CSV there.")
        return 1
    lab = pd.read_csv(lab_path, dtype={"id": str})
    pred = pd.read_csv(city.out_dir / "image_measurements.csv", dtype={"id": str})
    df = lab.merge(pred[["id", "present", "width_m", "obstructed_frac", "n_sidewalk_px"]],
                   on="id", how="inner")
    print(f"  {len(df)} labelled images matched to measurements")

    usable = df[df["presence"].isin(["absent", "present"])].copy()
    usable["truth"] = usable["presence"] == "present"
    tp = int(((usable["truth"]) & (usable["present"])).sum())
    fn = int(((usable["truth"]) & (~usable["present"])).sum())
    fp = int(((~usable["truth"]) & (usable["present"])).sum())
    tn = int(((~usable["truth"]) & (~usable["present"])).sum())
    n = tp + fn + fp + tn
    print("\n  presence, excluding occluded and not applicable")
    print(f"                 predicted present   predicted absent")
    print(f"    truly present {tp:>12,}      {fn:>15,}")
    print(f"    truly absent  {fp:>12,}      {tn:>15,}")
    if n:
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        print(f"\n    accuracy  {(tp + tn) / n:.3f}")
        print(f"    precision {prec:.3f}   recall {rec:.3f}   "
              f"F1 {2 * prec * rec / max(prec + rec, 1e-9):.3f}")

    print("\n  label distribution")
    print(df["presence"].value_counts().to_string())

    if "cycleway" in df.columns and df["cycleway"].notna().any():
        cyc = df[df["cycleway"] == 1]
        print(f"\n  cycleway on the near side in {len(cyc)} frames; of those, "
              f"{(cyc['presence'] == 'absent').sum()} have no sidewalk")

    na = (df["presence"] == "na").mean()
    print(f"\n  no right roadside in {100 * na:.1f}% of frames. If this is high the "
          f"near side assumption\n  is failing more often than expected and the "
          f"effective sample is smaller than the draw.")

    occ = (df["presence"] == "occluded").mean()
    print(f"\n  occluded in {100 * occ:.1f}% of frames, the ceiling on what any "
          f"single frame method can see")

    print("\n  accuracy by decile")
    u = usable.assign(correct=lambda d: d["truth"] == d["present"])
    print(u.groupby("decile")["correct"].agg(n="size", accuracy="mean")
          .to_string(float_format=lambda v: f"{v:.3f}"))

    if "blocked" in df.columns and df["blocked"].notna().any():
        b = df[df["presence"] == "present"]
        if len(b) > 5:
            print(f"\n  obstruction: mean predicted obstructed_frac is "
                  f"{b.loc[b['blocked'] == 1, 'obstructed_frac'].mean():.3f} on frames "
                  f"labelled blocked against "
                  f"{b.loc[b['blocked'] == 0, 'obstructed_frac'].mean():.3f} on frames not")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
