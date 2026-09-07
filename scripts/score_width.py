"""Regress estimated sidewalk width on hand measured width."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from banquetas.config import load_city


def main(city_name="monterrey"):
    city = load_city(city_name)
    p = city.out_dir / "width_check" / "width_check.csv"
    df = pd.read_csv(p)
    df = df[pd.to_numeric(df["measured_width_m"], errors="coerce").notna()]
    if len(df) < 8:
        print(f"only {len(df)} rows filled in, need at least 8"); return 1
    x = df["measured_width_m"].astype(float).to_numpy()
    y = df["estimated_width_m"].astype(float).to_numpy()

    err = y - x
    print(f"  n = {len(df)}")
    print(f"  mean absolute error {np.abs(err).mean():.2f} m")
    print(f"  bias (estimate minus truth) {err.mean():+.2f} m")
    print(f"  correlation {np.corrcoef(x, y)[0,1]:.3f}")
    slope = float(np.sum(x * y) / np.sum(x * x))          # through the origin
    print(f"  slope through origin {slope:.3f}")
    over = int((err > 0.5).sum()); under = int((err < -0.5).sum())
    print(f"  errors over +0.5 m: {over}   under -0.5 m: {under}")
    if over and under:
        print("    NOTE: both directions are present, so the mean bias and the slope are")
        print("    two opposing error modes cancelling, not a scale error. Do NOT rescale")
        print("    the camera height from this regression; use the lane marking check,")
        print("    which measures one thing and has no mixture in it.")
    within = np.mean(np.abs(err) <= 0.5)
    print(f"  within 0.5 m: {100*within:.0f}%   (UrbanVGGT reports 95.5% on Street View)")

    thr = 1.2
    tp = int(((x < thr) & (y < thr)).sum()); tn = int(((x >= thr) & (y >= thr)).sum())
    fp = int(((x >= thr) & (y < thr)).sum()); fn = int(((x < thr) & (y >= thr)).sum())
    acc = (tp + tn) / len(df)
    n_narrow = int((x < thr).sum())
    baseline = max(n_narrow, len(df) - n_narrow) / len(df)
    print(f"\n  narrow versus adequate at {thr} m")
    print(f"    truly narrow {n_narrow} of {len(df)}")
    print(f"    accuracy {acc:.3f}   ALWAYS SAYING THE MAJORITY CLASS SCORES {baseline:.3f}")
    if acc <= baseline:
        print("    the classifier does not beat the trivial baseline on this sample.")
        print("    Either it is not usable per face, or the sample has too few narrow")
        print("    cases to tell. Draw a validation set that oversamples low estimates")
        print("    before drawing any conclusion.")
    print(f"    precision {tp / max(tp + fp, 1):.3f}  recall {tp / max(tp + fn, 1):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
