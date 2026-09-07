"""Stage 0 figures. Run after `python -m banquetas zones --city monterrey`.

  1  coverage_map.png       observed vs unobserved network over the metro
  2  coverage_gradient.png  coverage by socioeconomic decile, split by road tier
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from banquetas.config import load_city

OBSERVED = "#1b6ca8"
UNOBSERVED = "#d9534f"


def main(city_name="monterrey"):
    city = load_city(city_name)
    out = city.out_dir
    path = out / "chunks_urban.gpkg"
    if not path.exists():
        print("chunks_urban.gpkg not found, run the zones command first"); return 1

    ch = gpd.read_file(path, layer="chunks")
    print(f"loaded {len(ch):,} urban chunks")

    fig, ax = plt.subplots(figsize=(11, 11))
    ch[~ch["observed"].astype(bool)].plot(ax=ax, color=UNOBSERVED, linewidth=0.15)
    ch[ch["observed"].astype(bool)].plot(ax=ax, color=OBSERVED, linewidth=0.25)
    ax.set_axis_off()
    ax.set_title("What Monterrey can be seen with\nblue: street level imagery exists   red: none",
                 loc="left", fontsize=13)
    fig.savefig(out / "coverage_map.png", dpi=200, bbox_inches="tight")
    print(f"wrote {out / 'coverage_map.png'}")

    dt = pd.read_csv(out / "coverage_by_ses_decile_tier.csv")
    dt = dt[dt["ses_decile"].notna()]
    fig, ax = plt.subplots(figsize=(9, 5))
    for tier, sub in dt.groupby("tier"):
        sub = sub.sort_values("ses_decile")
        ax.plot(sub["ses_decile"], sub["coverage_pct"], marker="o", label=tier)
    ax.set_xlabel("AGEB socioeconomic decile, D1 poorest")
    ax.set_ylabel("share of street length observed, %")
    ax.set_title("Does imagery coverage track income?", loc="left")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "coverage_gradient.png", dpi=200, bbox_inches="tight")
    print(f"wrote {out / 'coverage_gradient.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
