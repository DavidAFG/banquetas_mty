"""Prepare the evidence frames for a hosted copy of the map.

The frames are what make this an audit rather than a model output: every claim
on the map can be checked against the pixels it came from. They have to travel
with the site, so they get re-encoded to a size that a repository can carry.

WebP at this quality is roughly a third the weight of the JPEGs, which takes
the set from about 175 MB to something a git push finishes in one sitting.
"""
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from banquetas.config import load_city

QUALITY = 62
WIDTH = 640


def main(city_name="monterrey", quality=QUALITY):
    city = load_city(city_name)
    src = city.out_dir / "map" / "evidence"
    if not src.exists():
        print(f"{src} not found. Run build_map.py first."); return 1
    dst = Path(__file__).resolve().parents[1] / "site" / "evidence"
    dst.mkdir(parents=True, exist_ok=True)

    jpgs = sorted(src.glob("*.jpg"))
    made = skipped = 0
    for n, p in enumerate(jpgs, 1):
        out = dst / (p.stem + ".webp")
        if out.exists():
            skipped += 1
            continue
        im = Image.open(p).convert("RGB")
        if im.width != WIDTH:
            im = im.resize((WIDTH, round(im.height * WIDTH / im.width)))
        im.save(out, "WEBP", quality=int(quality), method=5)
        made += 1
        if n % 500 == 0:
            print(f"    {n}/{len(jpgs)}", flush=True)

    total = sum(f.stat().st_size for f in dst.glob("*.webp"))
    print(f"  {made:,} written, {skipped:,} already there")
    print(f"  site/evidence: {len(list(dst.glob('*.webp'))):,} frames, {total/1e6:.0f} MB")
    print(f"  was {sum(f.stat().st_size for f in jpgs)/1e6:.0f} MB as JPEG")
    print("\n  Commit site/ to a repository and turn on Pages. If the push feels heavy,")
    print("  rerun with a lower quality, for example:  python scripts/build_site_evidence.py monterrey 50")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
