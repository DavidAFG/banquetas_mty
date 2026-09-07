"""Round trip test of the camera geometry.

Builds a synthetic scene with a sidewalk of known width at a known offset,
projects it into a fisheye image using the forward model, rasterises it into a
label map, then runs the measurement code and checks the metres come back.

This is the only part of the pipeline where an error is invisible: a wrong
camera model produces plausible looking numbers that are simply wrong.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from banquetas import geometry as G

W, H = 1024, 576
FOCAL, K1, K2 = 0.4696, -0.05159, 0.00287     # a real Kaart frame's intrinsics
CAM_H = 1.3

SIDEWALK_ID, ROAD_ID, SKY_ID = 15, 13, 27
GROUPS = {"sidewalk": [SIDEWALK_ID], "road": [ROAD_ID], "curb": [],
          "pole": [], "sign": [], "furniture": [], "car": []}

TRUE_WIDTH, TRUE_OFFSET = 2.0, 3.0            # metres
ROAD_LEFT, ROAD_RIGHT = -6.0, 2.8


def project_fisheye(X, Y, Z):
    l = np.hypot(X, Y)
    l = np.where(l < 1e-9, 1e-9, l)
    theta = np.arctan2(l, Z)
    th2 = theta * theta
    theta_d = theta * (1 + K1 * th2 + K2 * th2 * th2)
    S = max(W, H)
    u = FOCAL * theta_d * X / l * S + W / 2 - 0.5
    v = FOCAL * theta_d * Y / l * S + H / 2 - 0.5
    return u, v


def paint(seg, x0, x1, label):
    zs = np.arange(2.0, 40.0, 0.02)
    xs = np.arange(x0, x1, 0.02)
    XX, ZZ = np.meshgrid(xs, zs)
    u, v = project_fisheye(XX.ravel(), np.full(XX.size, CAM_H), ZZ.ravel())
    u, v = np.round(u).astype(int), np.round(v).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    seg[v[ok], u[ok]] = label


def main():
    seg = np.full((H, W), SKY_ID, dtype=np.uint8)
    paint(seg, ROAD_LEFT, ROAD_RIGHT, ROAD_ID)
    paint(seg, TRUE_OFFSET, TRUE_OFFSET + TRUE_WIDTH, SIDEWALK_ID)

    res = G.measure_image(seg, GROUPS, ("fisheye", FOCAL, K1, K2), CAM_H, 0.0, "right")
    print(f"  bands used         {res['n_bands']}")
    print(f"  width      {res['width_m']:.2f} m   true {TRUE_WIDTH:.2f}")
    print(f"  kerb offset{res['kerb_offset_m']:8.2f} m   true {TRUE_OFFSET:.2f}")
    print(f"  road width {res['road_width_m']:.2f} m   true {ROAD_RIGHT - ROAD_LEFT:.2f}")

    ok = True
    def check(label, got, want, tol):
        nonlocal ok
        good = abs(got - want) <= tol
        print(f"  {'PASS' if good else 'FAIL'}  {label}: {got:.2f} vs {want:.2f} (tol {tol})")
        ok = ok and good

    print("\nchecks")
    check("width", res["width_m"], TRUE_WIDTH, 0.25)
    check("kerb offset", res["kerb_offset_m"], TRUE_OFFSET, 0.25)
    check("road width", res["road_width_m"], ROAD_RIGHT - ROAD_LEFT, 0.5)

    # a 10 percent error in assumed camera height must scale widths by 10 percent
    res2 = G.measure_image(seg, GROUPS, ("fisheye", FOCAL, K1, K2), CAM_H * 1.1, 0.0, "right")
    ratio = res2["width_m"] / res["width_m"]
    print(f"\n  height +10% scales width by {ratio:.3f} (expected 1.100)")
    check("height sensitivity", ratio, 1.10, 0.02)

    print("\nOK" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
