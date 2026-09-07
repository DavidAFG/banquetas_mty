"""Stage 1c: turn label maps into metres.

Every pixel is unprojected through the camera's actual model, then intersected
with an assumed ground plane. Two parameters govern the whole thing and both
live in the city config, not here: camera height and pitch. See the CAMERA
GEOMETRY block in config/cities/<city>.yml.

Mapillary publishes per image intrinsics, and for this campaign they are real
per image estimates from structure from motion rather than a single nominal
value, so they are used as given. The camera models follow OpenSfM, which is
what Mapillary's reconstruction uses.

  perspective   x_n = x/z,  d = 1 + k1 r^2 + k2 r^4,  pixel = f d x_n * S
  fisheye       theta = atan2(l, z),  theta_d = theta (1 + k1 th^2 + k2 th^4)
                pixel = f theta_d (x/l) * S

with S = max(width, height) and the principal point at the image centre.

The measurement is deliberately one sided. The vehicle drives in the right
lane, so the right kerb is a few metres away and the left is across the
carriageway. Only the near side is measured, and the opposite bearing pass
covers the other block face.
"""
from __future__ import annotations

import json

import numpy as np

# ranges over which a dashcam sees the near kerb usefully
Z_MIN_M, Z_MAX_M, Z_BAND_M = 4.0, 20.0, 2.0
MIN_PIXELS_PER_BAND = 25
# a gap wider than this separates the near footway from anything beyond it:
# a plaza, a driveway apron, or the sidewalk on the far side of the street
CLUSTER_GAP_M = 0.6
# nothing this wide is a footway; it is a forecourt, a plaza or a car park
MAX_PLAUSIBLE_WIDTH_M = 8.0
# lane markings sit one lane apart, and a lane is 3.2 to 3.5 m in Mexican
# urban design regardless of how many there are, which makes their spacing a
# tighter calibration standard than the width of the whole carriageway
LANE_GAP_M = 1.2
# wide enough not to clip the plausible range of camera heights, which
# would otherwise make the calibration partly self confirming
LANE_MIN_M, LANE_MAX_M = 2.0, 6.0
# Greenery is only judged in the near bands. A tree eight metres ahead sits
# above the image columns of every band behind it too, so measuring canopy
# over the whole 4 to 20 m window counts one tree many times. The vehicle
# drives past, so a tree further along simply gets measured by a later frame.
GREEN_MAX_Z_M = 10.0
# heights sampled straight up from the walking surface, in metres
CANOPY_HEIGHTS_M = (2.5, 4.0, 6.0, 8.0)
CANOPY_X_SAMPLES = 9


def near_cluster(xs: np.ndarray, min_px: int) -> np.ndarray:
    """The run of pixels closest to the kerb, not everything labelled sidewalk.

    Taking a percentile spread over all sidewalk pixels in a band silently
    merges the near footway with any other paved area at the same range, which
    is how a sidewalk ends up measuring eight metres wide.
    """
    xs = np.sort(xs)
    if xs.size == 0:
        return xs
    breaks = np.where(np.diff(xs) > CLUSTER_GAP_M)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [xs.size]))
    for a, b in zip(starts, ends):
        if b - a >= min_px:
            return xs[a:b]
    return xs[starts[0]:ends[0]]


def _solve_theta(theta_d: np.ndarray, k1: float, k2: float, iters: int = 8) -> np.ndarray:
    """Invert theta_d = theta (1 + k1 theta^2 + k2 theta^4) by Newton iteration."""
    th = theta_d.copy()
    for _ in range(iters):
        th2 = th * th
        f = th * (1 + k1 * th2 + k2 * th2 * th2) - theta_d
        df = 1 + 3 * k1 * th2 + 5 * k2 * th2 * th2
        th = th - f / np.where(np.abs(df) < 1e-9, 1e-9, df)
    return th


def _undistort_radial(r_d: np.ndarray, k1: float, k2: float, iters: int = 8) -> np.ndarray:
    """Invert r_d = r (1 + k1 r^2 + k2 r^4) for the perspective model."""
    r = r_d.copy()
    for _ in range(iters):
        r2 = r * r
        f = r * (1 + k1 * r2 + k2 * r2 * r2) - r_d
        df = 1 + 3 * k1 * r2 + 5 * k2 * r2 * r2
        r = r - f / np.where(np.abs(df) < 1e-9, 1e-9, df)
    return r


def rays(u: np.ndarray, v: np.ndarray, w: int, h: int, camera_type: str,
         focal: float, k1: float, k2: float) -> np.ndarray:
    """Unit direction vectors in camera coordinates: +X right, +Y down, +Z forward."""
    S = max(w, h)
    x_n = (u - w / 2 + 0.5) / S
    y_n = (v - h / 2 + 0.5) / S
    r_n = np.hypot(x_n, y_n)
    safe = np.where(r_n < 1e-9, 1e-9, r_n)

    if str(camera_type).lower().startswith("fisheye"):
        theta = _solve_theta(r_n / focal, k1, k2)
        st = np.sin(theta)
        d = np.stack([st * x_n / safe, st * y_n / safe, np.cos(theta)], axis=-1)
    else:
        r_u = _undistort_radial(r_n / focal, k1, k2)
        scale = r_u / safe * focal
        xu, yu = x_n * scale / focal, y_n * scale / focal
        d = np.stack([xu, yu, np.ones_like(xu)], axis=-1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d


def project(X, Y, Z, w: int, h: int, camera_type: str,
            focal: float, k1: float, k2: float):
    """World point to pixel. The inverse of `rays`, needed to ask where a
    specific point in space lands in the image.

    Coordinates are camera centred: +X right, +Y down, +Z forward. A point one
    metre above the road at camera height H has Y = H - 1.
    """
    X, Y, Z = np.asarray(X, float), np.asarray(Y, float), np.asarray(Z, float)
    S = max(w, h)
    if str(camera_type).lower().startswith("fisheye"):
        l = np.hypot(X, Y)
        l = np.where(l < 1e-9, 1e-9, l)
        theta = np.arctan2(l, Z)
        t2 = theta * theta
        theta_d = theta * (1 + k1 * t2 + k2 * t2 * t2)
        xn, yn = focal * theta_d * X / l, focal * theta_d * Y / l
    else:
        Z = np.where(np.abs(Z) < 1e-9, 1e-9, Z)
        xu, yu = X / Z, Y / Z
        r2 = xu * xu + yu * yu
        dd = 1 + k1 * r2 + k2 * r2 * r2
        xn, yn = focal * dd * xu, focal * dd * yu
    return xn * S + w / 2 - 0.5, yn * S + h / 2 - 0.5


def ground_points(d: np.ndarray, height_m: float, pitch_deg: float = 0.0):
    """Intersect rays with the ground plane. Returns lateral X and forward Z in metres.

    Pixels whose ray points at or above the horizon return NaN, which is
    correct: they see no ground.
    """
    if pitch_deg:
        p = np.radians(pitch_deg)
        cp, sp = np.cos(p), np.sin(p)
        dy = d[..., 1] * cp + d[..., 2] * sp
        dz = -d[..., 1] * sp + d[..., 2] * cp
    else:
        dy, dz = d[..., 1], d[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(dy > 1e-6, height_m / dy, np.nan)
    return d[..., 0] * t, dz * t


def parse_camera(row) -> tuple[str, float, float, float]:
    """Pull camera_type and the three intrinsics out of an image metadata row."""
    ctype = str(getattr(row, "camera_type", "perspective") or "perspective")
    raw = getattr(row, "camera_parameters", None)
    focal, k1, k2 = 0.47, 0.0, 0.0     # a sane fisheye default if absent
    if isinstance(raw, str) and raw.strip():
        try:
            vals = json.loads(raw)
            if isinstance(vals, (list, tuple)) and len(vals) >= 1:
                focal = float(vals[0])
                k1 = float(vals[1]) if len(vals) > 1 else 0.0
                k2 = float(vals[2]) if len(vals) > 2 else 0.0
        except Exception:
            pass
    return ctype, focal, k1, k2


def measure_image(seg: np.ndarray, groups: dict[str, list[int]], camera, height_m: float,
                  pitch_deg: float = 0.0, near_side: str = "right") -> dict:
    """Widths and obstructions for one frame, on the near kerb only."""
    ctype, focal, k1, k2 = camera
    h, w = seg.shape
    vv, uu = np.mgrid[0:h, 0:w]
    d = rays(uu.astype(float), vv.astype(float), w, h, ctype, focal, k1, k2)
    X, Z = ground_points(d, height_m, pitch_deg)

    sign = 1.0 if near_side == "right" else -1.0
    Xn = X * sign
    on_ground = np.isfinite(Z) & (Z > Z_MIN_M) & (Z < Z_MAX_M)

    sw = np.isin(seg, groups.get("sidewalk", [])) & on_ground & (Xn > 0)
    road = np.isin(seg, groups.get("road", [])) & on_ground
    curb = np.isin(seg, groups.get("curb", [])) & on_ground & (Xn > 0)

    # Obstructions are located by their BASE, not by all their pixels.
    # Ground plane projection is only valid where an object touches the ground:
    # a pole's shaft and a car's body sit above it, so their pixels project to
    # spurious far away positions. Taking the lowest obstruction pixel in each
    # image column approximates the contact point of whatever stands there,
    # which is the only part whose ground position is meaningful.
    obstruct_ids = sum((groups.get(g, []) for g in ("pole", "sign", "furniture", "car")), [])
    obs_any = np.isin(seg, obstruct_ids)
    obstruct = np.zeros_like(obs_any)
    if obs_any.any():
        rows_idx = np.arange(h)[:, None]
        lowest = np.where(obs_any, rows_idx, -1).max(axis=0)      # per column
        cols = np.flatnonzero(lowest >= 0)
        obstruct[lowest[cols], cols] = True
    obstruct &= on_ground & (Xn > 0)

    marks = np.isin(seg, groups.get("lane_marking", [])) & on_ground

    # Two different questions about greenery, and they need different geometry.
    #
    # Canopy is about what is OVERHEAD, so it needs no ground projection at all:
    # in each image column where the footway is, is there vegetation above it.
    # This is the shade measure, and it is the one that connects to the heat
    # island work.
    #
    # Rooted beside the footway is about where a trunk MEETS the ground, so it
    # uses the same base trick as obstructions: the lowest vegetation pixel in
    # a column. A hillside behind a wall has no base near the kerb; a street
    # tree does.
    veg = np.isin(seg, groups.get("vegetation", []))
    veg_base = np.zeros_like(veg)
    if veg.any():
        rows_idx = np.arange(h)[:, None]
        low = np.where(veg, rows_idx, -1).max(axis=0)
        cc = np.flatnonzero(low >= 0)
        veg_base[low[cc], cc] = True
    veg_base &= on_ground & (Xn > 0)
    veg_top = np.where(veg, np.arange(h)[:, None], h).min(axis=0)   # per column
    out = {"n_sidewalk_px": int(sw.sum()), "n_curb_px": int(curb.sum())}

    widths, insets, road_widths, blocked, lane_widths = [], [], [], [], []
    canopy, rooted, gaps, touch = [], [], [], []
    implausible = 0
    edges = np.arange(Z_MIN_M, Z_MAX_M + 1e-9, Z_BAND_M)
    for z0, z1 in zip(edges[:-1], edges[1:]):
        band = on_ground & (Z >= z0) & (Z < z1)
        b_sw, b_road = sw & band, road & band
        if b_road.sum() >= MIN_PIXELS_PER_BAND:
            xr = X[b_road]
            # only use bands where the carriageway is seen on both sides of the
            # camera, otherwise a clipped or occluded view reads as a narrow road
            # and drags the calibration down
            if (xr < -0.5).any() and (xr > 0.5).any():
                road_widths.append(np.percentile(xr, 97.5) - np.percentile(xr, 2.5))
        b_mark = marks & band
        if b_mark.sum() >= MIN_PIXELS_PER_BAND:
            xm = np.sort(X[b_mark])
            cuts = np.where(np.diff(xm) > LANE_GAP_M)[0]
            starts = np.concatenate(([0], cuts + 1))
            ends = np.concatenate((cuts + 1, [xm.size]))
            centres = [float(np.median(xm[a:b])) for a, b in zip(starts, ends)
                       if b - a >= 5]
            for gap in np.diff(sorted(centres)):
                if LANE_MIN_M <= gap <= LANE_MAX_M:
                    lane_widths.append(float(gap))

        if b_sw.sum() < MIN_PIXELS_PER_BAND:
            continue
        xs = near_cluster(Xn[b_sw], MIN_PIXELS_PER_BAND // 2)
        if xs.size < MIN_PIXELS_PER_BAND // 2:
            continue
        # same trim as the road width, so the two are comparable and the
        # residual bias from trimming cancels when one calibrates the other
        lo, hi = np.percentile(xs, 2.5), np.percentile(xs, 97.5)
        if hi - lo > MAX_PLAUSIBLE_WIDTH_M:
            implausible += 1
            continue
        widths.append(hi - lo)
        insets.append(lo)                       # kerb side edge, distance from camera axis
        # blocked is now a property of the stretch, not a pixel share: does
        # anything stand on the footway in this two metre band, yes or no
        xo = Xn[obstruct & band]
        blocked.append(float(bool(((xo >= lo) & (xo <= hi)).any())))

        # Alternative 1: something standing on the footway interrupts the mask,
        # so the widest unbroken run is narrower than the full extent. This is
        # a ratio measured at one range, so the scale error largely cancels.
        allx = np.sort(Xn[b_sw])
        if allx.size >= 4:
            full = np.percentile(allx, 97.5) - np.percentile(allx, 2.5)
            run = hi - lo
            gaps.append(float(max(0.0, 1.0 - run / full)) if full > 0.1 else 0.0)

        # Alternative 2: pure image space. Is an obstruction class in contact
        # with the top edge of the footway in this column, which is what a pole
        # rising off the walk looks like, and what a car on the road does not.
        cols_sw = np.nonzero(b_sw.any(axis=0))[0]
        if cols_sw.size:
            top = np.where(b_sw, np.arange(h)[:, None], h).min(axis=0)[cols_sw]
            above = np.clip(top - 2, 0, h - 1)
            touch.append(float(np.mean(obs_any[above, cols_sw])))

        # canopy: of the image columns this stretch of footway occupies, how
        # many have vegetation somewhere above the walking surface
        if z0 < GREEN_MAX_Z_M:
            # Shade needs canopy DIRECTLY ABOVE the walking surface, not merely
            # higher up the same image column. A column is a ray, not a vertical
            # line in the world, so a tree standing behind the footway appears
            # above it in the image while shading nothing a pedestrian uses.
            #
            # At Monterrey's latitude the summer sun at solar noon is within a
            # few degrees of vertical, so at the hours that matter, shade falls
            # almost straight down. Directly above is the shade condition.
            zc = 0.5 * (z0 + z1)
            xs_samp = np.linspace(lo, hi, CANOPY_X_SAMPLES) * sign
            hit = np.zeros(CANOPY_X_SAMPLES, dtype=bool)
            for hgt in CANOPY_HEIGHTS_M:
                uu, vv = project(xs_samp, height_m - hgt, zc, w, h,
                                 ctype, focal, k1, k2)
                ui = np.clip(np.round(uu).astype(int), 0, w - 1)
                vi = np.clip(np.round(vv).astype(int), 0, h - 1)
                hit |= veg[vi, ui]
            canopy.append(float(hit.mean()))

            xv = Xn[veg_base & band]
            rooted.append(float(bool(((xv >= lo - 1.0) & (xv <= hi + 1.0)).any())))

    out["width_m"] = float(np.median(widths)) if widths else np.nan
    out["width_iqr_m"] = float(np.subtract(*np.percentile(widths, [75, 25]))) if len(widths) > 2 else np.nan
    out["n_bands"] = len(widths)
    out["n_bands_implausible"] = implausible
    out["kerb_offset_m"] = float(np.median(insets)) if insets else np.nan
    out["road_width_m"] = float(np.median(road_widths)) if road_widths else np.nan
    # share of the measured footway length that has something standing on it
    out["obstructed_frac"] = float(np.mean(blocked)) if blocked else 0.0
    out["n_bands_blocked"] = int(sum(blocked))
    out["gap_frac"] = float(np.mean(gaps)) if gaps else 0.0
    out["touching_frac"] = float(np.mean(touch)) if touch else 0.0
    # share of the footway with greenery overhead, the shade proxy
    out["canopy_frac"] = float(np.mean(canopy)) if canopy else 0.0
    # share of the footway with something green rooted within a metre of it
    out["rooted_frac"] = float(np.mean(rooted)) if rooted else 0.0
    out["lane_width_m"] = float(np.median(lane_widths)) if lane_widths else np.nan
    out["n_lane_gaps"] = len(lane_widths)
    return out
