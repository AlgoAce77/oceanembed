
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi
from scipy.special import erf, erfinv

# --------------------------------------------------------------------------------------
# Grid (0.25 deg, North Indian Ocean) and the 15 standard depth levels from the problem statement
# --------------------------------------------------------------------------------------
LON = np.arange(45.0, 105.0001, 0.25, dtype=np.float32)   # 241
LAT = np.arange(5.0, 30.0001, 0.25, dtype=np.float32)     # 101
LONG, LATG = np.meshgrid(LON, LAT)                          # (H, W)
H, W = LATG.shape
DEPTHS = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000], dtype=np.float32)
K = len(DEPTHS)

# --------------------------------------------------------------------------------------
# Coarse coastline (demo only). Persian Gulf and Gulf of Thailand are treated as land.
# --------------------------------------------------------------------------------------
_MAINLAND = [
    (45, 30), (45, 12.8), (46.5, 13.0), (48.5, 14.0), (50.5, 15.2), (52.2, 16.0), (54.0, 17.0),
    (55.3, 17.8), (56.5, 18.9), (57.7, 19.7), (58.8, 20.4), (59.8, 22.4), (58.6, 23.6), (56.7, 24.5),
    (56.3, 25.4), (56.3, 26.3), (56.3, 27.2), (57.1, 26.9), (57.8, 25.7), (60.6, 25.3), (62.3, 25.1),
    (64.6, 25.2), (66.5, 25.4), (67.2, 24.8), (67.5, 24.1), (68.5, 23.6), (69.0, 22.9), (68.9, 22.3),
    (69.7, 21.6), (70.4, 20.9), (71.0, 20.7), (72.1, 21.75), (72.6, 22.3), (72.8, 21.2), (72.8, 20.4),
    (72.8, 19.0), (73.0, 18.0), (73.3, 17.0), (73.8, 15.5), (74.1, 14.8), (74.6, 13.4), (74.85, 12.85),
    (75.5, 11.5), (76.25, 9.95), (76.5, 9.0), (76.9, 8.5), (77.55, 8.1), (78.2, 8.8), (79.0, 9.2),
    (79.3, 9.35), (79.85, 10.3), (80.1, 11.5), (80.3, 13.1), (80.2, 14.4), (80.0, 15.5), (81.2, 16.2),
    (82.3, 17.0), (83.3, 17.7), (84.0, 18.3), (85.0, 19.3), (85.8, 19.8), (86.65, 20.3), (87.1, 21.5),
    (88.1, 21.65), (89.0, 21.75), (90.0, 22.0), (90.9, 22.4), (91.4, 22.7), (91.8, 22.3), (91.95, 21.4),
    (92.3, 20.9), (92.9, 20.15), (93.5, 19.6), (94.3, 18.3), (94.4, 16.8), (94.2, 16.0), (95.0, 15.8),
    (95.5, 15.9), (96.3, 16.5), (97.6, 16.4), (97.85, 15.25), (98.2, 14.1), (98.6, 12.4), (98.55, 10.0),
    (98.4, 9.0), (98.3, 7.9), (98.9, 8.0), (99.5, 7.4), (100.0, 6.7), (100.3, 5.4), (100.6, 5.0),
    (105, 5), (105, 30),
]
_SOMALIA = [(45, 10.4), (47, 11.2), (49, 11.4), (51.2, 11.8), (51.1, 10.4), (50.5, 9.0), (49.8, 8.0),
            (48.8, 6.0), (48.3, 5.0), (45, 5.0)]
_SRI_LANKA = [(79.85, 9.8), (80.7, 9.4), (81.3, 8.5), (81.9, 7.5), (81.6, 6.5), (81.2, 6.2), (80.6, 5.95),
              (80.0, 6.0), (79.85, 7.0), (79.8, 8.2)]
_ANDAMAN = [(92.7, 13.5), (93.2, 13.4), (93.1, 12.0), (92.9, 11.0), (92.6, 11.6), (92.6, 12.7)]
_SOCOTRA = [(53.3, 12.65), (54.3, 12.6), (54.0, 12.3), (53.4, 12.3)]

LAND_POLYS = [np.array(p, dtype=np.float32) for p in (_MAINLAND, _SOMALIA, _SRI_LANKA, _ANDAMAN, _SOCOTRA)]


def _in_poly(px, py, poly):
    x, y = poly[:, 0], poly[:, 1]
    inside = np.zeros(px.shape, dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi, xj, yj = x[i], y[i], x[j], y[j]
        cond = ((yi > py) != (yj > py)) & (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi)
        inside ^= cond
        j = i
    return inside


LAND = np.zeros((H, W), dtype=bool)
for _p in LAND_POLYS:
    LAND |= _in_poly(LONG, LATG, _p)
OCEAN = ~LAND
DIST = (ndi.distance_transform_edt(OCEAN) * 0.25).astype(np.float32)          # deg to nearest land
_, _NEAREST = ndi.distance_transform_edt(LAND, return_indices=True)              # nearest ocean pixel
COASTW = lambda scale: np.exp(-DIST / scale).astype(np.float32)                  # noqa: E731
BATHY = np.clip(60 + 3600 * (1 - np.exp(-DIST / 2.2)), 60, 4500).astype(np.float32)
CORIOLIS = (2 * 7.2921e-5 * np.sin(np.deg2rad(LATG))).astype(np.float32)

# Illustrative mooring positions. Replace with the real INCOIS OMNI / RAMA coordinates.
DEMO_BUOYS = [
    ("Demo mooring, north Bay of Bengal", 18.0, 89.0),
    ("Demo mooring, central Bay of Bengal", 12.0, 90.0),
    ("Demo mooring, south Bay of Bengal", 8.0, 90.0),
    ("Demo mooring, north-east Arabian Sea", 17.0, 68.5),
    ("Demo mooring, central Arabian Sea", 12.0, 65.0),
    ("Demo mooring, south-east Arabian Sea", 10.0, 72.0),
]
BUOY_DEPTHS = np.array([0, 10, 20, 50, 75, 100, 125, 150, 200, 300, 500], dtype=np.float32)


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _g(lat0, lon0, slat, slon):
    return np.exp(-(((LATG - lat0) / slat) ** 2) - ((LONG - lon0) / slon) ** 2).astype(np.float32)


def nearest_ocean_index(lat: float, lon: float) -> tuple[int, int]:
    i = int(np.clip(round((lat - float(LAT[0])) / 0.25), 0, H - 1))
    j = int(np.clip(round((lon - float(LON[0])) / 0.25), 0, W - 1))
    if LAND[i, j]:
        i, j = int(_NEAREST[0, i, j]), int(_NEAREST[1, i, j])
    return i, j


@lru_cache(maxsize=512)
def _base_noise(tag: int, k: int, sigma_cells: float) -> np.ndarray:
    rng = np.random.default_rng([tag, k])
    n = ndi.gaussian_filter(rng.standard_normal((H, W)), sigma_cells, mode="reflect")
    n = (n - n.mean()) / n.std()
    return n.astype(np.float32)


def _tnoise(tag: int, date: dt.date, period: float = 9.0, sigma_cells: float = 5.0) -> np.ndarray:
    """Smooth-in-space, slowly-evolving-in-time unit-variance noise."""
    t = date.toordinal() / period
    k = int(np.floor(t))
    a = t - k
    f = (1 - a) * _base_noise(tag, k, sigma_cells) + a * _base_noise(tag, k + 1, sigma_cells)
    return (f / np.sqrt((1 - a) ** 2 + a ** 2)).astype(np.float32)


# --------------------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    date: dt.date | None
    blurb: str
    fresh: float = 1.0      # strength of the Ganges-Brahmaputra plume
    upw: float = 1.0        # strength of coastal upwelling
    heat: float = 0.0       # marine heatwave amplitude


SCENARIOS = {
    "auto": Scenario("auto", "Seasonal climatology", None,
                     "Fields follow the calendar: monsoon winds, upwelling and the river plume."),
    "bob": Scenario("bob", "Bay of Bengal barrier layer", dt.date(2021, 12, 5),
                    "Post-monsoon freshwater cap traps warm water below a cooler surface (temperature inversion).",
                    fresh=1.35),
    "amphan": Scenario("amphan", "Cyclone Amphan replay (illustrative)", dt.date(2020, 5, 19),
                       "Approximate track 16-21 May 2020: strong winds, cold wake, deeper mixing, wider uncertainty."),
    "upwelling": Scenario("upwelling", "Arabian Sea monsoon upwelling", dt.date(2021, 7, 20),
                          "SW-monsoon winds lift cold water along the Somali, Omani and south-west Indian coasts.",
                          upw=1.3),
    "heatwave": Scenario("heatwave", "Marine heatwave", dt.date(2022, 5, 10),
                         "Warm anomaly over the western Bay of Bengal that reaches well below the surface.",
                         heat=1.0),
}

_AMPHAN = [  # (date, lat, lon) approximate
    (dt.date(2020, 5, 16), 10.5, 87.0), (dt.date(2020, 5, 17), 12.8, 87.0), (dt.date(2020, 5, 18), 15.4, 87.4),
    (dt.date(2020, 5, 19), 18.3, 87.7), (dt.date(2020, 5, 20), 21.3, 88.2), (dt.date(2020, 5, 21), 24.5, 89.0),
]


def cyclone_center(date: dt.date):
    for d, la, lo in _AMPHAN:
        if d == date:
            return la, lo
    return None


def _cyclone_fields(date: dt.date):
    """Return (wake influence 0..1.3, vortex u, vortex v, centre or None)."""
    wake = np.zeros((H, W), np.float32)
    center = cyclone_center(date)
    for age in range(0, 6):
        c = cyclone_center(date - dt.timedelta(days=age))
        if c is None:
            continue
        wake += np.exp(-age / 2.5) * _g(c[0], c[1], 2.4, 2.4)
    wake = np.clip(wake, 0, 1.3)
    u = np.zeros((H, W), np.float32)
    v = np.zeros((H, W), np.float32)
    if center is not None:
        yc, xc = center
        dy = LATG - yc
        dx = (LONG - xc) * np.cos(np.deg2rad(LATG))
        r = np.hypot(dx, dy) + 1e-3
        rm = 0.9
        vt = 38.0 * (r / rm) * np.exp(1 - r / rm)
        u = (-vt * dy / r).astype(np.float32)
        v = (vt * dx / r).astype(np.float32)
    return wake, u, v, center


# --------------------------------------------------------------------------------------
# Surface fields
# --------------------------------------------------------------------------------------
@dataclass
class SurfaceFields:
    date: dt.date
    scenario: str
    sst: np.ndarray
    sss: np.ndarray
    ssh: np.ndarray
    u: np.ndarray
    v: np.ndarray
    wu: np.ndarray
    wv: np.ndarray
    aux: dict = field(default_factory=dict)

    @property
    def wind_speed(self):
        return np.hypot(self.wu, self.wv)


def synth_surface(date: dt.date, scenario: str = "auto") -> SurfaceFields:
    sc = SCENARIOS.get(scenario, SCENARIOS["auto"])
    doy = date.timetuple().tm_yday
    ph = 2 * np.pi / 365.0
    m = np.cos(ph * (doy - 196))                       # +1 mid-July (SW monsoon), -1 mid-January (NE monsoon)
    msw, mne = max(m, 0.0), max(-m, 0.0)
    winter = float(np.clip(1.4 * np.cos(ph * (doy - 15)) - 0.1, 0, 1))
    fs = float(np.clip(0.45 + 0.55 * np.cos(ph * (doy - 285)), 0.1, 1.0))   # river plume season (peak Sep-Nov)

    E = _sig((LONG - 79.5) / 1.2).astype(np.float32)   # 1 in the Bay of Bengal, 0 in the Arabian Sea
    eddy = _tnoise(11, date, 9.0, 5.0)
    cyc, cu, cv, ccen = _cyclone_fields(date)

    # ---- winds -----------------------------------------------------------------------
    scale = 1.0 if m > 0 else 0.55
    jet = _g(12, 65, 7, 14)
    wu = m * (6.0 + 6.0 * jet) * scale + 1.4 * _tnoise(21, date, 6.0, 6.0)
    wv = m * (1.5 + 2.5 * _g(15, 62, 8, 8)) * (1.0 if m > 0 else 1.2) + 1.0 * _tnoise(22, date, 6.0, 6.0)
    wu = (wu + cu).astype(np.float32)
    wv = (wv + cv).astype(np.float32)
    wspd = np.hypot(wu, wv)

    # ---- wind-stress curl -> Ekman pumping -> thermocline displacement --------------------
    tau_x = 1.2e-3 * 1.225 * wspd * wu
    tau_y = 1.2e-3 * 1.225 * wspd * wv
    dx_m = (111e3 * 0.25 * np.cos(np.deg2rad(LATG))).astype(np.float32)
    dy_m = 111e3 * 0.25
    curl = np.gradient(tau_y, axis=1) / dx_m - np.gradient(tau_x, axis=0) / dy_m
    wek = curl / (1025.0 * np.maximum(CORIOLIS, 1.2e-5)) * 86400.0            # m/day
    wek = ndi.gaussian_filter(np.clip(wek, -8, 8), 2.0)

    # ---- coastal upwelling ----------------------------------------------------------------
    up_w = np.clip(_g(9.0, 52.0, 6, 3.0) + _g(18.5, 58.0, 4.5, 4.0) + 0.6 * _g(10.5, 75.5, 5, 3.0), 0, 1.2)
    upw = (up_w * COASTW(1.5) * msw * sc.upw).astype(np.float32)

    # ---- salinity and the river plume -----------------------------------------------------
    sss_as = 35.0 + 1.5 * _sig((LATG - 9.0) / 3.5) * _sig((78.5 - LONG) / 2.0)
    plume = (4.5 * _g(20.5, 89.0, 4.5, 6.5) + 2.2 * _g(14.0, 87.5, 5.0, 6.5) + 1.5 * _g(9.0, 82.0, 3.0, 4.0)
             + 2.0 * _g(15.5, 96.0, 3.0, 2.5))
    sss = (1 - E) * sss_as + E * 34.6 - E * fs * sc.fresh * plume + 0.25 * _tnoise(31, date, 12.0, 4.0)
    sss = np.clip(sss, 28.0, 37.5).astype(np.float32)
    fresh = (np.clip((34.2 - sss) / 3.5, 0, 1) * E).astype(np.float32)

    # ---- SST ------------------------------------------------------------------------------
    lat_pos = np.clip(LATG - 5.0, 0, None)
    base = 28.9 - 0.06 * np.clip(LATG - 8.0, 0, None)
    amp = 0.35 + 0.17 * lat_pos
    sst = base + amp * np.cos(ph * (doy - 140))
    sst -= 1.0 * msw * _g(15, 62, 7, 9)                 # open-ocean monsoon mixing
    sst -= 3.2 * upw                                     # coastal upwelling
    sst -= 1.6 * fresh * winter                          # cool, fresh winter surface in the north Bay
    sst -= 2.8 * cyc                                     # cyclone cold wake
    sst += 1.8 * sc.heat * _g(15.0, 86.0, 5.0, 6.0)      # marine heatwave
    sst += 0.25 * _tnoise(41, date, 10.0, 5.0)
    sst = np.clip(sst, 22.0, 32.5).astype(np.float32)

    # ---- SSH / SLA ---------------------------------------------------------------------------
    ssh = (0.08 * np.cos(ph * (doy - 300)) * E - 0.06 * msw * _g(10, 60, 8, 12) + 0.06 * eddy
           - 0.15 * upw - 0.22 * cyc + 0.09 * sc.heat * _g(15.0, 86.0, 5.0, 6.0))
    ssh = ssh.astype(np.float32)

    # ---- currents: geostrophic + monsoon drift ------------------------------------------------
    f = np.maximum(CORIOLIS, 1.2e-5)
    u_g = -(9.81 / f) * np.gradient(ssh, axis=0) / dy_m
    v_g = (9.81 / f) * np.gradient(ssh, axis=1) / dx_m
    u = ndi.gaussian_filter(np.clip(u_g, -1.2, 1.2), 1.0) + 0.45 * m * np.exp(-(((LATG - 6.5) / 3.5) ** 2))
    v = ndi.gaussian_filter(np.clip(v_g, -1.2, 1.2), 1.0)

    for arr in (sst, sss, ssh, u, v, wu, wv):
        arr[LAND] = np.nan

    aux = dict(doy=doy, m=m, msw=msw, winter=winter, E=E, fresh=fresh, upw=upw, cyc=cyc, eddy=eddy,
               wek=wek.astype(np.float32), heat=(sc.heat * _g(15.0, 86.0, 5.0, 6.0)).astype(np.float32),
               cyclone=ccen)
    return SurfaceFields(date, scenario, sst, sss, ssh, u.astype(np.float32), v.astype(np.float32), wu, wv, aux)


def stack_inputs(sf: SurfaceFields) -> np.ndarray:
    """(9, H, W) raw model inputs: SST, SSS, SSH, U, V, Wind-U, Wind-V, Coriolis, Bathymetry."""
    x = np.stack([sf.sst, sf.sss, sf.ssh, sf.u, sf.v, sf.wu, sf.wv, CORIOLIS, BATHY]).astype(np.float32)
    return np.nan_to_num(x, nan=0.0)


# --------------------------------------------------------------------------------------
# Subsurface reference ("truth") and the demo reconstruction
# --------------------------------------------------------------------------------------
def emulate_truth(sf: SurfaceFields) -> dict:
    a = sf.aux
    z = DEPTHS[:, None, None]
    wspd = ndi.gaussian_filter(np.nan_to_num(sf.wind_speed), 2.0)
    E, fresh, winter = a["E"], a["fresh"], a["winter"]

    mld = (20 + 2.8 * wspd + 30 * winter * _sig((LATG - 12) / 3) * (1 - 0.6 * E)
           - 16 * fresh + 30 * a["cyc"] + 8 * a["msw"])
    mld = np.clip(mld, 8, 95).astype(np.float32)

    d20 = (135 - 2.0 * np.clip(LATG - 5, 0, None) + 320 * np.nan_to_num(sf.ssh) - 55 * a["upw"]
           - 6.0 * a["wek"] + 25 * a["heat"])
    d20 = np.clip(np.maximum(d20, mld + 25), 30, 260).astype(np.float32)

    t1000 = (5.5 + 1.0 * _sig((LATG - 11) / 2.5) * (1 - E)).astype(np.float32)
    sst = np.nan_to_num(sf.sst, nan=28.0)

    # thermocline (error-function step) + slow deep decay; the 20 degC crossing is pinned near d20
    frac, Lt = 0.5, 250.0
    amp = sst - t1000
    a_th = frac * amp
    a_dp = amp - a_th

    def ramp(x):
        return 0.5 * ((x - mld) + np.sqrt((x - mld) ** 2 + 100.0))

    nrm = 1 - np.exp(-ramp(1000.0) / Lt)

    def deep(x):
        return a_dp * (1 - np.exp(-ramp(x) / Lt)) / nrm

    p = np.clip((sst - 20.0 - deep(d20)) / np.maximum(a_th, 1e-3), 0.03, 0.97)
    xq = erfinv(2 * p - 1)
    wth = np.clip((d20 - mld) / (xq + 1.6), 16.0, 90.0)
    dc = d20 - wth * xq

    def phi(x):
        return 0.5 * (1 + erf((x - dc) / wth))

    T = sst - a_th * phi(z) - deep(z)
    T = T + (sst - T[0:1]) * (1 - phi(z))        # keep the mixed layer at the observed SST

    # barrier-layer temperature inversion (warm water below the cool, fresh cap)
    inv_amp = 2.4 * fresh * winter
    zc = mld + 20.0
    T = T + inv_amp * np.exp(-(((z - zc) / 20.0) ** 2)) * (1 - np.exp(-z / 8.0))
    T = T.astype(np.float32)
    T[:, LAND] = np.nan
    return dict(T=T, mld=mld, d20_driver=d20, inv_amp=inv_amp.astype(np.float32), t1000=t1000)


def emulate_prediction(sf: SurfaceFields, ref: dict) -> tuple[np.ndarray, np.ndarray]:
    """Demo 'reconstruction': reference + an error whose size equals the reported sigma (well calibrated)."""
    a = sf.aux
    z = DEPTHS[:, None, None]
    base = 0.08 + 0.72 * (z / 1000.0) ** 1.3
    eddy_act = np.clip(np.abs(a["eddy"]) / 1.6, 0, 1.5)
    mult = 1 + 0.9 * eddy_act + 0.6 * a["fresh"] + 1.4 * a["cyc"] + 0.4 * a["upw"] + 0.3 * COASTW(0.75)
    bump = 0.30 * np.exp(-(((z - ref["d20_driver"]) / 45.0) ** 2))
    sig_true = (base * mult + bump * (1 + 0.5 * eddy_act)).astype(np.float32)

    # vertically correlated unit-variance noise from 5 latent fields
    lat_f = [_tnoise(200 + i, sf.date, 9.0, 4.0) for i in range(5)]
    pos = np.linspace(0, 4, K)
    eps = np.empty((K, H, W), np.float32)
    for k, p in enumerate(pos):
        i0 = int(np.floor(p))
        i1 = min(i0 + 1, 4)
        w1 = p - i0
        eps[k] = ((1 - w1) * lat_f[i0] + w1 * lat_f[i1]) / np.sqrt((1 - w1) ** 2 + w1 ** 2)

    mu = ref["T"] + sig_true * eps
    sigma = sig_true * np.exp(0.12 * _tnoise(300, sf.date, 9.0, 4.0))[None]
    mu = mu.astype(np.float32)
    sigma = sigma.astype(np.float32)
    mu[:, LAND] = np.nan
    sigma[:, LAND] = np.nan
    return mu, sigma


# --------------------------------------------------------------------------------------
# Derived physical metrics
# --------------------------------------------------------------------------------------
def d20_from_profiles(T: np.ndarray) -> np.ndarray:
    """Depth (m) of the deepest 20 degC crossing. T: (K, ...) -> (...). NaN if none."""
    above = T >= 20.0
    cross = above[:-1] & ~above[1:]
    has = cross.any(axis=0)
    last = (cross.shape[0] - 1) - np.argmax(cross[::-1], axis=0)
    t0 = np.take_along_axis(T[:-1], last[None], 0)[0]
    t1 = np.take_along_axis(T[1:], last[None], 0)[0]
    z0, z1 = DEPTHS[:-1][last], DEPTHS[1:][last]
    frac = np.clip((t0 - 20.0) / np.where(np.abs(t0 - t1) < 1e-3, 1e-3, t0 - t1), 0, 1)
    d = z0 + frac * (z1 - z0)
    return np.where(has, d, np.nan).astype(np.float32)


def uohc_gj_m2(T: np.ndarray, zmax: float = 700.0) -> np.ndarray:
    """Upper-ocean heat content, rho*cp*integral(T dz) from 0 to zmax, in GJ/m^2."""
    idx = np.where(DEPTHS <= zmax)[0]
    zz = DEPTHS[idx]
    tt = T[idx]
    dz = np.diff(zz)[:, None, None] if T.ndim == 3 else np.diff(zz)
    integral = np.sum(0.5 * (tt[:-1] + tt[1:]) * dz, axis=0)
    return (1025.0 * 3985.0 * integral / 1e9).astype(np.float32)


def profile_inversion(T_profile: np.ndarray) -> tuple[float, float]:
    """(strength degC, depth m) of a sub-surface temperature maximum warmer than the surface."""
    sel = (DEPTHS >= 10) & (DEPTHS <= 150)
    k = int(np.argmax(np.where(sel, T_profile, -np.inf)))
    return float(T_profile[k] - T_profile[0]), float(DEPTHS[k])


# --------------------------------------------------------------------------------------
# Synthetic mooring observations (independent of the model error, sensor noise only)
# --------------------------------------------------------------------------------------
def synthetic_buoy_profile(ref_T: np.ndarray, lat: float, lon: float, date: dt.date):
    i, j = nearest_ocean_index(lat, lon)
    prof = ref_T[:, i, j]
    rng = np.random.default_rng([int(round(lat * 100)) + 5000, int(round(lon * 100)) + 5000, date.toordinal()])
    vals = np.interp(BUOY_DEPTHS, DEPTHS, prof) + rng.normal(0, 0.07, size=BUOY_DEPTHS.shape)
    return BUOY_DEPTHS.copy(), vals.astype(np.float32)
