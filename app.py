"""
SubTherm / OceanEmbed - interactive prototype (Streamlit).

Run:  streamlit run app.py

DEMO MODE: surface fields and 'reconstructions' are synthetic (see emulator.py). Nothing has been trained
on GLORYS, ARGO or satellite data. The H-SCAN network in hscan.py is real PyTorch code with untrained weights.
"""
from __future__ import annotations

import datetime as dt
import inspect
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import emulator as em

try:
    import hscan
    TORCH_OK, TORCH_ERR = True, ""
except Exception as exc:  # torch not installed
    hscan, TORCH_OK, TORCH_ERR = None, False, str(exc)

st.set_page_config(page_title="SubTherm | OceanEmbed", layout="wide",
                   initial_sidebar_state="expanded")

# ======================================================================================
# Design tokens
# ======================================================================================
INK, MUTED, RULE, PAPER = "#0d2233", "#5b7080", "#c9d6d9", "#eef3f3"
SEA, PIN_A, PIN_B = "#0b5d7a", "#0b3c5d", "#d9822b"
LAND_FILL, COAST = "#e6dcc0", "#8b7f5d"
FONT = "Instrument Sans, system-ui, sans-serif"

THERMAL = [[0, "#1c2a6b"], [0.14, "#2a5aa8"], [0.28, "#3d8fc3"], [0.42, "#6cc3bd"], [0.56, "#c8e58f"],
           [0.70, "#f7dd7a"], [0.83, "#f59b4c"], [0.93, "#dc5133"], [1, "#a3162e"]]
PLUM = [[0, "#f4eef2"], [0.35, "#d9b6cf"], [0.7, "#9c4a8f"], [1, "#3d1350"]]
DEEP = [[0, "#eaf4e6"], [0.5, "#4aa3a0"], [1, "#0a3a5c"]]
SALT = [[0, "#f2f7d3"], [0.3, "#8fd0b5"], [0.6, "#2f8ea8"], [1, "#12385e"]]
DIVERGE = [[0, "#2b5fa8"], [0.5, "#f4f4ef"], [1, "#c4432b"]]
WIND = [[0, "#f4f7f5"], [0.5, "#7fb3c9"], [1, "#243b6b"]]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Instrument+Sans:wght@400;500;600&display=swap');
html, body, .stApp, [data-testid="stSidebar"] { font-family: 'Instrument Sans', system-ui, sans-serif; }
h1, h2, h3, h4 { font-family: 'Fraunces', Georgia, serif !important; font-weight: 600 !important; letter-spacing: -0.01em; }
.block-container { padding-top: 1.2rem; max-width: 1560px; }
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
[data-testid="stSidebar"] { border-right: 1px solid #c9d6d9; }
.stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid #c9d6d9; }
.stTabs [data-baseweb="tab"] { height: auto; padding: 0.45rem 0; font-weight: 500; }
.hero { display: grid; grid-template-columns: minmax(280px, 1.1fr) 1.4fr; gap: 2.2rem; align-items: end; padding: 0.4rem 0 1rem 0; }
.hero h1 { font-size: 3.1rem; line-height: 1; margin: 0 0 0.5rem 0; padding: 0; }
.hero p { margin: 0 0 0.35rem 0; color: #33495a; max-width: 34rem; line-height: 1.45; }
.hero .who { font-size: 0.82rem; color: #5b7080; }
.ribbon { height: 34px; border-radius: 3px; border: 1px solid #0d2233; }
.ribbon-ticks { display: flex; justify-content: space-between; font-size: 0.75rem; color: #5b7080; margin-top: 4px; }
.ribbon-cap { font-size: 0.82rem; color: #33495a; margin-bottom: 6px; }
.notice { border-left: 3px solid #d9822b; padding: 0.35rem 0.8rem; background: rgba(217,130,43,0.09); font-size: 0.88rem; margin-bottom: 0.7rem; }
.notice.ok { border-left-color: #0b5d7a; background: rgba(11,93,122,0.07); }
.readout { display: grid; grid-template-columns: 1.5fr 1fr 1fr 1.1fr 1.4fr; border-top: 1px solid #c9d6d9; border-bottom: 1px solid #c9d6d9; margin: 0.4rem 0 1rem 0; }
.readout > div { padding: 0.5rem 1rem; border-left: 1px solid #c9d6d9; }
.readout > div:first-child { border-left: none; padding-left: 0; }
.readout .k { font-size: 0.78rem; color: #5b7080; }
.readout .v { font-family: 'Fraunces', Georgia, serif; font-size: 1.5rem; line-height: 1.2; }
.readout .v small { font-family: 'Instrument Sans', sans-serif; font-size: 0.8rem; color: #5b7080; }
@media (max-width: 900px) { .hero { grid-template-columns: 1fr; } .readout { grid-template-columns: 1fr 1fr; } }
.brand { font-family: 'Fraunces', Georgia, serif; font-size: 1.5rem; font-weight: 600; line-height: 1; }
.brand-sub { font-size: 0.8rem; color: #5b7080; margin-bottom: 0.8rem; }
.flow { border-left: 2px solid #c9d6d9; margin-left: 0.4rem; padding-left: 1rem; }
.node { padding: 0.45rem 0.8rem; margin: 0.35rem 0; border-left: 4px solid #0d2233; background: rgba(255,255,255,0.6); }
.node b { font-family: 'Fraunces', Georgia, serif; }
.node span { display: block; font-size: 0.85rem; color: #5b7080; }
.node.att { border-left-color: #0b5d7a; }
.node.mean { border-left-color: #d9822b; }
.node.var { border-left-color: #9c4a8f; }
.split { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; }
.timeline { display: flex; height: 34px; border: 1px solid #0d2233; border-radius: 3px; overflow: hidden; font-size: 0.8rem; }
.timeline div { display: flex; align-items: center; justify-content: center; color: #fff; }
</style>
"""


def html(s: str):
    st.markdown("".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


html(CSS)

# ======================================================================================
# Constants and state
# ======================================================================================
DEPTH_INTS = [int(z) for z in em.DEPTHS]
REGIONS = {
    "Full North Indian Ocean": (45, 105, 5, 30, 470),
    "Bay of Bengal": (78, 102, 5, 24, 560),
    "Head of the Bay (river plume)": (82, 96, 14, 23, 500),
    "Arabian Sea": (45, 79, 5, 27, 560),
}
LAYERS = ["Temperature", "Uncertainty (±2σ)", "Thermocline depth (20 °C)", "Heat content 0–700 m",
          "SST (input)", "SSS (input)", "SSH (input)", "Wind speed (input)"]
DEPTH_LAYERS = {"Temperature", "Uncertainty (±2σ)"}
JUMPS = {"Head of the Bay (river plume)": (20.5, 89.0), "Central Bay of Bengal": (14.0, 87.0),
         "Southern Bay of Bengal": (8.5, 88.0), "Open Arabian Sea": (15.0, 66.0),
         "Off Oman (upwelling coast)": (19.0, 58.8), "Off Somalia": (9.0, 54.0),
         "Off Kerala": (10.5, 74.5)}
JUMPS.update({name: (la, lo) for name, la, lo in em.DEMO_BUOYS})

_HOVER_MASK = em.OCEAN[::2, ::2]
_HLON = em.LONG[::2, ::2][_HOVER_MASK]
_HLAT = em.LATG[::2, ::2][_HOVER_MASK]

ss = st.session_state
for key, val in dict(scenario="bob", date=dt.date(2021, 12, 5), depth=50, layer="Temperature",
                     region="Full North Indian Ocean", latA=18.5, lonA=88.5, latB=15.0, lonB=66.0,
                     target="A", compare=True, engine="emulator", show_buoys=True, stretched=True,
                     show_ref=True, hscan_model=None, hscan_stats={}, hscan_token=0, hscan_source="none",
                     buoy_df=None, validation=None, upname=None, ckname=None).items():
    ss.setdefault(key, val)


def snap(lat: float, lon: float):
    i, j = em.nearest_ocean_index(float(lat), float(lon))
    return float(em.LAT[i]), float(em.LON[j]), i, j


def set_active_point(lat, lon):
    la, lo, _, _ = snap(lat, lon)
    ss[f"lat{ss.target}"], ss[f"lon{ss.target}"] = la, lo


def on_scenario():
    d = em.SCENARIOS[ss.scenario].date
    if d:
        ss.date = d


def on_jump():
    if ss.get("jump"):
        set_active_point(*JUMPS[ss.jump])


def on_map_select():
    ev = ss.get("map")
    try:
        pts = ev["selection"]["points"]
    except Exception:
        pts = []
    if pts:
        set_active_point(float(pts[-1]["y"]), float(pts[-1]["x"]))


def use_hscan():
    ss.engine = "hscan"


# ======================================================================================
# Data access
# ======================================================================================
@st.cache_data(show_spinner="Building the ocean state…", max_entries=24)
def emulator_state(date_iso: str, scenario: str) -> dict:
    sf = em.synth_surface(dt.date.fromisoformat(date_iso), scenario)
    ref = em.emulate_truth(sf)
    mu, sigma = em.emulate_prediction(sf, ref)
    return dict(sf=sf, ref=ref, mu=mu, sigma=sigma)


def ensure_model():
    if ss.hscan_model is None:
        ss.hscan_model = hscan.HSCAN(base_channels=32).eval()
        ss.hscan_token += 1
        ss.hscan_source = "untrained"
    return ss.hscan_model


def get_recon(d: dt.date, scenario: str, engine: str) -> dict:
    R = emulator_state(d.isoformat(), scenario)
    if engine != "hscan" or not TORCH_OK:
        return R
    model = ensure_model()
    cache = ss.setdefault("hscan_cache", {})
    key = (d.isoformat(), scenario, ss.hscan_token)
    if key not in cache:
        stats = ss.hscan_stats or {}
        mu, sg = hscan.predict_numpy(model, em.stack_inputs(R["sf"]), stats.get("temp_mean"), stats.get("temp_std"))
        mu, sg = mu.astype(np.float32), sg.astype(np.float32)
        mu[:, em.LAND], sg[:, em.LAND] = np.nan, np.nan
        if len(cache) >= 16:
            cache.pop(next(iter(cache)))
        cache[key] = (mu, sg)
    out = dict(R)
    out["mu"], out["sigma"] = cache[key]
    return out


def pin_info(R: dict, label: str, color: str, lat: float, lon: float) -> dict:
    la, lo, i, j = snap(lat, lon)
    mu, sg = R["mu"][:, i, j], R["sigma"][:, i, j]
    inv, inv_z = em.profile_inversion(mu)
    return dict(label=label, color=color, lat=la, lon=lo, i=i, j=j, mu=mu, sg=sg, ref=R["ref"]["T"][:, i, j],
                d20=float(em.d20_from_profiles(mu)), inv=inv, inv_z=inv_z)


def obs_near(R: dict, d: dt.date, lat: float, lon: float):
    df = ss.buoy_df
    if df is not None:
        sel = df[(df["date"] == pd.Timestamp(d)) & ((df["lat"] - lat).abs() <= 0.5) & ((df["lon"] - lon).abs() <= 0.5)]
        if len(sel):
            g = sel.groupby("depth_m")["temp_c"].mean()
            return g.index.values.astype(float), g.values.astype(float), "Uploaded mooring"
        return None
    for name, la, lo in em.DEMO_BUOYS:
        if abs(la - lat) <= 0.5 and abs(lo - lon) <= 0.5:
            z, v = em.synthetic_buoy_profile(R["ref"]["T"], la, lo, d)
            return z, v, "Synthetic mooring"
    return None


# ======================================================================================
# Figures
# ======================================================================================
def fmt_ll(lat, lon):
    return f"{lat:.2f}°N, {lon:.2f}°E"


def _hex(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def scale_color(scale, t):
    t = min(max(float(t), 0.0), 1.0)
    for (p0, c0), (p1, c1) in zip(scale[:-1], scale[1:]):
        if t <= p1:
            f = (t - p0) / (p1 - p0 + 1e-9)
            a, b = _hex(c0), _hex(c1)
            return "rgb(%d,%d,%d)" % tuple(round(x + (y - x) * f) for x, y in zip(a, b))
    return "rgb(%d,%d,%d)" % _hex(scale[-1][1])


def layer_field(R, layer, k):
    mu, sf = R["mu"], R["sf"]
    z = int(em.DEPTHS[k])
    if layer == "Temperature":
        return mu[k], "°C", THERMAL, f"Temperature at {z} m", False
    if layer == "Uncertainty (±2σ)":
        return 2 * R["sigma"][k], "±°C", PLUM, f"Uncertainty (±2σ) at {z} m", False
    if layer == "Thermocline depth (20 °C)":
        return em.d20_from_profiles(mu), "m", DEEP, "Depth of the 20 °C isotherm", False
    if layer == "Heat content 0–700 m":
        return em.uohc_gj_m2(mu), "GJ/m²", THERMAL, "Upper-ocean heat content", False
    if layer == "SST (input)":
        return sf.sst, "°C", THERMAL, "Sea surface temperature", False
    if layer == "SSS (input)":
        return sf.sss, "psu", SALT, "Sea surface salinity", False
    if layer == "SSH (input)":
        return sf.ssh, "m", DIVERGE, "Sea surface height anomaly", True
    return sf.wind_speed, "m/s", WIND, "Wind speed", False


def color_range(z, diverging):
    v = z[np.isfinite(z)]
    lo, hi = np.percentile(v, [2, 98])
    if diverging:
        m = max(abs(lo), abs(hi))
        return -m, m
    return float(lo), float(hi)


def build_map(R, layer, k, region, pins, show_buoys, date):
    z, unit, scale, title, diverging = layer_field(R, layer, k)
    z = np.where(em.OCEAN, z, np.nan)
    zmin, zmax = color_range(z, diverging)
    x0, x1, y0, y1, height = REGIONS[region]
    d20 = em.d20_from_profiles(R["mu"])
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=em.LON, y=em.LAT, z=z, colorscale=scale, zmin=zmin, zmax=zmax, zsmooth="best",
                             hoverinfo="skip", colorbar=dict(title=dict(text=unit, side="right"), thickness=10,
                                                             len=0.78, outlinewidth=0, tickfont=dict(size=11))))
    fig.add_trace(go.Contour(x=em.LON, y=em.LAT, z=z, showscale=False, hoverinfo="skip", ncontours=16,
                             contours=dict(coloring="none", showlabels=False),
                             line=dict(width=0.5, color="rgba(255,255,255,0.45)")))
    for poly in em.LAND_POLYS:
        fig.add_trace(go.Scatter(x=poly[:, 0], y=poly[:, 1], mode="lines", fill="toself", fillcolor=LAND_FILL,
                                 line=dict(color=COAST, width=0.8), hoverinfo="skip", showlegend=False))
    # invisible hover / click grid (0.5 deg)
    cd = np.stack([z[::2, ::2][_HOVER_MASK], d20[::2, ::2][_HOVER_MASK],
                   2 * R["sigma"][k][::2, ::2][_HOVER_MASK]], axis=-1)
    fig.add_trace(go.Scatter(
        x=_HLON, y=_HLAT, mode="markers", customdata=cd, showlegend=False,
        marker=dict(size=9, opacity=0), selected=dict(marker=dict(opacity=0)), unselected=dict(marker=dict(opacity=0)),
        hovertemplate=("<b>%{y:.2f}°N, %{x:.2f}°E</b><br>" + title + ": %{customdata[0]:.2f} " + unit +
                       "<br>Thermocline (20 °C): %{customdata[1]:.0f} m<br>±2σ at this depth: %{customdata[2]:.2f} °C"
                       "<extra></extra>")))
    if show_buoys:
        if ss.buoy_df is not None:
            b = ss.buoy_df[["lat", "lon"]].drop_duplicates().head(80)
            bx, by, bt = b["lon"].values, b["lat"].values, ["Uploaded mooring"] * len(b)
        else:
            bx = [lo for _, _, lo in em.DEMO_BUOYS]
            by = [la for _, la, _ in em.DEMO_BUOYS]
            bt = [n for n, _, _ in em.DEMO_BUOYS]
        fig.add_trace(go.Scatter(x=bx, y=by, mode="markers", text=bt, hovertemplate="%{text}<extra></extra>",
                                 marker=dict(symbol="diamond-open", size=10, color=INK, line=dict(width=1.6)),
                                 showlegend=False))
    cyc = R["sf"].aux.get("cyclone")
    if cyc:
        fig.add_trace(go.Scatter(x=[cyc[1]], y=[cyc[0]], mode="markers+text", text=["Cyclone (illustrative)"],
                                 textposition="top center", showlegend=False, hoverinfo="skip",
                                 marker=dict(symbol="circle-open", size=30, color="#fff", line=dict(width=3, color="#fff"))))
    for p in pins:
        fig.add_trace(go.Scatter(x=[p["lon"]], y=[p["lat"]], mode="markers+text", text=[p["label"]],
                                 textposition="top right", textfont=dict(size=14, color=INK), showlegend=False,
                                 hoverinfo="skip", marker=dict(symbol="star", size=17, color=p["color"],
                                                               line=dict(width=1.5, color="#fff"))))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=6, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#dfe9ec",
        font=dict(family=FONT, color=INK, size=12), hovermode="closest", clickmode="event+select", dragmode="pan",
        xaxis=dict(range=[x0, x1], ticksuffix="°E", showgrid=True, gridcolor="rgba(13,34,51,0.10)", zeroline=False),
        yaxis=dict(range=[y0, y1], ticksuffix="°N", showgrid=True, gridcolor="rgba(13,34,51,0.10)", zeroline=False,
                   scaleanchor="x", scaleratio=1, constrain="domain"))
    return fig


def _y(depth, stretched):
    return np.sqrt(depth) if stretched else np.asarray(depth, dtype=float)


def profile_figure(pins, depth_now, stretched, show_ref, extra_obs):
    fig = go.Figure()
    ys = _y(em.DEPTHS, stretched)
    xs_all = []
    for p in pins:
        lo, hi = p["mu"] - 2 * p["sg"], p["mu"] + 2 * p["sg"]
        xs_all += [lo, hi]
        fig.add_trace(go.Scatter(x=hi, y=ys, mode="lines", line=dict(width=0, shape="spline"), showlegend=False,
                                 hoverinfo="skip"))
        band = "rgba(%d,%d,%d,0.20)" % _hex(p["color"])
        fig.add_trace(go.Scatter(x=lo, y=ys, mode="lines", line=dict(width=0, shape="spline"), fill="tonextx",
                                 fillcolor=band, name=f"{p['label']} ±2σ", hoverinfo="skip"))
        if show_ref:
            fig.add_trace(go.Scatter(x=p["ref"], y=ys, mode="lines", line=dict(width=1.4, dash="dash", color=p["color"]),
                                     opacity=0.7, name=f"{p['label']} synthetic reference", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=p["mu"], y=ys, mode="lines+markers", line=dict(width=2.6, color=p["color"], shape="spline"),
                                 marker=dict(size=6, color=p["color"]), name=f"{p['label']} reconstruction",
                                 customdata=np.stack([em.DEPTHS, 2 * p["sg"]], axis=-1),
                                 hovertemplate="%{customdata[0]:.0f} m: %{x:.2f} °C (±%{customdata[1]:.2f})<extra></extra>"))
        if np.isfinite(p["d20"]):
            fig.add_trace(go.Scatter(x=[20], y=[_y(p["d20"], stretched)], mode="markers+text", showlegend=False,
                                     text=[f"  D20 {p['d20']:.0f} m"], textposition="middle right",
                                     textfont=dict(size=11, color=p["color"]), hoverinfo="skip",
                                     marker=dict(symbol="line-ew", size=22, line=dict(width=3, color=p["color"]))))
        if p["inv"] > 0.3:
            fig.add_annotation(x=p["mu"][list(em.DEPTHS).index(p["inv_z"])], y=_y(p["inv_z"], stretched),
                               text=f"warm layer +{p['inv']:.1f} °C", showarrow=True, arrowhead=2, ax=50, ay=-30,
                               font=dict(size=11, color=p["color"]), arrowcolor=p["color"])
    for label, (z, v, name) in extra_obs.items():
        fig.add_trace(go.Scatter(x=v, y=_y(z, stretched), mode="markers", name=f"{label} {name}",
                                 marker=dict(symbol="diamond", size=9, color="#fff", line=dict(width=2, color=INK))))
        xs_all.append(np.asarray(v))
    xall = np.concatenate([np.asarray(a).ravel() for a in xs_all])
    fig.add_shape(type="line", xref="paper", x0=0, x1=1, y0=_y(depth_now, stretched), y1=_y(depth_now, stretched),
                  line=dict(color=MUTED, width=1, dash="dot"))
    ticks = [0, 20, 50, 100, 200, 300, 500, 700, 1000] if stretched else [0, 200, 400, 600, 800, 1000]
    fig.update_layout(
        height=480, margin=dict(l=8, r=8, t=6, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.55)",
        font=dict(family=FONT, color=INK, size=12), hovermode="closest",
        xaxis=dict(title="Temperature (°C)", range=[float(np.nanmin(xall)) - 0.8, float(np.nanmax(xall)) + 2.5],
                   gridcolor="rgba(13,34,51,0.10)", zeroline=False),
        yaxis=dict(title="Depth (m)", tickvals=[float(_y(t, stretched)) for t in ticks], ticktext=[str(t) for t in ticks],
                   range=[float(_y(1000, stretched)) * 1.02, -float(_y(1000, stretched)) * 0.02],
                   gridcolor="rgba(13,34,51,0.10)", zeroline=False),
        legend=dict(orientation="h", y=-0.16, x=0, font=dict(size=11)))
    return fig


def confidence_figure(pins):
    fig = go.Figure()
    for p in pins:
        fig.add_trace(go.Bar(x=[str(int(z)) for z in em.DEPTHS], y=2 * p["sg"], name=p["label"], marker_color=p["color"],
                             opacity=0.9))
    fig.update_layout(height=200, barmode="group", margin=dict(l=8, r=8, t=6, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(255,255,255,0.55)", font=dict(family=FONT, color=INK, size=12),
                      xaxis=dict(title="Depth (m)"), yaxis=dict(title="±2σ (°C)", gridcolor="rgba(13,34,51,0.10)"),
                      legend=dict(orientation="h", y=1.18, x=0), bargap=0.25)
    return fig


def section_figure(R, axis, fixed, var, pin_a):
    if axis == "Along a latitude":
        i = int(np.clip(round((fixed - float(em.LAT[0])) / 0.25), 0, em.H - 1))
        M, S, coord, land = R["mu"][:, i, :], R["sigma"][:, i, :], em.LON, em.LAND[i, :]
        xlab, mark = "Longitude (°E)", pin_a["lon"]
    else:
        j = int(np.clip(round((fixed - float(em.LON[0])) / 0.25), 0, em.W - 1))
        M, S, coord, land = R["mu"][:, :, j], R["sigma"][:, :, j], em.LAT, em.LAND[:, j]
        xlab, mark = "Latitude (°N)", pin_a["lat"]
    z = M if var == "Temperature" else 2 * S
    scale = THERMAL if var == "Temperature" else PLUM
    d20 = em.d20_from_profiles(M)
    y = np.sqrt(em.DEPTHS)
    fig = go.Figure()
    fig.add_trace(go.Contour(x=coord, y=y, z=z, colorscale=scale, ncontours=22, connectgaps=False,
                             contours=dict(coloring="heatmap", showlines=True),
                             line=dict(width=0.4, color="rgba(255,255,255,0.5)"),
                             colorbar=dict(title=dict(text="°C" if var == "Temperature" else "±2σ °C", side="right"),
                                           thickness=10, outlinewidth=0),
                             hovertemplate="%{x:.2f}: %{z:.2f} °C<extra></extra>"))
    fig.add_trace(go.Scatter(x=coord, y=np.where(np.isfinite(d20), np.sqrt(np.nan_to_num(d20)), np.nan), mode="lines",
                             line=dict(color="#ffffff", width=2.2), name="20 °C isotherm", hoverinfo="skip"))
    ymax = float(np.sqrt(1000))
    fig.add_shape(type="line", x0=mark, x1=mark, y0=0, y1=ymax, line=dict(color=INK, width=1, dash="dot"))
    ticks = [0, 20, 50, 100, 200, 300, 500, 700, 1000]
    fig.update_layout(height=470, margin=dict(l=8, r=8, t=6, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(255,255,255,0.55)", font=dict(family=FONT, color=INK, size=12),
                      xaxis=dict(title=xlab), showlegend=True, legend=dict(orientation="h", y=1.08, x=0),
                      yaxis=dict(title="Depth (m)", tickvals=[float(np.sqrt(t)) for t in ticks],
                                 ticktext=[str(t) for t in ticks], range=[ymax * 1.01, 0]))
    return fig


# ======================================================================================
# Validation helpers
# ======================================================================================
def parse_buoy_csv(file):
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"depth": "depth_m", "temp": "temp_c", "temperature": "temp_c", "latitude": "lat",
                            "longitude": "lon", "time": "date"})
    need = ["date", "lat", "lon", "depth_m", "temp_c"]
    if not set(need).issubset(df.columns):
        return None, f"Missing columns: {', '.join(sorted(set(need) - set(df.columns)))}"
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for c in need[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=need)
    df = df[df["depth_m"].between(0, 1000) & df["lat"].between(5, 30) & df["lon"].between(45, 105)
            & df["temp_c"].between(0, 40)]
    if df.empty:
        return None, "No valid rows inside 5–30°N, 45–105°E, 0–1000 m."
    return df.head(20000).reset_index(drop=True), ""


def collect_pairs(engine):
    rows = []
    if ss.buoy_df is None:
        dates = [dt.date(2020, 2, 15), dt.date(2020, 5, 19), dt.date(2020, 8, 15), dt.date(2020, 11, 15),
                 dt.date(2021, 2, 15), dt.date(2021, 5, 15), dt.date(2021, 8, 15), dt.date(2021, 12, 5)]
        for d in dates:
            R = get_recon(d, "auto", engine)
            for name, la, lo in em.DEMO_BUOYS:
                i, j = em.nearest_ocean_index(la, lo)
                z, v = em.synthetic_buoy_profile(R["ref"]["T"], la, lo, d)
                rows.append(pd.DataFrame(dict(depth=z, obs=v, pred=np.interp(z, em.DEPTHS, R["mu"][:, i, j]),
                                              sigma=np.interp(z, em.DEPTHS, R["sigma"][:, i, j]))))
    else:
        df = ss.buoy_df
        for d in sorted(df["date"].dt.date.unique())[:40]:
            R = get_recon(d, "auto", engine)
            sub = df[df["date"] == pd.Timestamp(d)]
            for (la, lo), g in sub.groupby(["lat", "lon"]):
                i, j = em.nearest_ocean_index(la, lo)
                z = g["depth_m"].values.astype(float)
                rows.append(pd.DataFrame(dict(depth=z, obs=g["temp_c"].values,
                                              pred=np.interp(z, em.DEPTHS, R["mu"][:, i, j]),
                                              sigma=np.interp(z, em.DEPTHS, R["sigma"][:, i, j]))))
    return pd.concat(rows, ignore_index=True)


def metrics_by_depth(df):
    bins = [-1, 25, 75, 150, 300, 600, 1001]
    labels = ["0–25 m", "25–75 m", "75–150 m", "150–300 m", "300–600 m", "600–1000 m"]
    d = df.copy()
    d["bin"] = pd.cut(d["depth"], bins=bins, labels=labels)
    d["err"] = d["pred"] - d["obs"]
    d["abs_err"] = d["err"].abs()
    d["sq_err"] = d["err"] ** 2
    d["two_sigma"] = 2 * d["sigma"]
    d["cover"] = (d["abs_err"] <= d["two_sigma"]).astype(float)
    t = d.groupby("bin", observed=True).agg(n=("err", "size"), bias=("err", "mean"), mae=("abs_err", "mean"),
                                            mse=("sq_err", "mean"), two_sigma=("two_sigma", "mean"),
                                            coverage=("cover", "mean")).reset_index()
    t["rmse"] = np.sqrt(t["mse"])
    return t.drop(columns="mse")


def reliability(R):
    m = em.OCEAN
    err = (R["mu"] - R["ref"]["T"])[:, m][:, ::3]
    s = R["sigma"][:, m][:, ::3]
    cover = np.mean(np.abs(err) <= 2 * s, axis=1)
    sf, ef = s.ravel(), err.ravel()
    edges = np.quantile(sf, np.linspace(0, 1, 11))
    idx = np.clip(np.digitize(sf, edges[1:-1]), 0, 9)
    bx = [float(sf[idx == b].mean()) for b in range(10)]
    by = [float(np.sqrt((ef[idx == b] ** 2).mean())) for b in range(10)]
    return cover, bx, by


# ======================================================================================
# Sidebar
# ======================================================================================
with st.sidebar:
    html('<div class="brand">SubTherm</div><div class="brand-sub">OceanEmbed prototype, team Echelon</div>')
    st.selectbox("Scenario", list(em.SCENARIOS), key="scenario", on_change=on_scenario,
                 format_func=lambda k: em.SCENARIOS[k].label)
    st.date_input("Date", key="date", min_value=dt.date(2010, 1, 1), max_value=dt.date(2022, 12, 31))
    st.selectbox("Map layer", LAYERS, key="layer")
    st.select_slider("Depth", options=DEPTH_INTS, key="depth", format_func=lambda z: f"{z} m",
                     disabled=ss.layer not in DEPTH_LAYERS)
    st.selectbox("Region", list(REGIONS), key="region")
    st.divider()
    st.radio("A click on the map moves point", ["A", "B"], key="target", horizontal=True)
    st.checkbox("Compare with point B", key="compare")
    st.selectbox("Jump the active point to", list(JUMPS), index=None, key="jump", on_change=on_jump,
                 placeholder="Choose a place or mooring")
    with st.expander("Exact coordinates"):
        c1, c2 = st.columns(2)
        c1.number_input("Lat A", min_value=5.0, max_value=30.0, step=0.25, format="%.2f", key="latA")
        c2.number_input("Lon A", min_value=45.0, max_value=105.0, step=0.25, format="%.2f", key="lonA")
        c1.number_input("Lat B", min_value=5.0, max_value=30.0, step=0.25, format="%.2f", key="latB")
        c2.number_input("Lon B", min_value=45.0, max_value=105.0, step=0.25, format="%.2f", key="lonB")
    st.divider()
    st.checkbox("Show moorings on the map", key="show_buoys")
    st.checkbox("Stretch the upper ocean in profiles", key="stretched")
    st.checkbox("Show synthetic reference line", key="show_ref")
    if TORCH_OK:
        st.radio("Engine", ["emulator", "hscan"], key="engine",
                 format_func=lambda k: "Physics-based demo emulator" if k == "emulator" else "H-SCAN (PyTorch)")
    else:
        ss.engine = "emulator"
        st.caption("H-SCAN engine needs PyTorch: `pip install torch`")

# ======================================================================================
# Compute the state for this run
# ======================================================================================
engine = ss.engine if TORCH_OK else "emulator"
if engine == "hscan":
    with st.spinner("Running H-SCAN on the daily surface tensor…"):
        R = get_recon(ss.date, ss.scenario, "hscan")
else:
    R = get_recon(ss.date, ss.scenario, "emulator")

k_depth = DEPTH_INTS.index(int(ss.depth))
pin_a = pin_info(R, "A", PIN_A, ss.latA, ss.lonA)
pins = [pin_a]
if ss.compare:
    pins.append(pin_info(R, "B", PIN_B, ss.latB, ss.lonB))
scn = em.SCENARIOS[ss.scenario]

# ======================================================================================
# Hero
# ======================================================================================
stops = ", ".join(f"{scale_color(THERMAL, (t - 4) / 29)} {100 * np.sqrt(z / 1000):.1f}%"
                  for z, t in zip(em.DEPTHS, pin_a["mu"]))
html(f"""
<div class="hero">
<div>
<h1>SubTherm</h1>
<p>Daily subsurface temperature of the North Indian Ocean, 0 to 1000 m, rebuilt from what satellites see at the surface, with an error bar at every depth.</p>
<div class="who">Smart India Hackathon 2026, problem statement 26066 (OceanEmbed), INCOIS. Team Echelon.</div>
</div>
<div>
<div class="ribbon-cap">Water column at point A ({fmt_ll(pin_a['lat'], pin_a['lon'])}), surface on the left</div>
<div class="ribbon" style="background: linear-gradient(90deg, {stops});"></div>
<div class="ribbon-ticks"><span>0 m</span><span>50 m</span><span>200 m</span><span>500 m</span><span>1000 m</span></div>
</div>
</div>
""")

if engine == "hscan" and ss.hscan_source == "untrained":
    html('<div class="notice">H-SCAN is running with <b>untrained weights</b>: the maps below are noise until you load a checkpoint in the H-SCAN lab tab.</div>')
elif engine == "hscan":
    html('<div class="notice ok">H-SCAN checkpoint loaded. Surface inputs are still synthetic until you connect real satellite data.</div>')
else:
    html('<div class="notice">Demo mode: surface fields and reconstructions are <b>synthetic</b>.</div>')

tab_console, tab_section, tab_val, tab_lab, tab_about = st.tabs(
    ["Console", "Vertical section", "Validation", "H-SCAN lab", "About this prototype"])

# ======================================================================================
# Console
# ======================================================================================
with tab_console:
    inv_txt = f"+{pin_a['inv']:.1f} <small>°C at {pin_a['inv_z']:.0f} m</small>" if pin_a["inv"] > 0.3 else "none <small>detected</small>"
    html(f"""
<div class="readout">
<div><div class="k">Point A</div><div class="v">{fmt_ll(pin_a['lat'], pin_a['lon'])}</div></div>
<div><div class="k">Surface temperature</div><div class="v">{pin_a['mu'][0]:.1f} <small>°C</small></div></div>
<div><div class="k">Thermocline (20 °C)</div><div class="v">{pin_a['d20']:.0f} <small>m</small></div></div>
<div><div class="k">Confidence at 1000 m</div><div class="v">±{2 * pin_a['sg'][-1]:.1f} <small>°C</small></div></div>
<div><div class="k">Warm layer under a cool surface</div><div class="v">{inv_txt}</div></div>
</div>
""")
    left, right = st.columns([1.45, 1])
    with left:
        st.caption(f"{scn.label} on {ss.date:%d %b %Y}. {scn.blurb} Click the ocean to move point {ss.target}.")
        fig = build_map(R, ss.layer, k_depth, ss.region, pins, ss.show_buoys, ss.date)
        st.plotly_chart(fig, key="map", on_select=on_map_select, selection_mode="points",
                        config={"displaylogo": False, "scrollZoom": True,
                                "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]})
    with right:
        extra = {}
        for p in pins:
            o = obs_near(R, ss.date, p["lat"], p["lon"])
            if o:
                extra[p["label"]] = o
        st.plotly_chart(profile_figure(pins, ss.depth if ss.layer in DEPTH_LAYERS else 0, ss.stretched,
                                       ss.show_ref, extra), key="profile")
        if extra:
            st.caption("Diamonds are mooring readings. #Demo mode they are synthetic.")
        if any(2 * p["sg"][-1] > 1.2 for p in pins):
            st.caption("Prediction confidence is low near 1000 m because the surface signal fades with depth. "
                       "That is what the shaded band shows.")
    lo_col, hi_col = st.columns([1.45, 1])
    with hi_col:
        st.plotly_chart(confidence_figure(pins), key="conf")
    with lo_col:
        prof = pd.DataFrame({"depth_m": em.DEPTHS, "temp_c": pin_a["mu"], "sigma_c": pin_a["sg"]})
        st.markdown("**Point A values**")
        st.dataframe(prof.round(3).T, hide_index=False)
        st.download_button("Download point A profile (CSV)", prof.to_csv(index=False), "profile_A.csv", "text/csv")

# ======================================================================================
# Section
# ======================================================================================
with tab_section:
    c1, c2, c3 = st.columns([1, 1, 1])
    axis = c1.radio("Cut", ["Along a latitude", "Along a longitude"], horizontal=True)
    var = c2.radio("Show", ["Temperature", "Uncertainty (±2σ)"], horizontal=True)
    if axis == "Along a latitude":
        fixed = c3.slider("Latitude (°N)", 5.0, 30.0, float(pin_a["lat"]), 0.25)
    else:
        fixed = c3.slider("Longitude (°E)", 45.0, 105.0, float(pin_a["lon"]), 0.25)
    st.plotly_chart(section_figure(R, axis, fixed, "Temperature" if var == "Temperature" else "sigma", pin_a),
                    key="section")
    st.caption("The white line is the 20 °C isotherm. Watch it rise near upwelling coasts and sink under warm eddies. "
               "The dotted line marks point A.")

# ======================================================================================
# Validation
# ======================================================================================
with tab_val:
    st.markdown("### Independent validation")
    html('<div class="notice ok">GLORYS already assimilates ARGO profiles, so scoring a GLORYS-trained model on ARGO is circular. '
         'Score against INCOIS OMNI or RAMA mooring records that were held out of training instead.</div>')
    up = st.file_uploader("Mooring temperature CSV (columns: date, lat, lon, depth_m, temp_c)", type=["csv"])
    tmpl = "date,lat,lon,depth_m,temp_c\n2021-12-05,18.0,89.0,10,24.3\n2021-12-05,18.0,89.0,50,25.2\n"
    cA, cB = st.columns([1, 1])
    cA.download_button("Download CSV template", tmpl, "mooring_template.csv", "text/csv")
    if up is not None and ss.get("upname") != up.name:
        parsed, msg = parse_buoy_csv(up)
        if parsed is None:
            st.error(msg)
        else:
            ss.buoy_df, ss.upname, ss.validation = parsed, up.name, None
            st.success(f"Loaded {len(parsed):,} readings.")
    if ss.buoy_df is not None and cB.button("Go back to synthetic moorings"):
        ss.buoy_df, ss.upname, ss.validation = None, None, None
        st.rerun()
    st.caption("Source: " + ("your uploaded moorings" if ss.buoy_df is not None else
                              "6 synthetic moorings over 8 dates in 2020 to 2021 (test block). Numbers only demonstrate the pipeline."))
    if st.button("Run validation"):
        with st.spinner("Co-locating moorings with the reconstruction…"):
            pairs = collect_pairs(engine)
            ss.validation = dict(pairs=pairs, table=metrics_by_depth(pairs), engine=engine)
    V = ss.validation
    if V:
        p, t = V["pairs"], V["table"]
        rmse_all = float(np.sqrt(((p["pred"] - p["obs"]) ** 2).mean()))
        cov = float((np.abs(p["pred"] - p["obs"]) <= 2 * p["sigma"]).mean())
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE, all depths (°C)", f"{rmse_all:.2f}")
        m2.metric("MAE (°C)", f"{float((p['pred'] - p['obs']).abs().mean()):.2f}")
        m3.metric("Readings inside ±2σ", f"{100 * cov:.0f}%", help="A calibrated model lands near 95%.")
        g1, g2 = st.columns(2)
        f1 = go.Figure()
        f1.add_trace(go.Bar(x=t["bin"].astype(str), y=t["rmse"], name="RMSE", marker_color=SEA))
        f1.add_trace(go.Scatter(x=t["bin"].astype(str), y=t["two_sigma"], name="Mean ±2σ", mode="lines+markers",
                                line=dict(color="#9c4a8f", width=2.5)))
        f1.update_layout(height=340, title="Error and stated uncertainty by depth", margin=dict(l=8, r=8, t=40, b=8),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.55)",
                         font=dict(family=FONT, color=INK, size=12), yaxis=dict(title="°C"),
                         legend=dict(orientation="h", y=-0.2))
        g1.plotly_chart(f1, key="val_bar")
        sm = p.sample(min(len(p), 1500), random_state=0)
        f2 = go.Figure(go.Scatter(x=sm["obs"], y=sm["pred"], mode="markers",
                                  marker=dict(size=6, color=sm["depth"], colorscale=DEEP, showscale=True,
                                              colorbar=dict(title="m", thickness=10), opacity=0.8)))
        lim = [float(min(sm["obs"].min(), sm["pred"].min())), float(max(sm["obs"].max(), sm["pred"].max()))]
        f2.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=INK, dash="dot", width=1), showlegend=False))
        f2.update_layout(height=340, title="Reconstruction against mooring", margin=dict(l=8, r=8, t=40, b=8),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.55)",
                         font=dict(family=FONT, color=INK, size=12), xaxis=dict(title="Mooring (°C)"),
                         yaxis=dict(title="Reconstruction (°C)"), showlegend=False)
        g2.plotly_chart(f2, key="val_scatter")
        st.dataframe(t.round(3), hide_index=True)

    st.markdown("### Does high uncertainty mean high error?")
    st.caption("Checked on the selected day against the synthetic reference (all ocean pixels). "
               "Points on the dotted line mean the stated σ matches the real error.")
    cover, bx, by = reliability(R)
    r1, r2 = st.columns(2)
    fr = go.Figure(go.Scatter(x=bx, y=by, mode="lines+markers", line=dict(color="#9c4a8f", width=2.5), name="Model"))
    top = max(max(bx), max(by)) * 1.05
    fr.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color=INK, dash="dot", width=1), name="Perfect"))
    fr.update_layout(height=320, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
                     plot_bgcolor="rgba(255,255,255,0.55)", font=dict(family=FONT, color=INK, size=12),
                     xaxis=dict(title="Predicted σ (°C), binned"), yaxis=dict(title="Actual RMSE (°C)"),
                     legend=dict(orientation="h", y=-0.22))
    r1.plotly_chart(fr, key="rel")
    fc = go.Figure(go.Scatter(x=cover * 100, y=np.sqrt(em.DEPTHS), mode="lines+markers", line=dict(color=SEA, width=2.5)))
    fc.add_vline(x=95.4, line=dict(color=INK, dash="dot", width=1))
    ticks = [0, 20, 50, 100, 200, 500, 1000]
    fc.update_layout(height=320, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
                     plot_bgcolor="rgba(255,255,255,0.55)", font=dict(family=FONT, color=INK, size=12),
                     xaxis=dict(title="Share of pixels inside ±2σ (%)"),
                     yaxis=dict(title="Depth (m)", tickvals=[float(np.sqrt(t)) for t in ticks],
                                ticktext=[str(t) for t in ticks], range=[31.9, -0.3]))
    r2.plotly_chart(fc, key="cov")
    d_mu, d_ref = em.d20_from_profiles(R["mu"]), em.d20_from_profiles(R["ref"]["T"])
    dd = np.abs(d_mu - d_ref)[em.OCEAN]
    uo = np.abs(em.uohc_gj_m2(R["mu"]) - em.uohc_gj_m2(R["ref"]["T"]))[em.OCEAN]
    s1, s2 = st.columns(2)
    s1.metric("Thermocline depth error, mean (m)", f"{np.nanmean(dd):.1f}")
    s2.metric("Heat content 0–700 m error, mean (GJ/m²)", f"{np.nanmean(uo):.2f}")

# ======================================================================================
# H-SCAN lab
# ======================================================================================
TRAIN_SNIPPET = '''# Not trained yet, so the output is noise.
'''

with tab_lab:
    st.markdown("### H-SCAN, the network behind the console")
    a, b = st.columns([1, 1])
    with a:
        html("""
<div class="flow">
<div class="node"><b>Daily surface tensor, 9 × 101 × 241</b><span>SST, SSS, SSH, currents U and V, winds, Coriolis, bathymetry</span></div>
<div class="node att"><b>CBAM on the inputs</b><span>Learns which variable matters where, for example salinity in the Bay of Bengal, wind off Oman</span></div>
<div class="node"><b>Convolution stem</b><span>Lifts 9 channels to a shared feature space</span></div>
<div class="node att"><b>Dilated ResNet, 8 blocks (dilation 1, 2, 4, 8, 1, 2, 4, 8)</b><span>Sees 25 km upwelling belts and 500 km Rossby waves without losing resolution; CBAM in every block</span></div>
<div class="split">
<div class="node mean"><b>Mean head</b><span>Temperature at 15 depths</span></div>
<div class="node var"><b>Variance head</b><span>Log-variance at 15 depths, so the model can say "I am unsure here"</span></div>
</div>
<div class="node"><b>Loss</b><span>Gaussian NLL (weighted at 50 to 150 m near river fronts) + static-stability penalty + thermocline-depth term</span></div>
</div>
""")
    with b:
        st.markdown("**Blocked time split, no shuffling**")
        html("""
<div class="timeline">
<div style="flex: 9; background: #0b5d7a;">Train 2010 to 2018</div>
<div style="flex: 1.3; background: #d9822b;">Val 2019</div>
<div style="flex: 3; background: #9c4a8f;">Test 2020 to 2022</div>
</div>
""")
        st.caption("Neighbouring days are almost identical in the ocean, so a random split lets the network memorise its test set.")
        st.code(TRAIN_SNIPPET, language="python")

    st.divider()
    if not TORCH_OK:
        st.info("PyTorch is not installed, so the network cannot run here. Install it with `pip install torch` and restart the app. "
                f"(Import message: {TORCH_ERR})")
    else:
        st.markdown("#### Build and test the network")
        cc1, cc2, cc3 = st.columns([1, 1, 1])
        base = cc1.select_slider("Base channels", [16, 24, 32, 48, 64], value=32)
        pattern = cc2.selectbox("Dilation pattern", ["1-2-4-8-1-2-4-8", "1-2-4-8", "1-2-4-8-16-1-2-4"])
        if cc3.button("Build untrained model"):
            dil = tuple(int(x) for x in pattern.split("-"))
            ss.hscan_model = hscan.HSCAN(base_channels=int(base), dilations=dil).eval()
            ss.hscan_stats, ss.hscan_source = {}, "untrained"
            ss.hscan_token += 1
            ss.hscan_cache = {}
        model = ss.hscan_model
        if model is not None:
            n = hscan.count_parameters(model)
            k1, k2, k3 = st.columns(3)
            k1.metric("Parameters", f"{n / 1e6:.2f} M")
            k2.metric("Size in memory", f"{n * 4 / 1e6:.1f} MB")
            k3.metric("Weights", "trained checkpoint" if ss.hscan_source == "checkpoint" else "untrained (random)")
            if st.button("Run one forward pass on the selected day"):
                x_raw = em.stack_inputs(R["sf"])
                t0 = time.time()
                mu, sg = hscan.predict_numpy(model, x_raw, (ss.hscan_stats or {}).get("temp_mean"),
                                             (ss.hscan_stats or {}).get("temp_std"))
                el = time.time() - t0
                st.success(f"Output tensors: mean {tuple(mu.shape)}, sigma {tuple(sg.shape)} in {el:.2f} s on CPU.")
                att = model.attention_maps()
                w = att["channel"][0, :, 0, 0].numpy()
                fw = go.Figure(go.Bar(x=list(hscan.INPUT_CHANNELS), y=w, marker_color=SEA))
                fw.update_layout(height=260, title="Input channel attention (last forward pass)",
                                 margin=dict(l=8, r=8, t=40, b=8), paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(255,255,255,0.55)", font=dict(family=FONT, color=INK, size=12),
                                 yaxis=dict(range=[0, 1]))
                st.plotly_chart(fw, key="lab_att")
                st.caption("With random weights every channel sits near 0.5. After training, salinity should rise in the Bay of Bengal "
                           "and wind near the upwelling coasts. That is your evidence that the attention works.")
                lm = np.where(em.OCEAN, sg[k_depth], np.nan)
                fs = go.Figure(go.Heatmap(x=em.LON, y=em.LAT, z=lm, colorscale=PLUM, zsmooth="best",
                                          colorbar=dict(thickness=10, title=dict(text="σ °C", side="right"))))
                fs.update_layout(height=330, title=f"Predicted σ at {DEPTH_INTS[k_depth]} m (untrained: meaningless)",
                                 margin=dict(l=8, r=8, t=40, b=8), paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(255,255,255,0.55)", font=dict(family=FONT, color=INK, size=12),
                                 yaxis=dict(scaleanchor="x", scaleratio=1))
                st.plotly_chart(fs, key="lab_sig")
        st.markdown("#### Load a trained checkpoint")
        ck = st.file_uploader("Checkpoint saved with hscan.save_checkpoint (.pt)", type=["pt", "pth"], key="ckpt")
        if ck is not None and ss.get("ckname") != ck.name:
            try:
                m, missing, unexpected, stats = hscan.load_checkpoint_bytes(ck.getvalue())
                ss.hscan_model, ss.hscan_stats, ss.hscan_source, ss.ckname = m.eval(), stats, "checkpoint", ck.name
                ss.hscan_token += 1
                ss.hscan_cache = {}
                st.success(f"Loaded {ck.name}. Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}.")
            except Exception as exc:
                st.error(f"Could not load the checkpoint: {exc}")
        st.button("Use H-SCAN in the console", on_click=use_hscan)
        with st.expander("Loss function source"):
            st.code(inspect.getsource(hscan.combined_loss), language="python")

# ======================================================================================
# About
# ======================================================================================
with tab_about:
    st.markdown("### What is real here and what is not")
    st.table(pd.DataFrame({
        "Part": ["Console, map, profiles, sections, download", "Surface fields (SST, SSS, SSH, currents, winds)",
                 "Reconstructed temperature and σ (emulator engine)", "Cyclone Amphan replay and moorings",
                 "H-SCAN network and losses (hscan.py)", "Validation tab metrics"],
        "Status": ["Working application code", "Synthetic, physics-inspired", "Synthetic, error sized to match σ",
                   "Illustrative track and positions", "Real PyTorch code, untrained",
                   "Real code; numbers are synthetic until we upload moorings"]}))
    st.markdown("""The model will be trained on real satellite data and validated against held-out moorings. The H-SCAN lab tab lets you inspect the network architecture, attention maps, and run a forward pass on the selected day.

""")
