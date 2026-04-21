"""
dashboard/app.py
Streamlit predictive maintenance monitoring dashboard.

Run: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import random
from datetime import datetime, timedelta


# ── Page config ───────────────────────────────────────────────────

st.set_page_config(
    page_title="PredMaint — IoT Monitoring",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .main .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 100%);
        border: 1px solid rgba(99, 179, 237, 0.2);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.5rem;
    }
    .kpi-label {
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #718096;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.9rem;
        font-weight: 600;
        color: #63b3ed;
        line-height: 1;
    }
    .kpi-sub {
        font-size: 0.72rem;
        color: #4a5568;
        margin-top: 0.3rem;
    }
    .kpi-critical .kpi-value { color: #fc8181; }
    .kpi-warning  .kpi-value { color: #f6e05e; }
    .kpi-ok       .kpi-value { color: #68d391; }

    /* Device health card */
    .device-card {
        background: #1a1d2e;
        border-left: 4px solid #63b3ed;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
    }
    .device-card.critical { border-left-color: #fc8181; }
    .device-card.warning  { border-left-color: #f6e05e; }
    .device-card.healthy  { border-left-color: #68d391; }

    /* Alert badge */
    .alert-badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .badge-critical { background: rgba(252,129,129,0.15); color: #fc8181; border: 1px solid rgba(252,129,129,0.3); }
    .badge-warning  { background: rgba(246,224,94,0.15);  color: #f6e05e; border: 1px solid rgba(246,224,94,0.3);  }
    .badge-normal   { background: rgba(104,211,145,0.1);  color: #68d391; border: 1px solid rgba(104,211,145,0.3); }

    /* Section headers */
    .section-header {
        font-size: 0.68rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #4a5568;
        border-bottom: 1px solid #2d3748;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
    /* Streamlit overrides */
    .stSelectbox label, .stSlider label { font-size: 0.75rem !important; }
    div[data-testid="metric-container"] {
        background: #1a1d2e;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Mock data generator (for standalone demo) ─────────────────────

DEVICES = {
    "motor_01":           {"type": "Motor",           "domain": "Industrial", "health": 0.88},
    "motor_02":           {"type": "Motor",           "domain": "Industrial", "health": 0.62},
    "pump_01":            {"type": "Pump",            "domain": "Industrial", "health": 0.45},
    "pump_02":            {"type": "Pump",            "domain": "Industrial", "health": 0.91},
    "compressor_01":      {"type": "Compressor",      "domain": "Industrial", "health": 0.33},
    "ac_01":              {"type": "Air Conditioner", "domain": "Smart Home", "health": 0.79},
    "ac_02":              {"type": "Air Conditioner", "domain": "Smart Home", "health": 0.28},
    "washing_machine_01": {"type": "Washing Machine", "domain": "Smart Home", "health": 0.85},
    "washing_machine_02": {"type": "Washing Machine", "domain": "Smart Home", "health": 0.55},
    "water_heater_01":    {"type": "Water Heater",    "domain": "Smart Home", "health": 0.72},
}

FAULT_NAMES = [
    "normal", "bearing_wear", "imbalance", "overheating",
    "cavitation", "valve_leak", "refrigerant_leak",
    "compressor_fail", "drum_imbalance", "pump_fail",
]


def _health_to_urgency(h: float) -> str:
    if h < 0.35: return "critical"
    if h < 0.6:  return "warning"
    if h < 0.8:  return "moderate"
    return "healthy"


def _health_color(h: float) -> str:
    if h < 0.35: return "#fc8181"
    if h < 0.6:  return "#f6e05e"
    if h < 0.8:  return "#63b3ed"
    return "#68d391"


def generate_live_reading(device_id: str) -> dict:
    dev = DEVICES[device_id]
    h   = dev["health"]
    deg = 1.0 - h
    r   = random.gauss

    if dev["domain"] == "Industrial":
        return {
            "vibration":   round(0.5 + deg * 2.5 + r(0, 0.05), 3),
            "temperature": round(65  + deg * 20  + r(0, 1.0),   1),
            "current":     round(10  + deg * 4   + r(0, 0.2),   2),
            "pressure":    round(max(0.1, 1.0 - deg * 0.3 + r(0, 0.02)), 3),
            "rpm":         round(1500 - deg * 200 + r(0, 15), 0),
            "humidity":    round(50 + r(0, 3), 1),
            "power_w":     round((10 + deg * 4) * 220 * 0.85, 1),
        }
    else:
        pw = 1200 + deg * 400 + r(0, 30)
        return {
            "vibration":   round(0.1 + deg * 0.6 + r(0, 0.02), 3),
            "temperature": round(22  + deg * 10  + r(0, 0.5),   1),
            "current":     round(pw / 220,   2),
            "pressure":    round(1.0 + r(0, 0.05), 3),
            "rpm":         round(1200 + r(0, 30), 0) if dev["type"] == "Washing Machine" else 0,
            "humidity":    round(55 + r(0, 4), 1),
            "power_w":     round(max(0, pw), 1),
        }


def generate_timeseries(device_id: str, n: int = 120) -> pd.DataFrame:
    dev   = DEVICES[device_id]
    h_seq = np.linspace(1.0, dev["health"], n)
    rows  = []
    now   = datetime.now()
    for i, h in enumerate(h_seq):
        DEVICES[device_id]["health"] = h
        r = generate_live_reading(device_id)
        r["timestamp"] = (now - timedelta(minutes=n - i)).isoformat()
        rows.append(r)
    DEVICES[device_id]["health"] = dev["health"]
    return pd.DataFrame(rows)


def mock_rul(health: float) -> float:
    return round(health * 3200 + random.gauss(0, 50), 1)


def mock_anomaly_score(health: float) -> float:
    base = (1 - health) * 0.08
    return round(max(0, base + random.gauss(0, 0.005)), 5)


def mock_fault(health: float) -> str:
    if health > 0.75: return "normal"
    if health > 0.5:  return random.choice(["bearing_wear", "imbalance", "refrigerant_leak"])
    return random.choice(FAULT_NAMES[1:])


# ── Sidebar ───────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ PredMaint")
    st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["Overview", "Device Deep Dive", "Alert Center", "Model Performance"], label_visibility="collapsed")

    st.markdown('<div class="section-header" style="margin-top:1.5rem">Filters</div>', unsafe_allow_html=True)
    domain_filter = st.multiselect("Domain", ["Industrial", "Smart Home"], default=["Industrial", "Smart Home"])
    urgency_filter = st.multiselect("Urgency", ["critical","warning","moderate","healthy"], default=["critical","warning","moderate","healthy"])

    st.markdown('<div class="section-header" style="margin-top:1.5rem">Settings</div>', unsafe_allow_html=True)
    auto_refresh = st.toggle("Live refresh", value=True)
    refresh_sec  = st.slider("Interval (s)", 3, 30, 5)

    st.markdown("---")
    st.caption("PredMaint v1.0 · Portfolio project\nIndustrial + Smart Home ML monitoring")


# ── Auto-refresh ──────────────────────────────────────────────────
if auto_refresh:
    time.sleep(0.1)


# ═══════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════

if page == "Overview":
    st.markdown("## System Overview")
    st.markdown(f"*Last updated: {datetime.now().strftime('%H:%M:%S')}*")

    # ── KPI row ────────────────────────────────────────────────────
    statuses = {
        did: {
            "health":  DEVICES[did]["health"],
            "urgency": _health_to_urgency(DEVICES[did]["health"]),
            "anomaly": mock_anomaly_score(DEVICES[did]["health"]) > 0.04,
            "rul":     mock_rul(DEVICES[did]["health"]),
        }
        for did in DEVICES
        if DEVICES[did]["domain"] in domain_filter
    }

    n_critical = sum(1 for s in statuses.values() if s["urgency"] == "critical")
    n_warning  = sum(1 for s in statuses.values() if s["urgency"] == "warning")
    n_healthy  = sum(1 for s in statuses.values() if s["urgency"] == "healthy")
    n_anomaly  = sum(1 for s in statuses.values() if s["anomaly"])
    avg_health = np.mean([s["health"] for s in statuses.values()]) * 100 if statuses else 0
    avg_rul    = np.mean([s["rul"] for s in statuses.values()]) if statuses else 0

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        cls = "kpi-critical" if n_critical > 0 else "kpi-ok"
        st.markdown(f'<div class="kpi-card {cls}"><div class="kpi-label">Critical</div><div class="kpi-value">{n_critical}</div><div class="kpi-sub">devices</div></div>', unsafe_allow_html=True)
    with col2:
        cls = "kpi-warning" if n_warning > 0 else "kpi-ok"
        st.markdown(f'<div class="kpi-card {cls}"><div class="kpi-label">Warning</div><div class="kpi-value">{n_warning}</div><div class="kpi-sub">devices</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card kpi-ok"><div class="kpi-label">Healthy</div><div class="kpi-value">{n_healthy}</div><div class="kpi-sub">devices</div></div>', unsafe_allow_html=True)
    with col4:
        cls = "kpi-warning" if n_anomaly > 0 else "kpi-ok"
        st.markdown(f'<div class="kpi-card {cls}"><div class="kpi-label">Anomalies</div><div class="kpi-value">{n_anomaly}</div><div class="kpi-sub">detected now</div></div>', unsafe_allow_html=True)
    with col5:
        cls = "kpi-critical" if avg_health < 50 else ("kpi-warning" if avg_health < 70 else "kpi-ok")
        st.markdown(f'<div class="kpi-card {cls}"><div class="kpi-label">Avg Health</div><div class="kpi-value">{avg_health:.0f}%</div><div class="kpi-sub">fleet average</div></div>', unsafe_allow_html=True)
    with col6:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Avg RUL</div><div class="kpi-value">{avg_rul:.0f}</div><div class="kpi-sub">cycles remaining</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Fleet health grid ──────────────────────────────────────────
    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.markdown('<div class="section-header">Fleet health grid</div>', unsafe_allow_html=True)

        filtered_devices = {
            did: DEVICES[did] for did in DEVICES
            if DEVICES[did]["domain"] in domain_filter
            and _health_to_urgency(DEVICES[did]["health"]) in urgency_filter
        }

        for did, dev in filtered_devices.items():
            h     = dev["health"]
            urg   = _health_to_urgency(h)
            score = mock_anomaly_score(h)
            rul   = mock_rul(h)
            fault = mock_fault(h)
            col_a, col_b, col_c, col_d = st.columns([2.2, 1.2, 1.2, 1])
            with col_a:
                bar_color = _health_color(h)
                bar_w = int(h * 100)
                st.markdown(f"""
                <div style="margin-bottom:6px">
                  <div style="font-size:0.8rem;font-weight:500;color:#e2e8f0">{did}</div>
                  <div style="font-size:0.68rem;color:#718096">{dev['type']} · {dev['domain']}</div>
                  <div style="background:#2d3748;border-radius:4px;height:5px;margin-top:5px">
                    <div style="width:{bar_w}%;height:5px;border-radius:4px;background:{bar_color}"></div>
                  </div>
                </div>""", unsafe_allow_html=True)
            with col_b:
                st.markdown(f'<div style="font-family:monospace;font-size:0.85rem;color:{bar_color};padding-top:4px">{h*100:.0f}%</div>', unsafe_allow_html=True)
            with col_c:
                st.markdown(f'<div style="font-family:monospace;font-size:0.85rem;color:#a0aec0;padding-top:4px">{rul:.0f} cyc</div>', unsafe_allow_html=True)
            with col_d:
                badge_cls = f"badge-{'critical' if urg=='critical' else 'warning' if urg in ('warning','moderate') else 'normal'}"
                st.markdown(f'<span class="alert-badge {badge_cls}" style="margin-top:6px;display:inline-block">{urg}</span>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">Health distribution</div>', unsafe_allow_html=True)

        health_vals = [DEVICES[did]["health"] * 100 for did in filtered_devices]
        device_names = list(filtered_devices.keys())
        colors = [_health_color(DEVICES[did]["health"]) for did in filtered_devices]

        fig_bar = go.Figure(go.Bar(
            x=health_vals,
            y=[d.replace("_", " ") for d in device_names],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.0f}%" for v in health_vals],
            textposition="inside",
            textfont=dict(size=11, family="IBM Plex Mono"),
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0,100], showgrid=True, gridcolor="#2d3748", color="#718096"),
            yaxis=dict(showgrid=False, color="#a0aec0", tickfont=dict(size=11)),
            margin=dict(l=0, r=20, t=0, b=20),
            height=320,
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Domain pie
        st.markdown('<div class="section-header">Domain split</div>', unsafe_allow_html=True)
        ind_count = sum(1 for d in filtered_devices.values() if d["domain"] == "Industrial")
        sh_count  = len(filtered_devices) - ind_count
        fig_pie = go.Figure(go.Pie(
            labels=["Industrial", "Smart Home"],
            values=[ind_count, sh_count],
            hole=0.55,
            marker_colors=["#63b3ed", "#9f7aea"],
            textinfo="label+percent",
            textfont=dict(size=11),
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=180,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Anomaly heatmap ────────────────────────────────────────────
    st.markdown('<div class="section-header" style="margin-top:1rem">Anomaly score heatmap (last 2 hours)</div>', unsafe_allow_html=True)

    n_time = 24
    time_labels = [(datetime.now() - timedelta(minutes=i*5)).strftime("%H:%M") for i in range(n_time, 0, -1)]
    all_devices  = list(filtered_devices.keys())
    heatmap_data = []
    for did in all_devices:
        h     = DEVICES[did]["health"]
        row   = [max(0, (1-h)*0.08 + random.gauss(0,0.008) + random.gauss(0, 0.01*(i/n_time))) for i in range(n_time)]
        heatmap_data.append(row)

    fig_heat = go.Figure(go.Heatmap(
        z=heatmap_data,
        x=time_labels,
        y=[d.replace("_"," ") for d in all_devices],
        colorscale=[[0,"#1a1d2e"],[0.5,"#d69e2e"],[1,"#e53e3e"]],
        zmin=0, zmax=0.1,
        colorbar=dict(title="Score", tickfont=dict(color="#718096"), titlefont=dict(color="#718096")),
    ))
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#718096", tickfont=dict(size=10)),
        yaxis=dict(color="#a0aec0", tickfont=dict(size=11)),
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    if auto_refresh:
        time.sleep(refresh_sec)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════
# PAGE: DEVICE DEEP DIVE
# ═══════════════════════════════════════════════════════════════════

elif page == "Device Deep Dive":
    st.markdown("## Device Deep Dive")

    selected = st.selectbox("Select device", list(DEVICES.keys()))
    dev      = DEVICES[selected]
    h        = dev["health"]
    urgency  = _health_to_urgency(h)
    rul      = mock_rul(h)
    score    = mock_anomaly_score(h)
    fault    = mock_fault(h)
    reading  = generate_live_reading(selected)

    # ── Device header ──────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("Health Score", f"{h*100:.1f}%",    _health_color(h)),
        ("RUL (cycles)", f"{rul:.0f}",        "#63b3ed"),
        ("Anomaly Score", f"{score:.5f}",     "#fc8181" if score > 0.04 else "#68d391"),
        ("Fault Status",  fault,              "#fc8181" if fault != "normal" else "#68d391"),
    ]
    for col, (label, value, color) in zip([col1,col2,col3,col4], metrics):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="color:{color};font-size:1.5rem">{value}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Time series charts ─────────────────────────────────────────
    st.markdown('<div class="section-header">Sensor time series (last 2 hours)</div>', unsafe_allow_html=True)
    df_ts = generate_timeseries(selected, n=120)
    df_ts["timestamp"] = pd.to_datetime(df_ts["timestamp"])

    sensor_display = {
        "vibration":   ("Vibration (g)", "#f6ad55"),
        "temperature": ("Temperature (°C)", "#fc8181"),
        "current":     ("Current (A)", "#63b3ed"),
        "power_w":     ("Power (W)", "#9f7aea"),
    }

    fig = make_subplots(rows=2, cols=2, subplot_titles=list(sensor_display.values()),
                        vertical_spacing=0.12, horizontal_spacing=0.08)
    positions = [(1,1),(1,2),(2,1),(2,2)]
    for (sensor, (label, color)), (row, col) in zip(sensor_display.items(), positions):
        fig.add_trace(go.Scatter(
            x=df_ts["timestamp"], y=df_ts[sensor],
            mode="lines", name=label,
            line=dict(color=color, width=1.5),
            fill="tozeroy", fillcolor=color.replace(")", ",0.08)").replace("rgb","rgba") if color.startswith("rgb") else color + "14",
        ), row=row, col=col)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(color="#718096", size=11),
    )
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(showgrid=True, gridcolor="#2d3748", color="#718096", row=i, col=j)
            fig.update_yaxes(showgrid=True, gridcolor="#2d3748", color="#718096", row=i, col=j)
    st.plotly_chart(fig, use_container_width=True)

    # ── RUL gauge ──────────────────────────────────────────────────
    col_gauge, col_proba = st.columns(2)

    with col_gauge:
        st.markdown('<div class="section-header">Remaining useful life gauge</div>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=rul,
            delta={"reference": 3200, "valueformat": ".0f"},
            title={"text": "RUL (cycles)", "font": {"color": "#a0aec0", "size": 13}},
            number={"font": {"color": _health_color(h), "family": "IBM Plex Mono", "size": 36}},
            gauge={
                "axis":       {"range": [0, 3500], "tickcolor": "#4a5568", "tickfont": {"color": "#4a5568"}},
                "bar":        {"color": _health_color(h), "thickness": 0.25},
                "bgcolor":    "#1a1d2e",
                "bordercolor":"#2d3748",
                "steps": [
                    {"range": [0, 350],   "color": "#2d1515"},
                    {"range": [350, 1050],"color": "#2d2515"},
                    {"range": [1050, 3500],"color": "#152d1c"},
                ],
                "threshold": {
                    "line": {"color": "#fc8181", "width": 3},
                    "thickness": 0.75,
                    "value": 350,
                },
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            height=260,
            margin=dict(l=20, r=20, t=30, b=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_proba:
        st.markdown('<div class="section-header">Fault probability breakdown</div>', unsafe_allow_html=True)
        fault_labels = ["normal","bearing_wear","imbalance","overheating","cavitation","valve_leak"]
        if dev["domain"] == "Smart Home":
            fault_labels = ["normal","refrigerant_leak","compressor_fail","drum_imbalance","pump_fail","heating_element_fail"]

        proba_base = np.random.dirichlet([10 if f == "normal" else max(0.1, (1-h)*3) for f in fault_labels])
        fig_prob = go.Figure(go.Bar(
            x=fault_labels,
            y=proba_base,
            marker_color=["#68d391" if f == "normal" else "#fc8181" for f in fault_labels],
            text=[f"{p:.2%}" for p in proba_base],
            textposition="outside",
            textfont=dict(size=10, family="IBM Plex Mono"),
        ))
        fig_prob.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#718096", tickfont=dict(size=9), tickangle=30),
            yaxis=dict(color="#718096", showgrid=True, gridcolor="#2d3748"),
            height=260,
            margin=dict(l=10, r=10, t=20, b=60),
            showlegend=False,
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    # ── Live reading table ─────────────────────────────────────────
    st.markdown('<div class="section-header">Current sensor values</div>', unsafe_allow_html=True)
    reading_df = pd.DataFrame([{
        "Sensor":    k.replace("_", " ").title(),
        "Value":     v,
        "Unit":      {"vibration":"g","temperature":"°C","current":"A","pressure":"bar","rpm":"RPM","humidity":"%","power_w":"W"}[k],
        "Status":    "⚠️ Elevated" if (k == "vibration" and v > 1.5) or (k == "temperature" and v > 80) else "✅ Normal",
    } for k, v in reading.items()])
    st.dataframe(reading_df, use_container_width=True, hide_index=True)

    if auto_refresh:
        time.sleep(refresh_sec)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════
# PAGE: ALERT CENTER
# ═══════════════════════════════════════════════════════════════════

elif page == "Alert Center":
    st.markdown("## Alert Center")

    # Generate mock alerts
    mock_alerts = []
    for did, dev in DEVICES.items():
        if dev["domain"] not in domain_filter:
            continue
        h   = dev["health"]
        urg = _health_to_urgency(h)
        rul = mock_rul(h)
        if urg == "critical":
            mock_alerts.append({
                "id": len(mock_alerts)+1, "device_id": did, "severity": "critical",
                "type": "rul_critical", "resolved": False,
                "message": f"RUL critically low: {rul:.0f} cycles remaining",
                "timestamp": (datetime.now() - timedelta(minutes=random.randint(2,60))).strftime("%H:%M:%S"),
            })
            mock_alerts.append({
                "id": len(mock_alerts)+1, "device_id": did, "severity": "critical",
                "type": "anomaly_detected", "resolved": False,
                "message": f"Anomaly detected — score {mock_anomaly_score(h):.5f}",
                "timestamp": (datetime.now() - timedelta(minutes=random.randint(1,30))).strftime("%H:%M:%S"),
            })
        elif urg == "warning":
            mock_alerts.append({
                "id": len(mock_alerts)+1, "device_id": did, "severity": "warning",
                "type": "rul_warning", "resolved": False,
                "message": f"RUL below threshold: {rul:.0f} cycles remaining",
                "timestamp": (datetime.now() - timedelta(minutes=random.randint(5,120))).strftime("%H:%M:%S"),
            })

    col_stats1, col_stats2, col_stats3 = st.columns(3)
    with col_stats1:
        n_crit = sum(1 for a in mock_alerts if a["severity"] == "critical")
        st.markdown(f'<div class="kpi-card kpi-critical"><div class="kpi-label">Critical alerts</div><div class="kpi-value">{n_crit}</div></div>', unsafe_allow_html=True)
    with col_stats2:
        n_warn = sum(1 for a in mock_alerts if a["severity"] == "warning")
        st.markdown(f'<div class="kpi-card kpi-warning"><div class="kpi-label">Warning alerts</div><div class="kpi-value">{n_warn}</div></div>', unsafe_allow_html=True)
    with col_stats3:
        st.markdown(f'<div class="kpi-card kpi-ok"><div class="kpi-label">Total active</div><div class="kpi-value">{len(mock_alerts)}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    if not mock_alerts:
        st.success("✅ No active alerts. All devices operating normally.")
    else:
        for alert in sorted(mock_alerts, key=lambda a: a["severity"]):
            sev   = alert["severity"]
            color = "#fc8181" if sev == "critical" else "#f6e05e"
            icon  = "🔴" if sev == "critical" else "🟡"
            st.markdown(f"""
            <div style="background:#1a1d2e;border-left:3px solid {color};border-radius:8px;
                        padding:0.8rem 1.1rem;margin-bottom:0.5rem;display:flex;align-items:center;gap:1rem">
              <div style="font-size:1.3rem">{icon}</div>
              <div style="flex:1">
                <div style="font-size:0.8rem;font-weight:500;color:#e2e8f0">{alert['device_id']}</div>
                <div style="font-size:0.75rem;color:#a0aec0;margin-top:2px">{alert['message']}</div>
              </div>
              <div style="text-align:right">
                <div style="font-size:0.65rem;color:#4a5568;font-family:monospace">{alert['timestamp']}</div>
                <span class="alert-badge badge-{'critical' if sev=='critical' else 'warning'}" style="margin-top:4px;display:inline-block">{sev}</span>
              </div>
            </div>""", unsafe_allow_html=True)

    # Alert trend
    st.markdown('<div class="section-header" style="margin-top:1.5rem">Alert frequency (last 24 hours)</div>', unsafe_allow_html=True)
    hours = [(datetime.now() - timedelta(hours=i)).strftime("%H:00") for i in range(23, -1, -1)]
    critical_counts = [random.randint(0, 3) for _ in hours]
    warning_counts  = [random.randint(0, 5) for _ in hours]

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(x=hours, y=critical_counts, name="Critical", marker_color="#fc8181"))
    fig_trend.add_trace(go.Bar(x=hours, y=warning_counts,  name="Warning",  marker_color="#f6e05e"))
    fig_trend.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#718096", tickfont=dict(size=10), tickangle=45),
        yaxis=dict(color="#718096", showgrid=True, gridcolor="#2d3748"),
        legend=dict(font=dict(color="#a0aec0")),
        height=250,
        margin=dict(l=10, r=10, t=10, b=60),
    )
    st.plotly_chart(fig_trend, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════

elif page == "Model Performance":
    st.markdown("## Model Performance")

    tab1, tab2, tab3 = st.tabs(["LSTM Autoencoder", "Fault Classifier", "RUL Estimator"])

    with tab1:
        st.markdown("### LSTM Autoencoder — Anomaly Detection")
        st.markdown("Trained on normal-only sequences. Anomaly = reconstruction error > threshold.")

        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Architecture",    "LSTM AE · 64→32→16→32→64")
        with col2: st.metric("Window size",     "30 timesteps")
        with col3: st.metric("Threshold pct",   "95th percentile (normal)")

        # Mock training curve
        epochs = list(range(1, 31))
        tr_loss = [0.08 * np.exp(-i*0.12) + 0.002 + random.gauss(0, 0.001) for i in epochs]
        vl_loss = [0.09 * np.exp(-i*0.11) + 0.003 + random.gauss(0, 0.0015) for i in epochs]
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=epochs, y=tr_loss, name="Train loss", line=dict(color="#63b3ed", width=2)))
        fig_loss.add_trace(go.Scatter(x=epochs, y=vl_loss, name="Val loss",   line=dict(color="#f6e05e", width=2, dash="dot")))
        fig_loss.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               xaxis=dict(title="Epoch",color="#718096",gridcolor="#2d3748"),
                               yaxis=dict(title="MSE Loss",color="#718096",gridcolor="#2d3748"),
                               legend=dict(font=dict(color="#a0aec0")), height=280, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_loss, use_container_width=True)

        # Reconstruction error distribution
        normal_errors  = np.random.exponential(0.008, 500) + 0.001
        anomaly_errors = np.random.exponential(0.04, 200) + 0.03
        threshold_val  = np.percentile(normal_errors, 95)
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=normal_errors,  name="Normal",  nbinsx=50, marker_color="#68d391", opacity=0.7))
        fig_dist.add_trace(go.Histogram(x=anomaly_errors, name="Anomaly", nbinsx=50, marker_color="#fc8181", opacity=0.7))
        fig_dist.add_vline(x=threshold_val, line_color="#f6e05e", line_dash="dash", annotation_text=f"Threshold={threshold_val:.4f}")
        fig_dist.update_layout(barmode="overlay", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               xaxis=dict(title="Reconstruction error",color="#718096",gridcolor="#2d3748"),
                               yaxis=dict(title="Count",color="#718096",gridcolor="#2d3748"),
                               legend=dict(font=dict(color="#a0aec0")), height=260, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab2:
        st.markdown("### Random Forest — Fault Classification")

        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Estimators",   "200 trees")
        with col2: st.metric("Max depth",    "20")
        with col3: st.metric("Class weight", "balanced")

        # Confusion matrix mock
        classes   = ["normal","bearing_wear","imbalance","overheating","cavitation","valve_leak"]
        n_cls     = len(classes)
        cm        = np.diag([random.randint(180, 220) for _ in range(n_cls)])
        for i in range(n_cls):
            for j in range(n_cls):
                if i != j: cm[i,j] = random.randint(0, 12)

        fig_cm = px.imshow(
            cm, x=classes, y=classes,
            color_continuous_scale=[[0,"#1a1d2e"],[1,"#63b3ed"]],
            labels=dict(x="Predicted", y="Actual", color="Count"),
            text_auto=True,
        )
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=360,
                             xaxis=dict(color="#a0aec0", tickangle=30, tickfont=dict(size=10)),
                             yaxis=dict(color="#a0aec0", tickfont=dict(size=10)),
                             coloraxis_colorbar=dict(tickfont=dict(color="#718096")),
                             margin=dict(l=10,r=10,t=20,b=80))
        st.plotly_chart(fig_cm, use_container_width=True)

        # Feature importance
        feat_names = ["vibration_rms","temperature_roll_mean","current_roc","vibration_fft_0","pressure_roll_std","rpm_roll_mean","power_w","humidity_roll_std","vibration_roll_max","temperature"]
        importances = sorted(np.random.dirichlet(np.ones(len(feat_names)) * 2), reverse=True)
        fig_fi = go.Figure(go.Bar(
            x=feat_names, y=importances,
            marker_color="#63b3ed",
            text=[f"{v:.3f}" for v in importances],
            textposition="outside", textfont=dict(size=9, family="IBM Plex Mono"),
        ))
        fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             xaxis=dict(color="#718096", tickangle=30, tickfont=dict(size=9)),
                             yaxis=dict(color="#718096", gridcolor="#2d3748", title="Importance"),
                             height=260, margin=dict(l=10,r=10,t=10,b=80), showlegend=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    with tab3:
        st.markdown("### Gradient Boosting — RUL Estimation")

        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Estimators",    "300 trees")
        with col2: st.metric("Learning rate", "0.05")
        with col3: st.metric("Subsample",     "0.8")

        # Actual vs predicted scatter
        n = 400
        y_true = np.random.uniform(0, 3200, n)
        noise  = np.random.normal(0, 180, n)
        y_pred = np.clip(y_true + noise, 0, 3500)

        mae  = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        r2   = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1: st.metric("MAE",  f"{mae:.1f} cycles")
        with col_m2: st.metric("RMSE", f"{rmse:.1f} cycles")
        with col_m3: st.metric("R²",   f"{r2:.4f}")

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=y_true, y=y_pred, mode="markers",
            marker=dict(color="#63b3ed", size=4, opacity=0.5),
            name="Predictions",
        ))
        fig_scatter.add_trace(go.Scatter(
            x=[0, 3500], y=[0, 3500], mode="lines",
            line=dict(color="#68d391", dash="dash", width=1.5),
            name="Perfect fit",
        ))
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Actual RUL", color="#718096", gridcolor="#2d3748"),
            yaxis=dict(title="Predicted RUL", color="#718096", gridcolor="#2d3748"),
            legend=dict(font=dict(color="#a0aec0")),
            height=320, margin=dict(l=10,r=10,t=10,b=10),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
