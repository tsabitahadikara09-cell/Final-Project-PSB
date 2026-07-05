import math
from typing import Dict
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ===================== SETTING HALAMAN =====================
st.set_page_config(
    page_title="FP PSB - Gait Parameter Extraction & Dynamic EMG",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: white; }
    section[data-testid="stSidebar"] { background-color: #262832; }
    .block-container { padding-top: 2.2rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: white !important; font-weight: 800 !important; }
    div[data-testid="stMetricValue"] { color: #14ff4b; }
    .small-box {
        background: #203b63;
        padding: 14px 16px;
        border-radius: 8px;
        color: #6bb6ff;
        font-size: 14px;
        margin-top: 12px;
        line-height: 1.6;
    }
    .success-box {
        background: #205438;
        padding: 14px 16px;
        border-radius: 8px;
        color: #39ff7d;
        font-weight: 700;
        margin-top: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ===================== CONSTANTS =====================
MUSCLE_SHORT = ["GMAX", "BFS", "BFL", "VM", "VL", "RF", "MG", "TA", "SOL"]


# ===================== BASIC HELPERS =====================
def parse_txt_file(uploaded_file) -> np.ndarray:
    content = uploaded_file.read().decode("utf-8", errors="ignore").splitlines()
    rows = []

    for line in content:
        parts = line.split()
        if len(parts) >= 15:
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue

    return np.array(rows, dtype=float)


def infer_fs(t: np.ndarray) -> float:
    if len(t) < 2:
        return 100.0

    dt = np.median(np.diff(t))

    if dt <= 0:
        return 100.0

    fs = 1.0 / dt

    if fs < 5:
        return 1000.0

    return float(fs)


def manual_normalize(signal):
    signal = list(np.asarray(signal, dtype=float))

    if len(signal) == 0:
        return []

    max_val = max(signal)
    min_val = min(signal)

    if max_val == min_val:
        return [0.0 for _ in signal]

    return [(x - min_val) / (max_val - min_val) for x in signal]


# ===================== LPF DIPERBAIKI =====================
def manual_lowpass(signal, fs=100.0, cutoff=6.0):
    """
    Low Pass Filter orde 1 manual.

    Versi ini hanya filter 1 kali dari depan ke belakang.
    Tujuannya agar cutoff 6 Hz tidak menghasilkan output yang terlalu halus.
    """

    signal = list(np.asarray(signal, dtype=float))

    if len(signal) == 0:
        return []

    if cutoff <= 0:
        return signal

    RC = 1.0 / (2.0 * math.pi * cutoff)
    dt = 1.0 / fs
    alpha = dt / (RC + dt)

    filtered = [0.0] * len(signal)
    filtered[0] = signal[0]

    for i in range(1, len(signal)):
        filtered[i] = filtered[i - 1] + alpha * (signal[i] - filtered[i - 1])

    return filtered


def extract_phase_crossings(t, signal, threshold=0.05):
    active = [x >= threshold for x in signal]
    on_times, off_times = [], []

    for i in range(1, len(active)):
        if active[i] and not active[i - 1]:
            on_times.append(float(t[i]))
        elif not active[i] and active[i - 1]:
            off_times.append(float(t[i]))

    return on_times, off_times


def extract_gait_parameters(t, heel_filt, toe_filt=None, threshold=0.05):
    heel_on, heel_off = extract_phase_crossings(t, heel_filt, threshold)

    if len(heel_on) > 1:
        cycles = []

        for i in range(len(heel_on) - 1):
            s = heel_on[i]
            e = heel_on[i + 1]

            if e <= s:
                continue

            if toe_filt is not None:
                toe_on, _ = extract_phase_crossings(t, toe_filt, threshold)
                toe_between = [x for x in toe_on if s < x < e]
                toe_off = toe_between[0] if toe_between else s + 0.63 * (e - s)
            else:
                toe_off = s + 0.63 * (e - s)

            cycles.append((s, e, toe_off))

        if cycles:
            cycle_times = [e - s for s, e, _ in cycles]
            mean_cycle = sum(cycle_times) / len(cycle_times)
            cadence = 60.0 / mean_cycle if mean_cycle > 0 else 0.0

            return mean_cycle, cadence, len(cycles), heel_on, cycles

    return 0.0, 0.0, 0, [], []


def interp_cycle_signal(t, signal, start_time, end_time, percent_axis):
    t = np.asarray(t, dtype=float)
    signal = np.asarray(signal, dtype=float)

    mask = (t >= start_time) & (t <= end_time)

    if np.sum(mask) < 2 or end_time <= start_time:
        return np.zeros_like(percent_axis, dtype=float)

    t_cycle = t[mask]
    s_cycle = signal[mask]

    cycle_percent = (t_cycle - start_time) / (end_time - start_time) * 100.0

    return np.interp(percent_axis, cycle_percent, s_cycle)


def manual_stft(signal, fs, nperseg=256, overlap=128):
    signal = np.asarray(signal, dtype=float)

    step = max(1, nperseg - overlap)

    if len(signal) < nperseg:
        padded = np.zeros(nperseg)
        padded[:len(signal)] = signal
        signal = padded

    window = np.hanning(nperseg)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)

    times = []
    mat = []

    for start in range(0, len(signal) - nperseg + 1, step):
        segment = signal[start:start + nperseg]
        spectrum = np.fft.rfft(segment * window)
        mag = np.abs(spectrum)

        mat.append(mag)
        times.append((start + nperseg / 2) / fs)

    if not mat:
        mat = [np.zeros_like(freqs)]
        times = [0]

    return freqs, np.array(times), np.array(mat).T


# ===================== PLOT HELPERS =====================
def dark_layout(fig, title, x_title="", y_title="", height=360):
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(color="white", size=18)),
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=height,
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend=dict(orientation="h", y=1.12, x=0.0, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=50, r=25, t=70, b=50)
    )

    fig.update_xaxes(gridcolor="rgba(180,180,180,0.25)")
    fig.update_yaxes(gridcolor="rgba(180,180,180,0.25)")

    return fig


def white_layout(fig, title, x_title="", y_title="", height=420):
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(color="black", size=20)),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=height,
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="lightgray", borderwidth=1),
        margin=dict(l=55, r=25, t=65, b=50)
    )

    fig.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray")

    return fig


def make_single_signal_plot(x, signal, title, line_name, color, x_title="Waktu (s)", y_title="Amplitude", height=340):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=signal,
            mode="lines",
            name=line_name,
            line=dict(color=color, width=2)
        )
    )

    return dark_layout(fig, title, x_title, y_title, height)


def make_heel_toe_plot(t, heel, toe, title, height=360):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=t,
            y=heel,
            mode="lines",
            name="Heel / FSR Biru",
            line=dict(color="blue", width=2)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=toe,
            mode="lines",
            name="Toe / FSR Merah",
            line=dict(color="red", width=2)
        )
    )

    return dark_layout(fig, title, "Waktu (s)", "Amplitude", height)


def make_heel_toe_threshold_plot(t, heel, toe, threshold=0.05, title="HEEL & TOE THRESHOLDING", height=360):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t, y=heel, mode="lines", name="Heel Filtering", line=dict(color="blue", width=2)))
    fig.add_trace(go.Scatter(x=t, y=toe, mode="lines", name="Toe Filtering", line=dict(color="red", width=2)))
    fig.add_trace(go.Scatter(x=t, y=[threshold] * len(t), mode="lines", name="Threshold 0.05", line=dict(color="gray", width=1.4, dash="dash")))

    fig = dark_layout(fig, title, "Waktu (s)", "Amplitude", height)
    fig.update_yaxes(range=[-0.05, 1.05])

    return fig


def make_single_threshold_plot(t, signal, threshold=0.05, title="THRESHOLDING", line_name="Filtering", color="blue", height=350):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t, y=signal, mode="lines", name=line_name, line=dict(color=color, width=2)))
    fig.add_trace(go.Scatter(x=t, y=[threshold] * len(t), mode="lines", name="Threshold 0.05", line=dict(color="gray", width=1.4, dash="dash")))

    fig = dark_layout(fig, title, "Waktu (s)", "Amplitude", height)
    fig.update_yaxes(range=[-0.05, 1.05])

    return fig


def make_phase_detection_plot(t, signal, on_times, off_times, title, line_name, color):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t, y=signal, mode="lines", name=line_name, line=dict(color=color, width=2.4)))
    fig.add_trace(go.Scatter(x=t, y=[0.05] * len(t), mode="lines", name="Threshold 0.05", line=dict(color="gray", width=1, dash="dash")))

    for x in on_times:
        fig.add_vline(x=x, line_color="lime", line_width=1.4, line_dash="dash")

    for x in off_times:
        fig.add_vline(x=x, line_color="red", line_width=1.4, line_dash="dash")

    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="ON / Naik", line=dict(color="lime", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="OFF / Turun", line=dict(color="red", width=2, dash="dash")))

    fig = dark_layout(fig, title, "Waktu (s)", "Amplitude", 380)
    fig.update_yaxes(range=[-0.05, 1.05])

    return fig


def make_heel_toe_phase_detection_plot(t, heel, toe, heel_on, heel_off, toe_on, toe_off, threshold=0.05, title="HEEL & TOE PHASE DETECTION", height=390):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t, y=heel, mode="lines", name="Heel Normalized", line=dict(color="blue", width=2)))
    fig.add_trace(go.Scatter(x=t, y=toe, mode="lines", name="Toe Normalized", line=dict(color="red", width=2)))
    fig.add_trace(go.Scatter(x=t, y=[threshold] * len(t), mode="lines", name="Threshold 0.05", line=dict(color="gray", width=1.2, dash="dash")))

    for x in heel_on:
        fig.add_vline(x=x, line_color="lime", line_width=1.2, line_dash="dash")

    for x in heel_off:
        fig.add_vline(x=x, line_color="red", line_width=1.2, line_dash="dash")

    for x in toe_on:
        fig.add_vline(x=x, line_color="cyan", line_width=1.2, line_dash="dot")

    for x in toe_off:
        fig.add_vline(x=x, line_color="orange", line_width=1.2, line_dash="dot")

    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Heel ON", line=dict(color="lime", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Heel OFF", line=dict(color="red", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Toe ON", line=dict(color="cyan", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Toe OFF", line=dict(color="orange", width=2, dash="dot")))

    fig = dark_layout(fig, title, "Waktu (s)", "Amplitude", height)
    fig.update_yaxes(range=[-0.05, 1.05])

    return fig


def plot_multi_input_output(x, signals: Dict[str, np.ndarray], title, x_title, y_title, colors, height=340):
    fig = go.Figure()

    for name, sig in signals.items():
        fig.add_trace(
            go.Scatter(
                x=x,
                y=sig,
                mode="lines",
                name=name,
                line=dict(color=colors.get(name, None), width=1.8)
            )
        )

    return dark_layout(fig, title, x_title, y_title, height)


def add_on_off_lines_for_channel(fig, on_times, off_times, y0, y1):
    for x in on_times:
        fig.add_shape(
            type="line",
            x0=x,
            x1=x,
            y0=y0,
            y1=y1,
            line=dict(color="lime", width=1.1, dash="dash")
        )

    for x in off_times:
        fig.add_shape(
            type="line",
            x0=x,
            x1=x,
            y0=y0,
            y1=y1,
            line=dict(color="red", width=1.1, dash="dash")
        )


def plot_emg_stacked(t, data, title, labels, mode="raw", threshold=0.05, add_onoff=False):
    fig = go.Figure()

    for i, label in enumerate(labels):
        sig = np.asarray(data[:, i], dtype=float)

        if mode in ["env", "rect", "activation"]:
            sig_plot = sig + i * 1.2
        else:
            sig_plot = np.asarray(manual_normalize(sig)) * 0.85 + i * 1.2

        fig.add_trace(
            go.Scatter(
                x=t,
                y=sig_plot,
                mode="lines",
                name=label,
                line=dict(color="lime", width=1.4)
            )
        )

        if add_onoff:
            on, off = extract_phase_crossings(t, sig, threshold)
            add_on_off_lines_for_channel(fig, on, off, i * 1.2, i * 1.2 + 1.0)

    if add_onoff:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="ON", line=dict(color="lime", dash="dash")))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="OFF", line=dict(color="red", dash="dash")))

    fig = dark_layout(fig, title, "Waktu (s)", "Muscle", 520)
    fig.update_yaxes(tickvals=[i * 1.2 for i in range(len(labels))], ticktext=labels)

    return fig


def activation_segments(t, signal, threshold=0.05):
    active = np.asarray(signal) >= threshold

    segs = []
    start = None

    for i in range(1, len(active)):
        if active[i] and not active[i - 1]:
            start = float(t[i])
        elif not active[i] and active[i - 1]:
            if start is not None:
                end = float(t[i])
                if end > start:
                    segs.append((start, end))
            start = None

    if start is not None:
        segs.append((start, float(t[-1])))

    out = []

    for s, e in segs:
        gap = min(0.03, max((e - s) * 0.2, 0.0))
        if e - s > 2 * gap:
            out.append((s + gap, e - gap))

    return out


def plot_activation_bars(t, env, labels, threshold=0.05):
    fig = go.Figure()

    for i, label in enumerate(labels):
        segs = activation_segments(t, env[:, i], threshold)

        for s, e in segs:
            fig.add_trace(
                go.Scatter(
                    x=[s, e],
                    y=[i, i],
                    mode="lines",
                    name=label,
                    line=dict(color="lime", width=8),
                    showlegend=False
                )
            )

    for label in labels:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=label, line=dict(color="lime", width=8)))

    fig = dark_layout(fig, "MUSCLE ACTIVATION EACH CYCLE", "Waktu (s)", "Muscle", 480)
    fig.update_yaxes(tickvals=list(range(len(labels))), ticktext=labels)

    return fig


def plot_single_emg(t, raw, rect, env, threshold, label):
    figs = []

    figs.append(
        dark_layout(
            go.Figure([
                go.Scatter(x=t, y=raw, mode="lines", name=f"{label} Raw", line=dict(color="cyan"))
            ]),
            f"RAW EMG - {label}",
            "Waktu (s)",
            "EMG",
            300
        )
    )

    figs.append(
        dark_layout(
            go.Figure([
                go.Scatter(x=t, y=rect, mode="lines", name=f"{label} Rectification", line=dict(color="orange"))
            ]),
            f"RECTIFICATION - {label}",
            "Waktu (s)",
            "Rectified EMG",
            300
        )
    )

    fig_env = go.Figure()

    fig_env.add_trace(go.Scatter(x=t, y=env, mode="lines", name=f"{label} Envelope", line=dict(color="lime", width=2)))
    fig_env.add_trace(go.Scatter(x=t, y=[threshold] * len(t), mode="lines", name="Threshold", line=dict(color="gray", dash="dash")))

    on, off = extract_phase_crossings(t, env, threshold)

    for x in on:
        fig_env.add_vline(x=x, line_color="lime", line_dash="dash", line_width=1.3)

    for x in off:
        fig_env.add_vline(x=x, line_color="red", line_dash="dash", line_width=1.3)

    figs.append(
        dark_layout(
            fig_env,
            f"ENVELOPED FILTER - {label}",
            "Waktu (s)",
            "Envelope",
            360
        )
    )

    fig_act = go.Figure()
    segs = activation_segments(t, env, threshold)

    for s, e in segs:
        fig_act.add_trace(
            go.Scatter(
                x=[s, e],
                y=[0, 0],
                mode="lines",
                line=dict(color="lime", width=10),
                showlegend=False
            )
        )

    fig_act = dark_layout(fig_act, f"MUSCLE ACTIVATION - {label}", "Waktu (s)", "Activation", 260)
    fig_act.update_yaxes(range=[-0.5, 0.5], tickvals=[0], ticktext=[label])

    figs.append(fig_act)

    return figs


def plot_gait_analysis_delphi(percent_axis, hip_mean, knee_mean, ankle_mean, heel_mean, toe_mean, to_percent):
    figs = []

    specs = [
        ("HIP JOINT", hip_mean, "red"),
        ("KNEE JOINT", knee_mean, "green"),
        ("ANKLE JOINT", ankle_mean, "blue")
    ]

    for title, sig, color in specs:
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=percent_axis, y=sig, mode="lines", name=title, line=dict(color=color, width=2)))

        for name, x, c in [
            ("IC", 0, "cyan"),
            ("FF", 12, "yellow"),
            ("HO", 33, "cyan"),
            ("TO", to_percent, "cyan")
        ]:
            fig.add_vline(x=x, line_dash="dot", line_color=c, line_width=1.3)
            fig.add_trace(go.Scatter(x=[x], y=[np.interp(x, percent_axis, sig)], mode="markers", name=name, marker=dict(size=8)))

        figs.append(dark_layout(fig, title, "gait cycle [%]", "Deg", 250))

    figp = go.Figure()

    figp.add_trace(go.Scatter(x=percent_axis, y=heel_mean * 2.2, mode="lines", name="HEEL", line=dict(color="red", dash="dash")))
    figp.add_trace(go.Scatter(x=percent_axis, y=toe_mean * 2.2, mode="lines", name="TOE", line=dict(color="blue", dash="dot")))
    figp.add_trace(go.Scatter(x=percent_axis, y=np.ones_like(percent_axis) * 0.2, mode="lines", name="THD", line=dict(color="gray")))

    figs.append(dark_layout(figp, "GAIT PHASE", "gait cycle [%]", "Volt", 250))

    return figs


# ===================== SIDEBAR =====================
st.sidebar.markdown("## Panel Kontrol")

uploaded = st.sidebar.file_uploader("LOAD DATA (TXT)", type=["txt", "TXT"])

cutoff_lpf = st.sidebar.slider(
    "Cutoff Frequency LPF (Hz)",
    1.0,
    30.0,
    6.0,
    0.5
)

emg_threshold = st.sidebar.slider(
    "Threshold EMG Activation",
    0.01,
    0.90,
    0.05,
    0.01
)

st.sidebar.markdown(
    """
    <div class="small-box">
    Format minimal 15 kolom:<br>
    time, heel, toe, hip, knee, ankle, EMG1-EMG9.
    </div>
    """,
    unsafe_allow_html=True
)


# ===================== TITLE =====================
st.title("Gait Parameter Extraction & STFT Analysis")

if uploaded is None:
    st.info("Upload file TXT terlebih dahulu.")
    st.stop()


# ===================== PROCESS DATA =====================
data = parse_txt_file(uploaded)

if data.size == 0 or data.shape[1] < 15:
    st.error("Format data tidak sesuai. Minimal 15 kolom: time, heel, toe, hip, knee, ankle, EMG1-EMG9.")
    st.stop()


t = data[:, 0]

if len(t) > 2 and np.median(np.diff(t)) >= 0.5:
    fs = 1000.0
    t_sec = np.arange(len(t)) / fs
else:
    t_sec = t
    fs = infer_fs(t_sec)

n = np.arange(len(t_sec))

heel_raw = data[:, 1]
toe_raw = data[:, 2]
hip_raw = data[:, 3]
knee_raw = data[:, 4]
ankle_raw = data[:, 5]
emg_raw = data[:, 6:15]


# ===================== FILTERING =====================
heel_filt = np.array(manual_lowpass(heel_raw, fs=fs, cutoff=cutoff_lpf))
toe_filt = np.array(manual_lowpass(toe_raw, fs=fs, cutoff=cutoff_lpf))

heel_norm = np.array(manual_normalize(heel_filt))
toe_norm = np.array(manual_normalize(toe_filt))

hip_filt = np.array(manual_lowpass(hip_raw, fs=fs, cutoff=cutoff_lpf))
knee_filt = np.array(manual_lowpass(knee_raw, fs=fs, cutoff=cutoff_lpf))
ankle_filt = np.array(manual_lowpass(ankle_raw, fs=fs, cutoff=cutoff_lpf))

heel_on, heel_off = extract_phase_crossings(t_sec, heel_norm, 0.05)
toe_on, toe_off = extract_phase_crossings(t_sec, toe_norm, 0.05)

mean_cycle, cadence, jumlah_cycle, heel_crossings, cycles = extract_gait_parameters(
    t_sec,
    heel_norm,
    toe_norm,
    0.05
)


# ===================== EMG PROCESSING =====================
emg_rect = np.abs(emg_raw)
emg_env = np.zeros_like(emg_rect, dtype=float)

for i in range(emg_rect.shape[1]):
    env = manual_lowpass(emg_rect[:, i], fs=fs, cutoff=cutoff_lpf)
    emg_env[:, i] = np.array(manual_normalize(env))


# ===================== INFO SIDEBAR =====================
st.sidebar.markdown(
    f"""
    <div class="success-box">
    Jumlah Data: {len(data)}<br>
    Fs: {fs:.2f} Hz
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("### Temporal Parameters:")
st.sidebar.markdown(f"- Rata-rata Cycle: **{mean_cycle:.3f} s**")
st.sidebar.markdown(f"- Cadence: **{cadence:.2f} step/min**")
st.sidebar.markdown(f"- Jumlah Cycle: **{jumlah_cycle}**")


# ===================== TABS =====================
tab_gait, tab_emg, tab_pre, tab_cycle_param, tab_stft = st.tabs([
    "GAIT PARAMETERS",
    "DYNAMIC EMG",
    "EMG PREPROCESSING",
    "SIKLUS & PARAMETER",
    "STFT ANALYSIS"
])


# ===================== TAB GAIT PARAMETERS =====================
with tab_gait:
    subt1, subt2, subt3, subt4 = st.tabs(["Gabungan", "Heel", "Toe", "Joint Angle"])

    with subt1:
        st.plotly_chart(make_heel_toe_plot(t_sec, heel_raw, toe_raw, "1. HEEL & TOE INPUT"), use_container_width=True)
        st.plotly_chart(make_heel_toe_plot(t_sec, heel_filt, toe_filt, f"2. HEEL & TOE OUTPUT FILTERING Cutoff {cutoff_lpf:.1f} Hz"), use_container_width=True)
        st.plotly_chart(make_heel_toe_plot(t_sec, heel_norm, toe_norm, "3. HEEL & TOE NORMALIZED 0-1"), use_container_width=True)
        st.plotly_chart(make_heel_toe_threshold_plot(t_sec, heel_norm, toe_norm, threshold=0.05, title="4. HEEL & TOE THRESHOLDING 5%"), use_container_width=True)
        st.plotly_chart(make_heel_toe_phase_detection_plot(t_sec, heel_norm, toe_norm, heel_on, heel_off, toe_on, toe_off), use_container_width=True)

    with subt2:
        st.plotly_chart(make_single_signal_plot(t_sec, heel_raw, "HEEL INPUT", "Heel Input", "blue"), use_container_width=True)
        st.plotly_chart(make_single_signal_plot(t_sec, heel_filt, f"HEEL OUTPUT FILTERING Cutoff {cutoff_lpf:.1f} Hz", "Heel Filtering", "blue"), use_container_width=True)
        st.plotly_chart(make_single_signal_plot(t_sec, heel_norm, "HEEL NORMALIZED", "Heel Normalized", "blue"), use_container_width=True)
        st.plotly_chart(make_single_threshold_plot(t_sec, heel_norm, threshold=0.05, title="HEEL THRESHOLDING 5%", line_name="Heel Normalized", color="blue"), use_container_width=True)
        st.plotly_chart(make_phase_detection_plot(t_sec, heel_norm, heel_on, heel_off, "HEEL PHASE DETECTION 5%", "Heel Normalized", "blue"), use_container_width=True)

    with subt3:
        st.plotly_chart(make_single_signal_plot(t_sec, toe_raw, "TOE INPUT", "Toe Input", "red"), use_container_width=True)
        st.plotly_chart(make_single_signal_plot(t_sec, toe_filt, f"TOE OUTPUT FILTERING Cutoff {cutoff_lpf:.1f} Hz", "Toe Filtering", "red"), use_container_width=True)
        st.plotly_chart(make_single_signal_plot(t_sec, toe_norm, "TOE NORMALIZED", "Toe Normalized", "red"), use_container_width=True)
        st.plotly_chart(make_single_threshold_plot(t_sec, toe_norm, threshold=0.05, title="TOE THRESHOLDING 5%", line_name="Toe Normalized", color="red"), use_container_width=True)
        st.plotly_chart(make_phase_detection_plot(t_sec, toe_norm, toe_on, toe_off, "TOE PHASE DETECTION 5%", "Toe Normalized", "red"), use_container_width=True)

    with subt4:
        st.plotly_chart(
            plot_multi_input_output(
                n,
                {
                    "Hip Input": hip_raw,
                    "Knee Input": knee_raw,
                    "Ankle Input": ankle_raw
                },
                "JOINT ANGLE INPUT",
                "n (sample)",
                "Degree",
                {
                    "Hip Input": "red",
                    "Knee Input": "green",
                    "Ankle Input": "blue"
                }
            ),
            use_container_width=True
        )

        st.plotly_chart(
            plot_multi_input_output(
                n,
                {
                    "Hip Filtering": hip_filt,
                    "Knee Filtering": knee_filt,
                    "Ankle Filtering": ankle_filt
                },
                f"JOINT ANGLE OUTPUT FILTERING Cutoff {cutoff_lpf:.1f} Hz",
                "n (sample)",
                "Degree",
                {
                    "Hip Filtering": "red",
                    "Knee Filtering": "green",
                    "Ankle Filtering": "blue"
                }
            ),
            use_container_width=True
        )

        ht, kt, at = st.tabs(["Hip", "Knee", "Ankle"])

        with ht:
            st.plotly_chart(make_single_signal_plot(n, hip_raw, "HIP JOINT INPUT", "Hip Input", "red", "n (sample)", "Degree"), use_container_width=True)
            st.plotly_chart(make_single_signal_plot(n, hip_filt, "HIP JOINT OUTPUT", "Hip Filtering", "red", "n (sample)", "Degree"), use_container_width=True)

        with kt:
            st.plotly_chart(make_single_signal_plot(n, knee_raw, "KNEE JOINT INPUT", "Knee Input", "green", "n (sample)", "Degree"), use_container_width=True)
            st.plotly_chart(make_single_signal_plot(n, knee_filt, "KNEE JOINT OUTPUT", "Knee Filtering", "green", "n (sample)", "Degree"), use_container_width=True)

        with at:
            st.plotly_chart(make_single_signal_plot(n, ankle_raw, "ANKLE JOINT INPUT", "Ankle Input", "blue", "n (sample)", "Degree"), use_container_width=True)
            st.plotly_chart(make_single_signal_plot(n, ankle_filt, "ANKLE JOINT OUTPUT", "Ankle Filtering", "blue", "n (sample)", "Degree"), use_container_width=True)


# ===================== TAB DYNAMIC EMG =====================
with tab_emg:
    st.subheader("Dynamic EMG")

    pilihan = st.selectbox("Pilih sinyal:", ["GABUNGAN 9 OTOT"] + MUSCLE_SHORT)

    if pilihan == "GABUNGAN 9 OTOT":
        st.plotly_chart(plot_emg_stacked(t_sec, emg_raw, "RAW EMG 9 CHANNEL", MUSCLE_SHORT, mode="raw"), use_container_width=True)
        st.plotly_chart(plot_emg_stacked(t_sec, emg_rect, "RECTIFICATION 9 CHANNEL", MUSCLE_SHORT, mode="rect"), use_container_width=True)
        st.plotly_chart(plot_emg_stacked(t_sec, emg_env, f"ENVELOPED FILTER Cutoff {cutoff_lpf:.1f} Hz", MUSCLE_SHORT, mode="env", threshold=emg_threshold, add_onoff=True), use_container_width=True)
        st.plotly_chart(plot_activation_bars(t_sec, emg_env, MUSCLE_SHORT, emg_threshold), use_container_width=True)

    else:
        idx = MUSCLE_SHORT.index(pilihan)
        figs = plot_single_emg(t_sec, emg_raw[:, idx], emg_rect[:, idx], emg_env[:, idx], emg_threshold, pilihan)

        for fig in figs:
            st.plotly_chart(fig, use_container_width=True)


# ===================== TAB EMG PREPROCESSING =====================
with tab_pre:
    ch = st.selectbox("Pilih channel EMG:", MUSCLE_SHORT)
    idx = MUSCLE_SHORT.index(ch)

    st.plotly_chart(
        make_single_signal_plot(
            t_sec,
            emg_raw[:, idx],
            f"RAW EMG SIGNAL - {ch}",
            f"{ch} Raw",
            "cyan",
            "time (sec)",
            "EMG"
        ),
        use_container_width=True
    )

    fig_pre = go.Figure()

    fig_pre.add_trace(
        go.Scatter(
            x=t_sec,
            y=emg_rect[:, idx],
            mode="lines",
            name="Rectified",
            line=dict(color="orange")
        )
    )

    fig_pre.add_trace(
        go.Scatter(
            x=t_sec,
            y=emg_env[:, idx],
            mode="lines",
            name="LPF Envelope",
            line=dict(color="lime", width=2)
        )
    )

    st.plotly_chart(
        dark_layout(
            fig_pre,
            f"PREPROCESSED EMG RECTIFIED & LPF Cutoff {cutoff_lpf:.1f} Hz",
            "time (sec)",
            "Processed EMG"
        ),
        use_container_width=True
    )


# ===================== TAB SIKLUS & PARAMETER =====================
with tab_cycle_param:
    st.subheader("Analisis Kinematik per Siklus Normalized 0-100%")

    if not cycles:
        st.warning("Siklus tidak terdeteksi. Periksa sinyal Heel/Toe atau threshold.")

    else:
        cycle_options = [f"Siklus {i + 1}" for i in range(len(cycles))]
        selected_cycle = st.selectbox("Pilih Siklus untuk Divisualisasikan:", cycle_options)

        idx_cycle = cycle_options.index(selected_cycle)
        start_time, end_time, toe_off_time = cycles[idx_cycle]

        percent_axis = np.linspace(0, 100, 101)

        hip_cycle = interp_cycle_signal(t_sec, hip_filt, start_time, end_time, percent_axis)
        knee_cycle = interp_cycle_signal(t_sec, knee_filt, start_time, end_time, percent_axis)
        ankle_cycle = interp_cycle_signal(t_sec, ankle_filt, start_time, end_time, percent_axis)

        heel_cycle = interp_cycle_signal(t_sec, heel_filt, start_time, end_time, percent_axis)
        toe_cycle = interp_cycle_signal(t_sec, toe_filt, start_time, end_time, percent_axis)

        to_percent = ((toe_off_time - start_time) / (end_time - start_time)) * 100.0

        st.markdown(f"### Detail Titik Sentuh - {selected_cycle}")

        event_df = pd.DataFrame({
            "Joint": ["Hip", "Hip", "Knee", "Knee", "Ankle", "Ankle"],
            "Event": [
                "IC 0%",
                f"TO {to_percent:.2f}%",
                "IC 0%",
                f"TO {to_percent:.2f}%",
                "IC 0%",
                f"TO {to_percent:.2f}%"
            ],
            "Derajat": [
                round(float(hip_cycle[0]), 2),
                round(float(np.interp(to_percent, percent_axis, hip_cycle)), 2),
                round(float(knee_cycle[0]), 2),
                round(float(np.interp(to_percent, percent_axis, knee_cycle)), 2),
                round(float(ankle_cycle[0]), 2),
                round(float(np.interp(to_percent, percent_axis, ankle_cycle)), 2)
            ]
        })

        st.dataframe(event_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Temporal Parameters Detailed per Cycle")

        rows = []

        for i, (s, e, toe_off) in enumerate(cycles):
            gait_cycle_time = e - s
            stance_time = toe_off - s
            swing_time = e - toe_off

            stance_percent = (stance_time / gait_cycle_time) * 100 if gait_cycle_time > 0 else 0
            swing_percent = (swing_time / gait_cycle_time) * 100 if gait_cycle_time > 0 else 0

            rows.append([
                i + 1,
                round(s, 3),
                round(toe_off, 3),
                round(e, 3),
                round(gait_cycle_time, 3),
                round(stance_time, 3),
                round(swing_time, 3),
                round(stance_percent, 2),
                round(swing_percent, 2)
            ])

        temporal_df = pd.DataFrame(
            rows,
            columns=[
                "cycle",
                "start_time",
                "toe_off_time",
                "end_time",
                "gait_cycle_time",
                "stance_time",
                "swing_time",
                "stance_percent",
                "swing_percent"
            ]
        )

        st.dataframe(temporal_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Joint Angle Parameters")

        selected_joint = st.selectbox("Pilih Joint untuk ROM Total:", ["hip", "knee", "ankle"])

        joint_signal = {
            "hip": hip_filt,
            "knee": knee_filt,
            "ankle": ankle_filt
        }[selected_joint]

        joint_param_df = pd.DataFrame({
            "Parameter": ["Max deg", "Min deg", "ROM deg"],
            "Nilai": [
                round(float(np.max(joint_signal)), 2),
                round(float(np.min(joint_signal)), 2),
                round(float(np.max(joint_signal) - np.min(joint_signal)), 2)
            ]
        })

        st.dataframe(joint_param_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Tampilan Gait Analysis seperti Delphi")

        for fig in plot_gait_analysis_delphi(
            percent_axis,
            hip_cycle,
            knee_cycle,
            ankle_cycle,
            heel_cycle,
            toe_cycle,
            to_percent
        ):
            st.plotly_chart(fig, use_container_width=True)


# ===================== TAB STFT =====================
with tab_stft:
    st.subheader("STFT Analysis")

    signal_options = ["heel", "toe", "hip", "knee", "ankle"] + MUSCLE_SHORT
    sel = st.selectbox("Pilih sinyal:", signal_options)

    sig_map = {
        "heel": heel_filt,
        "toe": toe_filt,
        "hip": hip_filt,
        "knee": knee_filt,
        "ankle": ankle_filt
    }

    for i, name in enumerate(MUSCLE_SHORT):
        sig_map[name] = emg_env[:, i]

    sig = sig_map[sel]

    freqs, times, power = manual_stft(sig, fs, nperseg=256, overlap=128)

    fig = go.Figure(
        data=go.Heatmap(
            x=times,
            y=freqs,
            z=power,
            colorscale="Viridis"
        )
    )

    fig = dark_layout(fig, f"STFT Spectrogram - {sel}", "Time (s)", "Frequency (Hz)", 520)
    fig.update_yaxes(range=[0, min(30, fs / 2)])

    st.plotly_chart(fig, use_container_width=True)
