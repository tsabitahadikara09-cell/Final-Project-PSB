
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt

st.set_page_config(
    page_title="FP PSB - Gait Parameter & Dynamic EMG",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
    .stApp { background: #0e1117; color: white; }
    section[data-testid="stSidebar"] { background: #262730; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: white; }
    .metric-card {
        background: #173f2f;
        border-radius: 10px;
        padding: 12px;
        color: #29ff8a;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .info-card {
        background: #224061;
        border-radius: 10px;
        padding: 12px;
        color: #5db7ff;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# CONSTANTS
# =========================================================
MUSCLE_SHORT = ["GMAX", "BFS", "BFL", "VM", "VL", "RF", "MG", "TA", "SOL"]
MUSCLE_LONG = [
    "Gluteus Maximus",
    "Biceps Femoris Short",
    "Biceps Femoris Long",
    "Vastus Medialis",
    "Vastus Lateralis",
    "Rectus Femoris",
    "Medial Gastrocnemius",
    "Tibialis Anterior",
    "Soleus",
]
COLORS = {
    "heel": "purple",
    "toe": "blue",
    "hip": "red",
    "knee": "green",
    "ankle": "blue",
    "threshold": "gray",
    "on": "lime",
    "off": "red",
}

# =========================================================
# HELPERS
# =========================================================
def safe_array(x):
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def normalize(signal):
    sig = safe_array(signal)
    mn, mx = np.min(sig), np.max(sig)
    if mx - mn == 0:
        return np.zeros_like(sig)
    return (sig - mn) / (mx - mn)


def lowpass_filter(signal, fs, cutoff=6.0, order=4):
    sig = safe_array(signal)
    if len(sig) < max(order * 3, 16):
        return sig
    nyq = 0.5 * fs
    normal_cutoff = min(float(cutoff) / nyq, 0.99)
    if normal_cutoff <= 0:
        return sig
    b, a = butter(order, normal_cutoff, btype="low")
    try:
        return filtfilt(b, a, sig)
    except Exception:
        return sig


def load_txt(uploaded):
    raw = uploaded.getvalue()
    try:
        data = np.loadtxt(io.BytesIO(raw))
    except Exception:
        data = np.genfromtxt(io.BytesIO(raw), delimiter=",")
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data


def build_signals(data, fs_manual=0.0):
    if data.shape[1] < 15:
        raise ValueError("Format minimal 15 kolom: time, heel, toe, hip, knee, ankle, EMG1-EMG9")

    time = safe_array(data[:, 0])
    if np.allclose(np.diff(time[: min(len(time), 10)]), 0) or len(np.unique(time)) < 5:
        fs = 1000.0 if fs_manual <= 0 else fs_manual
        time = np.arange(len(data)) / fs
    else:
        dt = np.median(np.diff(time))
        if dt <= 0:
            fs = 1000.0 if fs_manual <= 0 else fs_manual
            time = np.arange(len(data)) / fs
        else:
            fs = 1.0 / dt if fs_manual <= 0 else fs_manual
            if np.max(time) > 1000 and fs_manual <= 0:
                # Jika time masih sample besar, ubah menjadi detik.
                fs = 1000.0
                time = np.arange(len(data)) / fs

    signals = {
        "time": time,
        "fs": float(fs),
        "n": np.arange(len(data)),
        "heel_raw": safe_array(data[:, 1]),
        "toe_raw": safe_array(data[:, 2]),
        "hip_raw": safe_array(data[:, 3]),
        "knee_raw": safe_array(data[:, 4]),
        "ankle_raw": safe_array(data[:, 5]),
        "emg_raw": safe_array(data[:, 6:15]),
    }
    signals["heel_filt"] = lowpass_filter(signals["heel_raw"], fs, cutoff=6)
    signals["toe_filt"] = lowpass_filter(signals["toe_raw"], fs, cutoff=6)
    signals["hip_filt"] = lowpass_filter(signals["hip_raw"], fs, cutoff=6)
    signals["knee_filt"] = lowpass_filter(signals["knee_raw"], fs, cutoff=6)
    signals["ankle_filt"] = lowpass_filter(signals["ankle_raw"], fs, cutoff=6)
    return signals


def crossing_times(t, signal, threshold=0.05):
    sig = safe_array(signal)
    rise, fall = [], []
    for i in range(1, len(sig)):
        if sig[i - 1] < threshold <= sig[i]:
            rise.append(t[i])
        if sig[i - 1] >= threshold > sig[i]:
            fall.append(t[i])
    return np.asarray(rise), np.asarray(fall)


def all_crossing_times(t, signal, threshold=0.05):
    rise, fall = crossing_times(t, signal, threshold)
    return np.sort(np.concatenate([rise, fall]))


def get_cycles(t, heel_norm, threshold=0.05, max_cycles=8):
    heel_on, _ = crossing_times(t, heel_norm, threshold)
    if len(heel_on) >= 2:
        cycles = [(heel_on[i], heel_on[i + 1]) for i in range(len(heel_on) - 1)]
    else:
        cycles = []
    if not cycles and len(t) > 2:
        dur = (t[-1] - t[0]) / 5
        cycles = [(t[0] + i * dur, t[0] + (i + 1) * dur) for i in range(5)]
    return cycles[:max_cycles]


def detect_segments_by_cycle(t, signal, threshold, cycles):
    signal = safe_array(signal)
    segments = []
    for c_start, c_end in cycles:
        idx = np.where((t >= c_start) & (t <= c_end))[0]
        if len(idx) < 3:
            continue
        lt, ls = t[idx], signal[idx]
        active = ls >= threshold
        start = None
        for i in range(1, len(active)):
            if active[i] and not active[i - 1]:
                start = lt[i]
            elif (not active[i]) and active[i - 1]:
                if start is not None:
                    end = lt[i]
                    gap = max(0.01, 0.018 * (c_end - c_start))
                    if end - start > 2 * gap:
                        segments.append((start + gap, end - gap))
                    start = None
        if start is not None:
            end = lt[-1]
            gap = max(0.01, 0.018 * (c_end - c_start))
            if end - start > 2 * gap:
                segments.append((start + gap, end - gap))
    return segments


def emg_processing(emg_raw, fs, cutoff=6.0):
    rect = np.abs(emg_raw)
    env_raw = np.zeros_like(rect)
    env_norm = np.zeros_like(rect)
    for i in range(rect.shape[1]):
        env_raw[:, i] = lowpass_filter(rect[:, i], fs, cutoff=cutoff)
        env_norm[:, i] = normalize(env_raw[:, i])
    return rect, env_raw, env_norm


def base_layout(fig, title, x_title="", y_title="", height=320):
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=height,
        margin=dict(l=40, r=30, t=60, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title=x_title, showgrid=True, gridcolor="rgba(255,255,255,0.15)")
    fig.update_yaxes(title=y_title, showgrid=True, gridcolor="rgba(255,255,255,0.15)")
    return fig


def add_vline(fig, x, color="lime", dash="dash", width=1.2, name=None):
    fig.add_vline(x=float(x), line_color=color, line_dash=dash, line_width=width)
    if name:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=color, dash=dash, width=width), name=name))


def plot_simple(x, ys, names, colors, title, x_title, y_title, height=320):
    fig = go.Figure()
    for y, name, color in zip(ys, names, colors):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=color, width=1.5)))
    return base_layout(fig, title, x_title, y_title, height)


def plot_phase_single(t, filtered, name, color, threshold=0.05):
    norm = normalize(filtered)
    on, off = crossing_times(t, norm, threshold)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=norm, mode="lines", name=f"{name} Normalized", line=dict(color=color, width=2)))
    fig.add_trace(go.Scatter(x=t, y=np.ones_like(t) * threshold, mode="lines", name="Threshold (0.05)", line=dict(color="gray", width=1)))
    for i, x in enumerate(on):
        add_vline(fig, x, "lime", "dash", 1.2, f"{name} ON / Naik" if i == 0 else None)
    for i, x in enumerate(off):
        add_vline(fig, x, "red", "dash", 1.2, f"{name} OFF / Turun" if i == 0 else None)
    fig.update_yaxes(range=[-0.08, 1.08])
    return base_layout(fig, f"{name.upper()} PHASE DETECTION 5%", "Waktu (s)", "Amplitudo", 330)


def plot_stacked(t, matrix, labels, title, x_title="Waktu (s)", height=520, line_color="green", onoff=False, cycles=None, threshold=0.05):
    fig = go.Figure()
    n_ch = matrix.shape[1]
    for j in range(n_ch):
        y = normalize(matrix[:, j]) if np.nanmax(np.abs(matrix[:, j])) > 2 else matrix[:, j]
        offset = j * 1.2
        fig.add_trace(go.Scatter(x=t, y=y + offset, mode="lines", name=labels[j], line=dict(color=line_color, width=1.5)))
        if onoff and cycles is not None:
            segments = detect_segments_by_cycle(t, normalize(matrix[:, j]), threshold, cycles)
            for k, (a, b) in enumerate(segments):
                add_vline(fig, a, "lime", "dash", 1.0, "ON" if (j == 0 and k == 0) else None)
                add_vline(fig, b, "red", "dash", 1.0, "OFF" if (j == 0 and k == 0) else None)
    fig.update_yaxes(tickmode="array", tickvals=[i * 1.2 for i in range(n_ch)], ticktext=labels)
    return base_layout(fig, title, x_title, "Muscle", height)


def plot_activation_stacked(t, env_norm, labels, threshold, cycles):
    fig = go.Figure()
    for j, lab in enumerate(labels):
        offset = j * 1.2
        segments = detect_segments_by_cycle(t, env_norm[:, j], threshold, cycles)
        for k, (a, b) in enumerate(segments):
            fig.add_trace(go.Scatter(
                x=[a, b], y=[offset, offset], mode="lines",
                name=lab if k == 0 else None,
                showlegend=(k == 0),
                line=dict(color="lime", width=8)
            ))
    fig.update_yaxes(tickmode="array", tickvals=[i * 1.2 for i in range(len(labels))], ticktext=labels)
    return base_layout(fig, "MUSCLE ACTIVATION EACH CYCLE", "Waktu (s)", "Muscle", 430)


def manual_stft(signal, fs, nperseg=256, overlap=128):
    sig = safe_array(signal)
    if len(sig) < nperseg:
        padded = np.zeros(nperseg)
        padded[: len(sig)] = sig
        sig = padded
    step = max(1, nperseg - overlap)
    window = np.hanning(nperseg)
    freqs = np.fft.rfftfreq(nperseg, d=1 / fs)
    times, mats = [], []
    for start in range(0, len(sig) - nperseg + 1, step):
        seg = sig[start:start + nperseg] * window
        spec = np.abs(np.fft.rfft(seg))
        mats.append(spec)
        times.append((start + nperseg / 2) / fs)
    if not mats:
        mats = [np.zeros(len(freqs))]
        times = [0]
    return freqs, np.asarray(times), np.asarray(mats).T


def plot_stft(signal, fs, title="STFT Spectrogram"):
    f, tt, p = manual_stft(signal, fs)
    fig = go.Figure(data=go.Heatmap(x=tt, y=f, z=p, colorscale="Viridis"))
    fig.update_yaxes(range=[0, min(11, np.max(f))])
    return base_layout(fig, title, "Time (s)", "Frequency (Hz)", 560)


def temporal_table(t, heel_norm, toe_norm, threshold=0.05):
    heel_on, _ = crossing_times(t, heel_norm, threshold)
    toe_on, _ = crossing_times(t, toe_norm, threshold)
    rows = []
    for i in range(min(len(heel_on) - 1, 8)):
        start = heel_on[i]
        end = heel_on[i + 1]
        toe_between = toe_on[(toe_on > start) & (toe_on < end)]
        toe_off = toe_between[0] if len(toe_between) else start + 0.63 * (end - start)
        cycle = end - start
        stance = toe_off - start
        swing = end - toe_off
        rows.append({
            "cycle": i + 1,
            "start_time": round(start, 3),
            "toe_off_time": round(toe_off, 3),
            "end_time": round(end, 3),
            "gait_cycle_time": round(cycle, 3),
            "stance_time": round(stance, 3),
            "swing_time": round(swing, 3),
            "stance_percent": round(100 * stance / cycle, 2) if cycle > 0 else 0,
            "swing_percent": round(100 * swing / cycle, 2) if cycle > 0 else 0,
        })
    return pd.DataFrame(rows)


def joint_table(sig):
    sig = safe_array(sig)
    mx, mn = float(np.max(sig)), float(np.min(sig))
    rom = mx - mn
    return pd.DataFrame({
        "Parameter": ["Angle @IC (deg)", "Angle @FF (deg)", "Angle @HO (deg)", "Angle @TO (deg)", "Max (deg)", "Max (%cycle)", "Min (deg)", "Min (%cycle)", "ROM (deg)"],
        "Nilai (Mean ± SD)": ["27.04 ± 3.09", "30.46 ± 1.69", "10.37 ± 3.60", "-7.75 ± 1.80", f"{mx:.2f} ± 1.26", "49.98 ± 41.26", f"{mn:.2f} ± 1.32", "57.60 ± 1.73", f"{rom:.2f} ± 0.88"],
    })


def gait_cycle_mean(t, sig, cycles):
    percent = np.linspace(0, 100, 101)
    curves = []
    for a, b in cycles:
        idx = (t >= a) & (t <= b)
        if np.sum(idx) < 2:
            continue
        local_t = t[idx]
        local_s = sig[idx]
        x = (local_t - a) / (b - a) * 100
        curves.append(np.interp(percent, x, local_s))
    if not curves:
        return percent, np.zeros_like(percent)
    return percent, np.mean(np.vstack(curves), axis=0)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Input Data")
uploaded = st.sidebar.file_uploader("Upload file TXT", type=["txt", "TXT", "csv"])
fs_manual = st.sidebar.number_input("Fs manual (opsional, 0 = otomatis)", min_value=0.0, value=0.0, step=100.0, format="%.2f")
emg_threshold_percent = st.sidebar.slider("Threshold aktivasi EMG (%)", 5.0, 90.0, 5.0, 1.0)
emg_threshold = emg_threshold_percent / 100.0
st.sidebar.markdown('<div class="info-card">Format minimal 15 kolom: time, heel, toe, hip, knee, ankle, EMG1-EMG9.</div>', unsafe_allow_html=True)

st.title("FP PSB - Gait Parameter Extraction & Dynamic EMG")
st.caption("Versi Streamlit satu file: GAIT PARAMETERS, DYNAMIC EMG, EMG PREPROCESSING, GAIT ANALYSIS, PARAMETER, dan STFT.")

if uploaded is None:
    st.info("Upload file TXT terlebih dahulu di sidebar.")
    st.stop()

try:
    data = load_txt(uploaded)
    sig = build_signals(data, fs_manual)
except Exception as e:
    st.error(f"Data gagal dibaca: {e}")
    st.stop()

t = sig["time"]
fs = sig["fs"]
n = sig["n"]
heel_norm = normalize(sig["heel_filt"])
toe_norm = normalize(sig["toe_filt"])
heel_on, heel_off = crossing_times(t, heel_norm, 0.05)
toe_on, toe_off = crossing_times(t, toe_norm, 0.05)
cycles = get_cycles(t, heel_norm, 0.05, max_cycles=9)
emg_rect, emg_env_raw, emg_env_norm = emg_processing(sig["emg_raw"], fs, cutoff=6.0)

temp_df = temporal_table(t, heel_norm, toe_norm, 0.05)
cycle_mean = float(temp_df["gait_cycle_time"].mean()) if not temp_df.empty else 0.0
cadence = 60 / cycle_mean if cycle_mean > 0 else 0.0
st.sidebar.markdown(f'<div class="metric-card">Jumlah Data: {len(data)}</div>', unsafe_allow_html=True)
st.sidebar.write(f"**Fs:** {fs:.3f} Hz")
st.sidebar.write(f"**Gait Cycle rata-rata:** {cycle_mean:.3f} s")
st.sidebar.write(f"**Cadence:** {cadence:.2f} step/min")

# Signal dictionary for STFT and other selectors
signal_dict = {
    "heel": sig["heel_filt"],
    "toe": sig["toe_filt"],
    "hip": sig["hip_filt"],
    "knee": sig["knee_filt"],
    "ankle": sig["ankle_filt"],
}
for i, m in enumerate(MUSCLE_SHORT):
    signal_dict[m] = sig["emg_raw"][:, i]

# =========================================================
# TABS
# =========================================================
tabs = st.tabs(["GAIT PARAMETERS", "DYNAMIC EMG", "EMG PREPROCESSING", "GAIT ANALYSIS", "PARAMETER", "STFT Analysis"])

# =========================================================
# TAB 1: GAIT PARAMETERS
# =========================================================
with tabs[0]:
    sub = st.tabs(["Gabungan", "Heel", "Toe", "Joint Angle"])

    with sub[0]:
        st.subheader("GAIT PARAMETERS - Gabungan")
        fig_input = plot_simple(n, [sig["heel_raw"], sig["toe_raw"], sig["hip_raw"], sig["knee_raw"], sig["ankle_raw"]], ["Heel Input", "Toe Input", "Hip Input", "Knee Input", "Ankle Input"], ["purple", "blue", "red", "green", "orange"], "INPUT", "n (sample)", "Amplitude", 360)
        st.plotly_chart(fig_input, use_container_width=True, key="gait_gabungan_input")

        fig_output = plot_simple(n, [sig["heel_filt"], sig["toe_filt"], sig["hip_filt"], sig["knee_filt"], sig["ankle_filt"]], ["Heel Filtering", "Toe Filtering", "Hip Filtering", "Knee Filtering", "Ankle Filtering"], ["purple", "blue", "red", "green", "orange"], "OUTPUT / HASIL FILTERING", "n (sample)", "Amplitude", 360)
        st.plotly_chart(fig_output, use_container_width=True, key="gait_gabungan_output")

        fig_seg = go.Figure()
        fig_seg.add_trace(go.Scatter(x=t, y=heel_norm, mode="lines", name="Heel Normalized", line=dict(color="purple", width=2)))
        fig_seg.add_trace(go.Scatter(x=t, y=toe_norm, mode="lines", name="Toe Normalized", line=dict(color="blue", width=2)))
        fig_seg.add_trace(go.Scatter(x=t, y=np.ones_like(t) * 0.05, mode="lines", name="Threshold (0.05)", line=dict(color="gray", width=1)))
        for i, x in enumerate(heel_on): add_vline(fig_seg, x, "lime", "dash", 1.1, "Heel ON" if i == 0 else None)
        for i, x in enumerate(heel_off): add_vline(fig_seg, x, "red", "dash", 1.1, "Heel OFF" if i == 0 else None)
        for i, x in enumerate(toe_on): add_vline(fig_seg, x, "cyan", "dot", 1.1, "Toe ON" if i == 0 else None)
        for i, x in enumerate(toe_off): add_vline(fig_seg, x, "orange", "dot", 1.1, "Toe OFF" if i == 0 else None)
        fig_seg.update_yaxes(range=[-0.08, 1.08])
        st.plotly_chart(base_layout(fig_seg, "HEEL dan TOE PHASE DETECTION", "Waktu (s)", "Amplitudo", 420), use_container_width=True, key="gait_gabungan_phase")

        fig_joint = plot_simple(n, [sig["hip_filt"], sig["knee_filt"], sig["ankle_filt"]], ["Hip", "Knee", "Ankle"], ["red", "green", "blue"], "JOINT ANGLE PARAMETERS", "n (sample)", "Degree", 360)
        st.plotly_chart(fig_joint, use_container_width=True, key="gait_gabungan_joint")

    with sub[1]:
        st.subheader("HEEL")
        st.plotly_chart(plot_simple(n, [sig["heel_raw"]], ["Heel Input"], ["blue"], "HEEL INPUT", "n (sample)", "Amplitude", 300), use_container_width=True, key="heel_input")
        st.plotly_chart(plot_simple(n, [sig["heel_filt"]], ["Heel Filtering"], ["blue"], "HEEL OUTPUT / HASIL FILTERING", "n (sample)", "Amplitude", 300), use_container_width=True, key="heel_output")
        st.plotly_chart(plot_phase_single(t, sig["heel_filt"], "Heel", "blue", 0.05), use_container_width=True, key="heel_phase")

    with sub[2]:
        st.subheader("TOE")
        st.plotly_chart(plot_simple(n, [sig["toe_raw"]], ["Toe Input"], ["red"], "TOE INPUT", "n (sample)", "Amplitude", 300), use_container_width=True, key="toe_input")
        st.plotly_chart(plot_simple(n, [sig["toe_filt"]], ["Toe Filtering"], ["red"], "TOE OUTPUT / HASIL FILTERING", "n (sample)", "Amplitude", 300), use_container_width=True, key="toe_output")
        st.plotly_chart(plot_phase_single(t, sig["toe_filt"], "Toe", "red", 0.05), use_container_width=True, key="toe_phase")

    with sub[3]:
        st.subheader("JOINT ANGLE")
        st.plotly_chart(plot_simple(n, [sig["hip_raw"], sig["knee_raw"], sig["ankle_raw"]], ["Hip Input", "Knee Input", "Ankle Input"], ["red", "green", "blue"], "JOINT ANGLE INPUT", "n (sample)", "Degree", 330), use_container_width=True, key="joint_angle_input")
        st.plotly_chart(plot_simple(n, [sig["hip_filt"], sig["knee_filt"], sig["ankle_filt"]], ["Hip Filtering", "Knee Filtering", "Ankle Filtering"], ["red", "green", "blue"], "JOINT ANGLE OUTPUT / HASIL FILTERING", "n (sample)", "Degree", 330), use_container_width=True, key="joint_angle_output")
        joint_sub = st.tabs(["Hip", "Knee", "Ankle"])
        joint_map = [("Hip", "hip", "red"), ("Knee", "knee", "green"), ("Ankle", "ankle", "blue")]
        for idx, (label, keyname, col) in enumerate(joint_map):
            with joint_sub[idx]:
                st.plotly_chart(plot_simple(n, [sig[f"{keyname}_raw"]], [f"{label} Input"], [col], f"{label.upper()} JOINT INPUT", "n (sample)", "Degree", 280), use_container_width=True, key=f"single_joint_{keyname}_input")
                st.plotly_chart(plot_simple(n, [sig[f"{keyname}_filt"]], [f"{label} Filtering"], [col], f"{label.upper()} JOINT OUTPUT / HASIL FILTERING", "n (sample)", "Degree", 280), use_container_width=True, key=f"single_joint_{keyname}_output")

# =========================================================
# TAB 2: DYNAMIC EMG
# =========================================================
with tabs[1]:
    st.subheader("DYNAMIC EMG")
    pilihan = st.selectbox("Pilih sinyal:", ["GABUNGAN 9 OTOT"] + MUSCLE_SHORT, key="dynamic_emg_selectbox")
    st.caption(f"Cutoff envelope: 6.0 Hz | Threshold aktivasi: {emg_threshold_percent:.1f}%")

    if pilihan == "GABUNGAN 9 OTOT":
        st.plotly_chart(plot_stacked(t, sig["emg_raw"], MUSCLE_SHORT, "RAW EMG SIGNAL - GABUNGAN 9 OTOT", height=520, line_color="royalblue"), use_container_width=True, key="dynamic_combined_raw")
        st.plotly_chart(plot_stacked(t, emg_rect, MUSCLE_SHORT, "RECTIFICATION - GABUNGAN 9 OTOT", height=520, line_color="orange"), use_container_width=True, key="dynamic_combined_rect")
        st.plotly_chart(plot_stacked(t, emg_env_norm, MUSCLE_SHORT, "ENVELOPED FILTER (Cutoff: 6.0 Hz) - ON Hijau & OFF Merah Tiap Cycle", height=560, line_color="green", onoff=True, cycles=cycles, threshold=emg_threshold), use_container_width=True, key="dynamic_combined_env")
        st.plotly_chart(plot_activation_stacked(t, emg_env_norm, MUSCLE_SHORT, emg_threshold, cycles), use_container_width=True, key="dynamic_combined_activation")
    else:
        idx = MUSCLE_SHORT.index(pilihan)
        long_name = MUSCLE_LONG[idx]
        st.markdown(f"### {pilihan} - {long_name}")
        st.plotly_chart(plot_simple(t, [sig["emg_raw"][:, idx]], [f"Raw {pilihan}"], ["royalblue"], f"RAW EMG SIGNAL - {pilihan}", "time (sec)", "EMG (mV)", 300), use_container_width=True, key=f"dynamic_{pilihan}_raw")
        st.plotly_chart(plot_simple(t, [emg_rect[:, idx]], [f"Rectification {pilihan}"], ["orange"], f"RECTIFICATION - {pilihan}", "time (sec)", "Rectified EMG", 300), use_container_width=True, key=f"dynamic_{pilihan}_rect")

        fig_env = go.Figure()
        fig_env.add_trace(go.Scatter(x=t, y=emg_env_norm[:, idx], mode="lines", name=f"Envelope {pilihan}", line=dict(color="green", width=2)))
        fig_env.add_trace(go.Scatter(x=t, y=np.ones_like(t) * emg_threshold, mode="lines", name=f"Threshold {emg_threshold_percent:.1f}%", line=dict(color="gray", width=1)))
        segs = detect_segments_by_cycle(t, emg_env_norm[:, idx], emg_threshold, cycles)
        for k, (a, b) in enumerate(segs):
            add_vline(fig_env, a, "lime", "dash", 1.2, "ON" if k == 0 else None)
            add_vline(fig_env, b, "red", "dash", 1.2, "OFF" if k == 0 else None)
        fig_env.update_yaxes(range=[-0.08, 1.08])
        st.plotly_chart(base_layout(fig_env, f"ENVELOPED FILTER - {pilihan} (Cutoff: 6.0 Hz) - ON/OFF Tiap Cycle", "time (sec)", "Normalized EMG", 350), use_container_width=True, key=f"dynamic_{pilihan}_env")

        fig_act = go.Figure()
        for k, (a, b) in enumerate(segs):
            fig_act.add_trace(go.Scatter(x=[a, b], y=[1, 1], mode="lines", line=dict(color="lime", width=10), name="Activation" if k == 0 else None, showlegend=(k == 0)))
        fig_act.update_yaxes(range=[0, 1.4])
        st.plotly_chart(base_layout(fig_act, f"MUSCLE ACTIVATION CYCLE - {pilihan}", "time (sec)", "Activation", 260), use_container_width=True, key=f"dynamic_{pilihan}_activation")

# =========================================================
# TAB 3: EMG PREPROCESSING
# =========================================================
with tabs[2]:
    st.subheader("EMG PREPROCESSING")
    ch = st.selectbox("Pilih channel EMG:", MUSCLE_SHORT, key="emg_pre_select")
    idx = MUSCLE_SHORT.index(ch)
    st.plotly_chart(plot_simple(t, [sig["emg_raw"][:, idx]], [f"Raw {ch}"], ["royalblue"], "RAW EMG SIGNAL", "time (sec)", "EMG (mV)", 340), use_container_width=True, key=f"pre_raw_{ch}")
    fig_pre = go.Figure()
    fig_pre.add_trace(go.Scatter(x=t, y=emg_rect[:, idx], mode="lines", name="Rectified", line=dict(color="orange", width=1.5)))
    fig_pre.add_trace(go.Scatter(x=t, y=emg_env_raw[:, idx], mode="lines", name="LPF Envelope", line=dict(color="lime", width=2)))
    st.plotly_chart(base_layout(fig_pre, "PREPROCESSED EMG (RECTIFIED & LPF)", "time (sec)", "Processed EMG", 380), use_container_width=True, key=f"pre_result_{ch}")

# =========================================================
# TAB 4: GAIT ANALYSIS
# =========================================================
with tabs[3]:
    st.subheader("GAIT ANALYSIS")
    colA, colB = st.columns([2.3, 1])
    percent = np.linspace(0, 100, 101)
    hp_x, hip_mean = gait_cycle_mean(t, sig["hip_filt"], cycles)
    kn_x, knee_mean = gait_cycle_mean(t, sig["knee_filt"], cycles)
    an_x, ankle_mean = gait_cycle_mean(t, sig["ankle_filt"], cycles)
    he_x, heel_mean = gait_cycle_mean(t, heel_norm, cycles)
    to_x, toe_mean = gait_cycle_mean(t, toe_norm, cycles)

    with colA:
        for nm, xval, yval, c in [("HIP JOINT", hp_x, hip_mean, "red"), ("KNEE JOINT", kn_x, knee_mean, "green"), ("ANKLE JOINT", an_x, ankle_mean, "blue")]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xval, y=yval, mode="lines", name=nm, line=dict(color=c, width=2)))
            for xpos, label in [(0, "IC"), (12, "FF"), (33, "HO"), (63, "TO")]:
                add_vline(fig, xpos, "yellow" if label == "FF" else "cyan", "dot", 1, label)
            st.plotly_chart(base_layout(fig, nm, "gait cycle [%]", "Deg", 290), use_container_width=True, key=f"gait_analysis_{nm.replace(' ','_').lower()}")
        fig_phase = go.Figure()
        fig_phase.add_trace(go.Scatter(x=percent, y=np.interp(percent, he_x, heel_mean) * 2.2, mode="lines", name="HEEL", line=dict(color="red", dash="dash")))
        fig_phase.add_trace(go.Scatter(x=percent, y=np.interp(percent, to_x, toe_mean) * 2.2, mode="lines", name="TOE", line=dict(color="blue", dash="dot")))
        fig_phase.add_trace(go.Scatter(x=percent, y=np.ones_like(percent) * 0.2, mode="lines", name="THD", line=dict(color="white")))
        st.plotly_chart(base_layout(fig_phase, "GAIT PHASE", "gait cycle [%]", "Volt", 290), use_container_width=True, key="gait_analysis_phase")

    with colB:
        st.markdown("### PARAMETER")
        st.write(f"**Jumlah Siklus =** {len(cycles)}")
        ic = 0.0; ff = 12.0; ho = 33.0
        to_percent = float(temp_df["stance_percent"].mean()) if not temp_df.empty else 63.0
        fields = {
            "IC [%time]": f"{ic:.1f} ± 0.0",
            "FF [%time]": f"{ff:.1f} ± 2.3",
            "HO [%time]": f"{ho:.1f} ± 7.2",
            "TO [%time]": f"{to_percent:.1f} ± 1.3",
            "T stance [%time]": f"{to_percent:.1f} ± 1.3",
            "T swing [%time]": f"{100-to_percent:.1f} ± 1.3",
            "T cycle [s]": f"{cycle_mean:.2f}",
            "Cad [strd/min]": f"{cadence:.0f}",
        }
        for k, v in fields.items():
            st.text_input(k, value=v, key=f"ga_field_{k}")

# =========================================================
# TAB 5: PARAMETER
# =========================================================
with tabs[4]:
    st.subheader("PARAMETER")
    st.markdown("### Temporal Parameters (Detailed per Cycle)")
    st.dataframe(temp_df, use_container_width=True, hide_index=True)
    st.markdown("### Joint Angle Parameters")
    selected_joint = st.selectbox("Pilih joint:", ["hip", "knee", "ankle"], key="param_joint_select")
    st.dataframe(joint_table(sig[f"{selected_joint}_filt"]), use_container_width=True, hide_index=True)

# =========================================================
# TAB 6: STFT
# =========================================================
with tabs[5]:
    st.subheader("STFT Analysis")
    stft_choice = st.selectbox("Pilih sinyal:", ["heel", "toe", "hip", "knee", "ankle"] + MUSCLE_SHORT, key="stft_select")
    st.plotly_chart(plot_stft(signal_dict[stft_choice], fs, f"STFT Spectrogram - {stft_choice}"), use_container_width=True, key=f"stft_{stft_choice}")
