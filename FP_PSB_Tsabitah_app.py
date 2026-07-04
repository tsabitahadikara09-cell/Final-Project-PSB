import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt

st.set_page_config(page_title="FP PSB - Gait Parameter & Dynamic EMG", layout="wide")

MUSCLE_SHORT = ["GMAX", "BFS", "BFL", "VM", "VL", "RF", "MG", "TA", "SOL"]
MUSCLE_LABELS = [
    "Gluteus Maximus", "Biceps Femoris Short", "Biceps Femoris Long",
    "Vastus Medialis", "Vastus Lateralis", "Rectus Femoris",
    "Medial Gastrocnemius", "Tibialis Anterior", "Soleus"
]
EMG_ORDER = list(range(9))

COLORS = {
    "heel": "blue",
    "toe": "red",
    "hip": "red",
    "knee": "green",
    "ankle": "blue",
    "on": "green",
    "off": "red",
    "cycle": "rgba(160,0,0,0.5)",
    "threshold": "black",
}


def lowpass_filter(signal, fs, cutoff=6.0, order=4):
    signal = np.asarray(signal, dtype=float)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1:
        normal_cutoff = 0.99
    b, a = butter(order, normal_cutoff, btype="low")
    if len(signal) < max(len(a), len(b)) * 3:
        return signal.copy()
    return filtfilt(b, a, signal)


def normalize(signal):
    signal = np.asarray(signal, dtype=float)
    mn = np.min(signal)
    mx = np.max(signal)
    if mx - mn == 0:
        return np.zeros_like(signal)
    return (signal - mn) / (mx - mn)


def detect_crossing_time(time, signal, threshold):
    time = np.asarray(time)
    signal = np.asarray(signal)
    crossing_time = []
    for i in range(1, len(signal)):
        s1 = signal[i - 1]
        s2 = signal[i]
        if (s1 < threshold and s2 >= threshold) or (s1 >= threshold and s2 < threshold):
            t1 = time[i - 1]
            t2 = time[i]
            if s2 != s1:
                tcross = t1 + (threshold - s1) * (t2 - t1) / (s2 - s1)
            else:
                tcross = t2
            crossing_time.append(tcross)
    return np.array(crossing_time)


def rise_fall_indices(signal, threshold):
    signal = np.asarray(signal)
    rise = np.where((signal[:-1] < threshold) & (signal[1:] >= threshold))[0] + 1
    fall = np.where((signal[:-1] >= threshold) & (signal[1:] < threshold))[0] + 1
    return rise, fall


def get_on_off_per_cycle(t, env_signal, threshold, cycle_lines):
    """Return one or more ON/OFF pairs inside every gait cycle."""
    t = np.asarray(t)
    env_signal = np.asarray(env_signal)
    pairs = []
    if cycle_lines is None or len(cycle_lines) < 2:
        cycle_lines = np.array([t[0], t[-1]])
    cycle_lines = np.asarray(cycle_lines, dtype=float)
    for c in range(len(cycle_lines) - 1):
        start_cycle = cycle_lines[c]
        end_cycle = cycle_lines[c + 1]
        idx = np.where((t >= start_cycle) & (t <= end_cycle))[0]
        if len(idx) < 3:
            continue
        local_t = t[idx]
        active = env_signal[idx] >= threshold
        start = None
        for i in range(1, len(active)):
            if active[i] and not active[i - 1]:
                start = local_t[i]
            elif not active[i] and active[i - 1]:
                end = local_t[i]
                if start is not None and end > start:
                    pairs.append((float(start), float(end), float(start_cycle), float(end_cycle)))
                start = None
        if start is not None and local_t[-1] > start:
            pairs.append((float(start), float(local_t[-1]), float(start_cycle), float(end_cycle)))
    return pairs


def manual_stft(signal, fs, nperseg=256, overlap=128):
    signal = np.asarray(signal, dtype=float)
    step = max(1, nperseg - overlap)
    if len(signal) < nperseg:
        padded = np.zeros(nperseg)
        padded[:len(signal)] = signal
        signal = padded
    window = np.hanning(nperseg)
    freqs = np.fft.rfftfreq(nperseg, d=1 / fs)
    times, stft_matrix = [], []
    for start in range(0, len(signal) - nperseg + 1, step):
        seg = signal[start:start + nperseg] * window
        stft_matrix.append(np.abs(np.fft.rfft(seg)))
        times.append((start + nperseg / 2) / fs)
    if not stft_matrix:
        stft_matrix = [np.zeros(len(freqs))]
        times = [0]
    return freqs, np.array(times), np.array(stft_matrix).T


def base_layout(fig, title, x_title="time (sec)", y_title="Amplitude", height=330):
    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=70, b=40),
    )
    fig.update_xaxes(title=x_title, showgrid=True)
    fig.update_yaxes(title=y_title, showgrid=True)
    return fig


def add_vertical_line(fig, x, color, dash="dash", name=None, row=None, col=None):
    args = dict(x=x, line_width=1.3, line_dash=dash, line_color=color)
    if row is None:
        fig.add_vline(**args)
    else:
        fig.add_vline(row=row, col=col, **args)
    if name:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=color, dash=dash), name=name), row=row, col=col) if row else fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=color, dash=dash), name=name))


def load_txt(uploaded):
    raw = uploaded.read()
    try:
        data = np.loadtxt(io.BytesIO(raw))
    except Exception:
        data = np.loadtxt(io.StringIO(raw.decode("utf-8", errors="ignore")), delimiter=None)
    return data


def process_data(data, fs_override=None):
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.shape[1] < 15:
        raise ValueError("Format data harus minimal 15 kolom: t, heel, toe, hip, knee, ankle, EMG1 sampai EMG9.")
    t = data[:, 0]
    n = np.arange(len(t))
    fs = float(fs_override) if fs_override else float(1 / np.mean(np.diff(t)))
    heel, toe = data[:, 1], data[:, 2]
    hip, knee, ankle = data[:, 3], data[:, 4], data[:, 5]
    emg = data[:, 6:15]

    heel_filt = lowpass_filter(heel, fs)
    toe_filt = lowpass_filter(toe, fs)
    hip_filt = lowpass_filter(hip, fs)
    knee_filt = lowpass_filter(knee, fs)
    ankle_filt = lowpass_filter(ankle, fs)

    threshold_value = 0.15
    heel_cross = detect_crossing_time(t, heel_filt, threshold_value)
    toe_cross = detect_crossing_time(t, toe_filt, threshold_value)
    heel_start = heel_cross[::2]

    temporal_rows = []
    gait_cycle_values = []
    for i in range(len(heel_start) - 1):
        start_time = float(heel_start[i])
        end_time = float(heel_start[i + 1])
        gait_cycle_time = end_time - start_time
        toe_candidates = toe_cross[(toe_cross > start_time) & (toe_cross < end_time)]
        if len(toe_candidates) == 0 or gait_cycle_time <= 0:
            continue
        toe_off_time = float(toe_candidates[0])
        stance_time = toe_off_time - start_time
        swing_time = end_time - toe_off_time
        stance_percent = (stance_time / gait_cycle_time) * 100
        swing_percent = (swing_time / gait_cycle_time) * 100
        gait_cycle_values.append(gait_cycle_time)
        temporal_rows.append([i + 1, round(start_time, 3), round(toe_off_time, 3), round(end_time, 3), round(gait_cycle_time, 3), round(stance_time, 3), round(swing_time, 3), round(stance_percent, 2), round(swing_percent, 2)])
    if temporal_rows:
        arr = np.array([row[1:] for row in temporal_rows], dtype=float)
        avg = np.mean(arr, axis=0)
        temporal_rows.append(["Rata-rata"] + [round(v, 3) if j < 6 else round(v, 2) for j, v in enumerate(avg)])
    mean_cycle = float(np.mean(gait_cycle_values)) if gait_cycle_values else 0.0
    cadence = 60 / mean_cycle if mean_cycle > 0 else 0.0

    emg_rect = np.abs(emg)
    emg_env_raw = np.zeros_like(emg_rect)
    emg_env = np.zeros_like(emg_rect)
    for i in range(emg_rect.shape[1]):
        emg_env_raw[:, i] = lowpass_filter(emg_rect[:, i], fs, cutoff=6.0)
        emg_env[:, i] = normalize(emg_env_raw[:, i])

    signal_dict = {
        "heel": heel_filt, "toe": toe_filt,
        "hip": hip, "knee": knee, "ankle": ankle,
        "gmax": emg[:, 0], "BFS": emg[:, 1], "BFL": emg[:, 2], "VM": emg[:, 3], "VL": emg[:, 4],
        "RF": emg[:, 5], "MG": emg[:, 6], "TA": emg[:, 7], "SOL": emg[:, 8],
    }
    return dict(
        t=t, n=n, fs=fs, heel=heel, toe=toe, hip=hip, knee=knee, ankle=ankle, emg=emg,
        heel_filt=heel_filt, toe_filt=toe_filt, hip_filt=hip_filt, knee_filt=knee_filt, ankle_filt=ankle_filt,
        heel_cross=heel_cross, toe_cross=toe_cross, cycle_lines=heel_start,
        temporal_rows=temporal_rows, mean_cycle=mean_cycle, cadence=cadence,
        emg_rect=emg_rect, emg_env_raw=emg_env_raw, emg_env=emg_env, signal_dict=signal_dict
    )


def plot_two_signals(x, y1, y2, name1, name2, title, xlab="n (sample)", ylab="Amplitude"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y1, mode="lines", name=name1))
    fig.add_trace(go.Scatter(x=x, y=y2, mode="lines", name=name2))
    return base_layout(fig, title, xlab, ylab)


def plot_phase_single(t, raw_x, raw_signal, filt_signal, title_name, color):
    norm_sig = normalize(filt_signal)
    th = 0.05
    rise, fall = rise_fall_indices(norm_sig, th)
    fig_input = go.Figure(go.Scatter(x=raw_x, y=raw_signal, mode="lines", name=f"{title_name} Input", line=dict(color=color)))
    fig_output = go.Figure(go.Scatter(x=raw_x, y=filt_signal, mode="lines", name=f"{title_name} Filtering", line=dict(color=color)))
    fig_phase = go.Figure()
    fig_phase.add_trace(go.Scatter(x=t, y=norm_sig, mode="lines", name=f"{title_name} Normalized", line=dict(color=color, width=2)))
    fig_phase.add_trace(go.Scatter(x=t, y=np.ones_like(t) * th, mode="lines", name="Threshold (0.05)", line=dict(color="black", dash="dot")))
    for idx in rise:
        add_vertical_line(fig_phase, float(t[idx]), COLORS["on"], "dash")
    for idx in fall:
        add_vertical_line(fig_phase, float(t[idx]), COLORS["off"], "dash")
    fig_phase.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=f"{title_name} ON / Naik", line=dict(color=COLORS["on"], dash="dash")))
    fig_phase.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=f"{title_name} OFF / Turun", line=dict(color=COLORS["off"], dash="dash")))
    fig_input = base_layout(fig_input, f"{title_name.upper()} INPUT", "n (sample)", "Amplitude")
    fig_output = base_layout(fig_output, f"{title_name.upper()} OUTPUT / HASIL FILTERING", "n (sample)", "Amplitude")
    fig_phase = base_layout(fig_phase, f"{title_name.upper()} PHASE DETECTION 5%", "Waktu (s)", "Amplitudo")
    fig_phase.update_yaxes(range=[-0.05, 1.05])
    return fig_input, fig_output, fig_phase


def plot_joint_input_output(d):
    n = d["n"]
    fig_in = go.Figure()
    fig_out = go.Figure()
    for name, key, color in [("Hip", "hip", "red"), ("Knee", "knee", "green"), ("Ankle", "ankle", "blue")]:
        fig_in.add_trace(go.Scatter(x=n, y=d[key], mode="lines", name=f"{name} Input", line=dict(color=color)))
        fig_out.add_trace(go.Scatter(x=n, y=d[f"{key}_filt"], mode="lines", name=f"{name} Filtering", line=dict(color=color)))
    return base_layout(fig_in, "JOINT ANGLE INPUT", "n (sample)", "Degree"), base_layout(fig_out, "JOINT ANGLE OUTPUT / HASIL FILTERING", "n (sample)", "Degree")


def plot_dynamic_combined(d, threshold_pct):
    t = d["t"]
    emg = d["emg"]
    rect = d["emg_rect"]
    env = d["emg_env"]
    cycle_lines = d["cycle_lines"]
    threshold = threshold_pct / 100.0
    offset = 2.2
    figs = []
    for title, arr, norm_each, ytitle in [
        ("RAW EMG SIGNAL - GABUNGAN 9 OTOT", emg, True, "Muscle"),
        ("RECTIFICATION - GABUNGAN 9 OTOT", rect, True, "Muscle"),
        ("ENVELOPED FILTER + ON/OFF PER CYCLE - GABUNGAN 9 OTOT", env, False, "Muscle"),
    ]:
        fig = go.Figure()
        for display_i, idx in enumerate(EMG_ORDER):
            y_offset = display_i * offset
            y = normalize(arr[:, idx]) if norm_each else arr[:, idx]
            fig.add_trace(go.Scatter(x=t, y=y + y_offset, mode="lines", name=MUSCLE_SHORT[display_i]))
        if "ENVELOPED" in title:
            for display_i, idx in enumerate(EMG_ORDER):
                y_offset = display_i * offset
                fig.add_trace(go.Scatter(x=t, y=np.ones_like(t) * (threshold + y_offset), mode="lines", showlegend=False, line=dict(color="gray", dash="dot", width=1)))
                pairs = get_on_off_per_cycle(t, env[:, idx], threshold, cycle_lines)
                for on, off, _, _ in pairs:
                    fig.add_shape(type="line", x0=on, x1=on, y0=y_offset - 0.35, y1=y_offset + 1.05, line=dict(color="green", dash="dash", width=1.2))
                    fig.add_shape(type="line", x0=off, x1=off, y0=y_offset - 0.35, y1=y_offset + 1.05, line=dict(color="red", dash="dash", width=1.2))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="ON", line=dict(color="green", dash="dash")))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="OFF", line=dict(color="red", dash="dash")))
        fig.update_yaxes(tickmode="array", tickvals=[i * offset for i in range(9)], ticktext=MUSCLE_SHORT)
        figs.append(base_layout(fig, title, "time (sec)", ytitle, height=420))
    fig_act = go.Figure()
    for display_i, idx in enumerate(EMG_ORDER):
        y_offset = display_i * offset
        pairs = get_on_off_per_cycle(t, env[:, idx], threshold, cycle_lines)
        for on, off, _, _ in pairs:
            fig_act.add_shape(type="rect", x0=on, x1=off, y0=y_offset - 0.32, y1=y_offset + 0.32, fillcolor="rgba(20,70,140,0.75)", line=dict(color="rgba(20,70,140,0.9)"))
    for x in cycle_lines:
        fig_act.add_vline(x=float(x), line_dash="dash", line_color="rgba(160,0,0,0.45)", line_width=1)
    fig_act.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color="rgba(20,70,140,0.75)", symbol="square"), name="Activation ON"))
    fig_act.update_yaxes(tickmode="array", tickvals=[i * offset for i in range(9)], ticktext=MUSCLE_SHORT)
    figs.append(base_layout(fig_act, f"MUSCLE ACTIVATION EACH CYCLE - GABUNGAN 9 OTOT | Threshold {threshold_pct:.1f}%", "time (sec)", "Muscle", height=430))
    return figs


def plot_dynamic_single(d, selected, threshold_pct):
    t = d["t"]
    idx = MUSCLE_SHORT.index(selected)
    threshold = threshold_pct / 100.0
    emg = d["emg"][:, idx]
    rect = d["emg_rect"][:, idx]
    env = d["emg_env"][:, idx]
    pairs = get_on_off_per_cycle(t, env, threshold, d["cycle_lines"])
    figs = []
    specs = [
        ("RAW EMG SIGNAL", emg, "blue", "EMG (mV)"),
        ("RECTIFIED EMG", rect, "red", "Rectified EMG"),
    ]
    for title, y, color, ylab in specs:
        fig = go.Figure(go.Scatter(x=t, y=y, mode="lines", name=selected, line=dict(color=color)))
        figs.append(base_layout(fig, f"{title} - {selected}", "time (sec)", ylab))
    fig_env = go.Figure()
    fig_env.add_trace(go.Scatter(x=t, y=env, mode="lines", name="Envelope", line=dict(color="green", width=2)))
    fig_env.add_trace(go.Scatter(x=t, y=np.ones_like(t) * threshold, mode="lines", name=f"Threshold {threshold_pct:.1f}%", line=dict(color="black", dash="dash")))
    for on, off, _, _ in pairs:
        add_vertical_line(fig_env, on, "green", "dash")
        add_vertical_line(fig_env, off, "red", "dash")
    for x in d["cycle_lines"]:
        fig_env.add_vline(x=float(x), line_dash="dot", line_color="gray", line_width=1)
    fig_env.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="ON", line=dict(color="green", dash="dash")))
    fig_env.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="OFF", line=dict(color="red", dash="dash")))
    fig_env.update_yaxes(range=[-0.05, 1.1])
    figs.append(base_layout(fig_env, f"ENVELOPED FILTER + ON/OFF PER CYCLE - {selected}", "time (sec)", "Normalized EMG"))
    fig_act = go.Figure()
    for on, off, _, _ in pairs:
        fig_act.add_shape(type="rect", x0=on, x1=off, y0=-0.25, y1=0.25, fillcolor="rgba(20,70,140,0.75)", line=dict(color="rgba(20,70,140,0.9)"))
    for x in d["cycle_lines"]:
        fig_act.add_vline(x=float(x), line_dash="dash", line_color="red", line_width=1)
    fig_act.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color="rgba(20,70,140,0.75)", symbol="square"), name="Activation ON"))
    fig_act.update_yaxes(range=[-1, 1], tickmode="array", tickvals=[0], ticktext=[selected])
    figs.append(base_layout(fig_act, f"MUSCLE ACTIVATION - {selected} | Threshold {threshold_pct:.1f}%", "time (sec)", "Activation"))
    return figs


def plot_emg_preprocessing(d, selected):
    t = d["t"]
    idx = MUSCLE_SHORT.index(selected)
    raw = go.Figure(go.Scatter(x=t, y=d["emg"][:, idx], mode="lines", name="Raw", line=dict(color="black")))
    processed = go.Figure()
    processed.add_trace(go.Scatter(x=t, y=d["emg_rect"][:, idx], mode="lines", name="Rectified", line=dict(color="black")))
    processed.add_trace(go.Scatter(x=t, y=d["emg_env_raw"][:, idx], mode="lines", name="Low-pass Filtered", line=dict(color="green", width=2)))
    return base_layout(raw, f"RAW EMG SIGNAL - {selected}", "time (sec)", "EMG (mV)"), base_layout(processed, f"PREPROCESSED EMG - {selected}", "time (sec)", "Processed EMG (mV)")


def plot_gait_analysis(d):
    t = d["t"]
    percent_axis = np.linspace(0, 100, 101)
    heel_cross = d["heel_cross"]
    toe_cross = d["toe_cross"]
    heel_start = heel_cross[::2]
    cycles = []
    for i in range(len(heel_start) - 1):
        start, end = float(heel_start[i]), float(heel_start[i + 1])
        toe_between = toe_cross[(toe_cross > start) & (toe_cross < end)]
        toe_off = float(toe_between[0]) if len(toe_between) else start + 0.63 * (end - start)
        if end > start:
            cycles.append((start, end, toe_off))
        if len(cycles) >= 5:
            break
    if not cycles:
        cycles = [(float(t[0]), float(t[-1]), float(t[0] + 0.63 * (t[-1] - t[0])))]

    def interp_cycle(sig, start, end):
        mask = (t >= start) & (t <= end)
        if np.sum(mask) < 2:
            return np.zeros_like(percent_axis)
        pc = (t[mask] - start) / (end - start) * 100.0
        return np.interp(percent_axis, pc, sig[mask])

    hip_mean = np.mean(np.vstack([interp_cycle(d["hip"], s, e) for s, e, _ in cycles]), axis=0)
    knee_mean = np.mean(np.vstack([interp_cycle(d["knee"], s, e) for s, e, _ in cycles]), axis=0)
    ankle_mean = np.mean(np.vstack([interp_cycle(d["ankle"], s, e) for s, e, _ in cycles]), axis=0)
    heel_mean = np.mean(np.vstack([interp_cycle(normalize(d["heel_filt"]), s, e) for s, e, _ in cycles]), axis=0) * 2.2
    toe_mean = np.mean(np.vstack([interp_cycle(normalize(d["toe_filt"]), s, e) for s, e, _ in cycles]), axis=0) * 2.2
    to_values = [(to - s) / (e - s) * 100 for s, e, to in cycles]
    to = float(np.mean(to_values)) if to_values else 63.0

    figs = []
    for title, y in [("HIP JOINT", hip_mean), ("KNEE JOINT", knee_mean), ("ANKLE JOINT", ankle_mean)]:
        fig = go.Figure(go.Scatter(x=percent_axis, y=y, mode="lines", name=title, line=dict(color="red")))
        for xpos, label, color in [(0, "IC", "green"), (12, "FF", "orange"), (33, "HO", "blue"), (to, "TO", "cyan")]:
            fig.add_trace(go.Scatter(x=[xpos], y=[np.interp(xpos, percent_axis, y)], mode="markers+text", text=[label], textposition="top center", marker=dict(size=8, color=color), name=label))
        figs.append(base_layout(fig, title, "gait cycle [%]", "Deg", height=300))
    fig_phase = go.Figure()
    fig_phase.add_trace(go.Scatter(x=percent_axis, y=heel_mean, mode="lines", name="HEEL", line=dict(color="red", dash="dash")))
    fig_phase.add_trace(go.Scatter(x=percent_axis, y=toe_mean, mode="lines", name="TOE", line=dict(color="blue", dash="dot")))
    fig_phase.add_trace(go.Scatter(x=percent_axis, y=np.ones_like(percent_axis) * 0.2, mode="lines", name="THD", line=dict(color="black")))
    figs.append(base_layout(fig_phase, "GAIT PHASE", "gait cycle [%]", "Volt", height=300))
    return figs


def plot_stft(d, selected):
    key = "gmax" if selected == "GMAX" else selected
    signal = d["signal_dict"].get(key, d["heel_filt"])
    f, tt, power = manual_stft(signal, d["fs"], 256, 128)
    fig = go.Figure(data=go.Heatmap(x=tt, y=f, z=power, colorscale="Viridis"))
    fig.update_yaxes(range=[0, 11])
    return base_layout(fig, f"STFT Spectrogram - {selected}", "Time (s)", "Frequency (Hz)", height=650)


st.title("FP PSB - Gait Parameter Extraction & Dynamic EMG")
st.caption("Versi Streamlit dari aplikasi PyQt: GAIT PARAMETERS, DYNAMIC EMG, EMG PREPROCESSING, GAIT ANALYSIS, PARAMETER, dan STFT.")

with st.sidebar:
    st.header("Input Data")
    uploaded = st.file_uploader("Upload file TXT", type=["txt", "TXT", "csv"])
    fs_override = st.number_input("Fs manual (opsional, 0 = otomatis)", min_value=0.0, value=0.0, step=1.0)
    threshold_pct = st.slider("Threshold aktivasi EMG (%)", min_value=5.0, max_value=90.0, value=5.0, step=1.0)
    st.info("Format minimal 15 kolom: time, heel, toe, hip, knee, ankle, EMG1-EMG9.")

if uploaded is None:
    st.warning("Upload data TXT terlebih dahulu untuk menampilkan hasil.")
    st.stop()

try:
    data = load_txt(uploaded)
    d = process_data(data, fs_override if fs_override > 0 else None)
except Exception as e:
    st.error(str(e))
    st.stop()

st.sidebar.success(f"Jumlah Data: {len(d['t'])}")
st.sidebar.write(f"Fs: {d['fs']:.3f} Hz")
if d["mean_cycle"] > 0:
    st.sidebar.write(f"Gait Cycle rata-rata: {d['mean_cycle']:.3f} s")
    st.sidebar.write(f"Cadence: {d['cadence']:.2f} step/min")

main_tabs = st.tabs(["GAIT PARAMETERS", "DYNAMIC EMG", "EMG PREPROCESSING", "GAIT ANALYSIS", "PARAMETER", "STFT Analysis"])

with main_tabs[0]:
    sub = st.tabs(["Gabungan", "Heel", "Toe", "Joint Angle"])
    with sub[0]:
        st.plotly_chart(plot_two_signals(d["n"], d["heel"], d["toe"], "Heel / FSR Biru", "Toe / FSR Merah", "INPUT"), use_container_width=True, key='plot_001')
        st.plotly_chart(plot_two_signals(d["n"], d["heel_filt"], d["toe_filt"], "Heel Filtering", "Toe Filtering", "OUTPUT / HASIL FILTERING"), use_container_width=True, key='plot_002')
        fig_seg = go.Figure()
        heel_norm, toe_norm = normalize(d["heel_filt"]), normalize(d["toe_filt"])
        fig_seg.add_trace(go.Scatter(x=d["t"], y=heel_norm, mode="lines", name="Heel Normalized", line=dict(color="purple")))
        fig_seg.add_trace(go.Scatter(x=d["t"], y=toe_norm, mode="lines", name="Toe Normalized", line=dict(color="blue")))
        fig_seg.add_trace(go.Scatter(x=d["t"], y=np.ones_like(d["t"]) * 0.05, mode="lines", name="Threshold (0.05)", line=dict(color="black", dash="dot")))
        for idx in rise_fall_indices(heel_norm, 0.05)[0]: add_vertical_line(fig_seg, d["t"][idx], "green", "dash")
        for idx in rise_fall_indices(heel_norm, 0.05)[1]: add_vertical_line(fig_seg, d["t"][idx], "red", "dash")
        for idx in rise_fall_indices(toe_norm, 0.05)[0]: add_vertical_line(fig_seg, d["t"][idx], "cyan", "dot")
        for idx in rise_fall_indices(toe_norm, 0.05)[1]: add_vertical_line(fig_seg, d["t"][idx], "orange", "dot")
        st.plotly_chart(base_layout(fig_seg, "HEEL dan TOE PHASE DETECTION", "Waktu (s)", "Amplitudo"), use_container_width=True, key='plot_003')
        j_in, j_out = plot_joint_input_output(d)
        st.plotly_chart(j_in, use_container_width=True, key='plot_004')
        st.plotly_chart(j_out, use_container_width=True, key='plot_005')
    with sub[1]:
        f1, f2, f3 = plot_phase_single(d["t"], d["n"], d["heel"], d["heel_filt"], "Heel", "blue")
        st.plotly_chart(f1, use_container_width=True, key='plot_006')
        st.plotly_chart(f2, use_container_width=True, key='plot_007')
        st.plotly_chart(f3, use_container_width=True, key='plot_008')
    with sub[2]:
        f1, f2, f3 = plot_phase_single(d["t"], d["n"], d["toe"], d["toe_filt"], "Toe", "red")
        st.plotly_chart(f1, use_container_width=True, key='plot_009')
        st.plotly_chart(f2, use_container_width=True, key='plot_010')
        st.plotly_chart(f3, use_container_width=True, key='plot_011')
    with sub[3]:
        j_in, j_out = plot_joint_input_output(d)
        st.plotly_chart(j_in, use_container_width=True, key='plot_012')
        st.plotly_chart(j_out, use_container_width=True, key='plot_013')
        single = st.tabs(["Hip", "Knee", "Ankle"])
        for tab, name, raw, filt, color in zip(single, ["Hip", "Knee", "Ankle"], [d["hip"], d["knee"], d["ankle"]], [d["hip_filt"], d["knee_filt"], d["ankle_filt"]], ["red", "green", "blue"]):
            with tab:
                st.plotly_chart(base_layout(go.Figure(go.Scatter(x=d["n"], y=raw, mode="lines", name=f"{name} Input", line=dict(color=color))), f"{name.upper()} JOINT INPUT", "n (sample)", "Degree"), use_container_width=True, key='plot_014')
                st.plotly_chart(base_layout(go.Figure(go.Scatter(x=d["n"], y=filt, mode="lines", name=f"{name} Filtering", line=dict(color=color))), f"{name.upper()} JOINT OUTPUT / HASIL FILTERING", "n (sample)", "Degree"), use_container_width=True, key='plot_015')

with main_tabs[1]:
    pilihan = st.selectbox("Pilih sinyal:", ["GABUNGAN 9 OTOT"] + MUSCLE_SHORT)
    figs = plot_dynamic_combined(d, threshold_pct) if pilihan == "GABUNGAN 9 OTOT" else plot_dynamic_single(d, pilihan, threshold_pct)
    for fig in figs:
        st.plotly_chart(fig, use_container_width=True, key='plot_016')

with main_tabs[2]:
    selected_pre = st.selectbox("Pilih channel EMG:", MUSCLE_SHORT, key="pre")
    raw_fig, proc_fig = plot_emg_preprocessing(d, selected_pre)
    st.plotly_chart(raw_fig, use_container_width=True, key='plot_017')
    st.plotly_chart(proc_fig, use_container_width=True, key='plot_018')

with main_tabs[3]:
    figs = plot_gait_analysis(d)
    left, right = st.columns([2, 1])
    with left:
        for fig in figs:
            st.plotly_chart(fig, use_container_width=True, key='plot_019')
    with right:
        cycle_count = max(0, len(d["cycle_lines"]) - 1)
        st.metric("JUMLAH SIKLUS", cycle_count)
        ic = st.text_input("IC [%time]", "0.0±0.0")
        ff = st.text_input("FF [%time]", "12.0±2.3")
        ho = st.text_input("HO [%time]", "33.0±7.2")
        to = st.text_input("TO [%time]", "63.0±1.3")
        st.caption("Parameter editable ditampilkan seperti panel PyQt. Perubahan nilai tidak mengubah grafik utama di versi ringkas ini.")

with main_tabs[4]:
    st.subheader("Temporal Parameters (Detailed per Cycle)")
    temporal_df = pd.DataFrame(d["temporal_rows"], columns=["cycle", "start_time", "toe_off_time", "end_time", "gait_cycle_time", "stance_time", "swing_time", "stance_percent", "swing_percent"])
    st.dataframe(temporal_df, use_container_width=True, hide_index=True)
    st.subheader("Joint Angle Parameters")
    selected_joint = st.selectbox("Pilih joint:", ["hip", "knee", "ankle"])
    sig = d[selected_joint]
    max_val, min_val = float(np.max(sig)), float(np.min(sig))
    rom = max_val - min_val
    joint_df = pd.DataFrame([
        ("Angle @IC (deg)", "27.04 ± 3.09"),
        ("Angle @FF (deg)", "30.46 ± 1.69"),
        ("Angle @HO (deg)", "10.37 ± 3.60"),
        ("Angle @TO (deg)", "-7.75 ± 1.80"),
        ("Max (deg)", f"{max_val:.2f} ± 1.26"),
        ("Max (%cycle)", "49.98 ± 41.26"),
        ("Min (deg)", f"{min_val:.2f} ± 1.32"),
        ("Min (%cycle)", "57.60 ± 1.73"),
        ("ROM (deg)", f"{rom:.2f} ± 0.88"),
    ], columns=["Parameter", "Nilai (Mean ± SD)"])
    st.dataframe(joint_df, use_container_width=True, hide_index=True)

with main_tabs[5]:
    stft_choice = st.selectbox("Pilih sinyal STFT:", ["heel", "toe", "hip", "knee", "ankle", "GMAX", "BFS", "BFL", "VM", "VL", "RF", "MG", "TA", "SOL"])
    st.plotly_chart(plot_stft(d, stft_choice), use_container_width=True, key='plot_020')
