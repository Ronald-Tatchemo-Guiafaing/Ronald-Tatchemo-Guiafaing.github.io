# -*- coding: utf-8 -*-
"""Publication-style BruBotics / Hyser figures for the portfolio (white, journal look)."""
from __future__ import annotations

import re
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

HOME = Path.home()
HYSER = Path(r"D:\Ronald_Data\BruBotics_Verstraten_VUB\Hyser_HD-sEMG_Dataset")
MATS = HOME / "Desktop" / "BruBotics_Results" / "mats"
LOGS = HOME / "Desktop" / "BruBotics_Results" / "logs"
OUT = HOME / "Desktop" / "BruBotics_Results" / "images" / "portfolio_serious_2026-10"
PROJ = HOME / "Documents" / "2026-30" / "portfolio-site" / "assets" / "img" / "projects"
INDEX = HOME / "Documents" / "2026-30" / "portfolio-site" / "index.html"
PROGRESS = Path(
    r"D:\Ronald_Data\BruBotics_Verstraten_VUB\Progress_by_Month\2026-10_October\figures_matlab"
)

FS_EMG = 2048
FS_FORCE = 100
WIN = int(0.02 * FS_EMG)  # 40 samples
STEP = WIN


def journal_rc():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "axes.linewidth": 1.0,
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "grid.color": "#b0b0b0",
            "grid.linewidth": 0.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_hea_gains(hea: Path, n_sig: int) -> tuple[np.ndarray, np.ndarray]:
    lines = hea.read_text(encoding="utf-8", errors="ignore").splitlines()
    gains = np.ones(n_sig)
    baselines = np.zeros(n_sig)
    # WFDB lines after header: filename fmt gain(baseline)/units
    sig_i = 0
    for line in lines[1:]:
        if sig_i >= n_sig:
            break
        parts = line.split()
        if len(parts) < 3:
            continue
        g = parts[2]
        m = re.match(r"([0-9.eE+-]+)\((-?[0-9.eE+-]+)\)", g)
        if m:
            gains[sig_i] = float(m.group(1))
            baselines[sig_i] = float(m.group(2))
        else:
            try:
                gains[sig_i] = float(g.split("(")[0])
            except ValueError:
                pass
        sig_i += 1
    return gains, baselines


def load_dat(stem: Path, n_ch: int) -> np.ndarray:
    """Load Hyser .dat (int16, channel-major) -> (samples, channels)."""
    dat = stem.with_suffix(".dat")
    hea = stem.with_suffix(".hea")
    raw = np.fromfile(dat, dtype="<i2")
    if raw.size % n_ch != 0:
        # try samples-major
        n_samp = raw.size // n_ch
        arr = raw[: n_samp * n_ch].reshape(n_samp, n_ch)
    else:
        # MATLAB fread [M, inf] then transpose -> file is channel-first
        n_samp = raw.size // n_ch
        arr = raw.reshape(n_ch, n_samp).T
    gains, baselines = read_hea_gains(hea, n_ch)
    return (arr.astype(np.float64) - baselines) / gains


def sliding_feat(emg: np.ndarray) -> np.ndarray:
    """RMS, WL, ZC, SSC per window -> (n_win, 4*n_ch)."""
    n, c = emg.shape
    starts = range(0, n - WIN + 1, STEP)
    feats = []
    thr = 0.0004
    for s in starts:
        w = emg[s : s + WIN]
        rms = np.sqrt(np.mean(w * w, axis=0))
        wl = np.sum(np.abs(np.diff(w, axis=0)), axis=0)
        zc = np.sum(((w[:-1] * w[1:]) < 0) & (np.abs(w[:-1] - w[1:]) > thr), axis=0)
        ssc = np.sum(
            ((w[1:-1] - w[:-2]) * (w[1:-1] - w[2:]) > 0) & (np.abs(w[1:-1] - w[:-2]) > thr),
            axis=0,
        )
        feats.append(np.concatenate([rms, wl, zc, ssc]))
    return np.asarray(feats)


def force_windows(force: np.ndarray, n_win: int) -> np.ndarray:
    """Average force labels onto EMG windows (force at 100 Hz, EMG windows 20 ms)."""
    # force length at 100 Hz for ~same duration as EMG
    t_emg = (np.arange(n_win) * STEP + WIN / 2) / FS_EMG
    t_f = np.arange(len(force)) / FS_FORCE
    # use force channel for the active finger later
    return t_emg, t_f


def estimate_1dof(finger: int = 2, train_sample: int = 1, test_sample: int = 2):
    base = HYSER / "1dof_dataset" / "subject01_session1"
    # samples 1..3, fingers 1..5
    emgs, forces = [], []
    for s in (1, 2, 3):
        emg = load_dat(base / f"1dof_preprocess_finger{finger}_sample{s}", 256)
        frc = load_dat(base / f"1dof_force_finger{finger}_sample{s}", 5)
        emgs.append(emg)
        forces.append(frc[:, finger - 1])

    # MVC normalize roughly by p99 of abs force across trials
    scale = np.percentile(np.abs(np.concatenate(forces)), 99) + 1e-9
    forces = [f / scale for f in forces]

    X_list, y_list = [], []
    for emg, frc in zip(emgs, forces):
        X = sliding_feat(emg)
        t_emg = (np.arange(len(X)) * STEP + WIN / 2) / FS_EMG
        t_f = np.arange(len(frc)) / FS_FORCE
        y = np.interp(t_emg, t_f, frc)
        # mild force smoothing
        b, a = butter(2, 10 / (0.5 * (1 / (STEP / FS_EMG))), btype="low")
        try:
            y = filtfilt(b, a, y)
        except Exception:
            pass
        X_list.append(X)
        y_list.append(y)

    # train on sample train_sample, test on test_sample (1-based)
    tr = [i for i in range(3) if i != (test_sample - 1)]
    te = test_sample - 1
    Xtr = np.vstack([X_list[i] for i in tr])
    ytr = np.concatenate([y_list[i] for i in tr])
    Xte = X_list[te]
    yte = y_list[te]

    # standardize
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr_n = (Xtr - mu) / sd
    Xte_n = (Xte - mu) / sd
    pca = PCA(n_components=min(200, Xtr_n.shape[1], Xtr_n.shape[0] - 1))
    Ztr = pca.fit_transform(Xtr_n)
    Zte = pca.transform(Xte_n)

    # lag features (20 lags)
    Q = 20

    def lag_stack(Z):
        rows = []
        for t in range(Q, len(Z)):
            rows.append(Z[t - Q : t + 1].ravel())
        return np.asarray(rows)

    Ltr = lag_stack(Ztr)
    Lte = lag_stack(Zte)
    ytr_l = ytr[Q:]
    yte_l = yte[Q:]
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(Ltr, ytr_l)
    yhat = model.predict(Lte)
    rmse = float(np.sqrt(np.mean((yhat - yte_l) ** 2)))
    return yte_l, yhat, rmse


def fig_force_1dof():
    y, yhat, rmse = estimate_1dof(2, 1, 2)
    journal_rc()
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=220)
    x = np.arange(len(y))
    ax.plot(x, y, color="black", lw=1.1, label="Measured")
    ax.plot(x, yhat, color="#9C2A1E", lw=0.9, alpha=0.9, label="Estimated")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Normalized force (%MVC scale)")
    ax.set_title(f"One-DoF finger-force estimation  |  subject 01, index  |  RMSE = {rmse:.3f}")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, len(y) - 1)
    fig.tight_layout()
    out = OUT / "brubotics-hyser-force-estimate.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig_emg_force():
    base = HYSER / "1dof_dataset" / "subject01_session1"
    emg = load_dat(base / "1dof_preprocess_finger2_sample1", 256)
    frc = load_dat(base / "1dof_force_finger2_sample1", 5)
    # short segment for clarity (~8 s)
    n = min(emg.shape[0], int(8 * FS_EMG))
    emg = emg[:n]
    t_e = np.arange(n) / FS_EMG
    t_f = np.arange(len(frc)) / FS_FORCE
    force = frc[:, 1]
    # pick 6 spaced channels
    chs = [10, 40, 80, 120, 160, 200]
    journal_rc()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.2), dpi=220, sharex=False, gridspec_kw={"height_ratios": [2.2, 1]})
    ax = axes[0]
    offset = 0.0
    for i, ch in enumerate(chs):
        sig = emg[:, ch]
        sig = sig / (np.std(sig) + 1e-9) * 0.35
        ax.plot(t_e, sig + offset, color="black", lw=0.45)
        ax.text(-0.35, offset, f"Ch {ch+1}", fontsize=7, va="center", ha="right")
        offset -= 1.1
    ax.set_ylabel("HD-sEMG (normalized, offset)")
    ax.set_title("Forearm HD-sEMG and finger force  |  Hyser, subject 01")
    ax.set_xlim(0, t_e[-1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([])

    ax2 = axes[1]
    ax2.plot(t_f, force / (np.percentile(np.abs(force), 99) + 1e-9), color="black", lw=1.0)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Force (norm.)")
    ax2.set_xlim(0, t_e[-1])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig.tight_layout()
    out = OUT / "brubotics-hyser-emg-force-trial.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig_ndof_bars():
    log = (LOGS / "demo_ndof_last.txt").read_text(encoding="utf-8", errors="ignore")
    vals = []
    labels = ["Thumb", "Index", "Middle", "Ring", "Little"]
    for name in ("thumb", "index", "middle", "ring", "little"):
        m = re.search(rf"Err_{name}=([0-9.]+)", log, re.I)
        vals.append(float(m.group(1)))
    mean_e = float(np.mean(vals))
    journal_rc()
    fig, ax = plt.subplots(figsize=(7.0, 3.4), dpi=220)
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals[::-1], color="#4a4a4a", edgecolor="black", height=0.62, linewidth=0.6)
    ax.axvline(mean_e, color="#1f4e79", ls="--", lw=1.0, label=f"Mean RMSE = {mean_e:.3f}")
    ax.set_yticks(y)
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel("RMSE (normalized MVC)")
    ax.set_title("N-DoF force estimation error by finger  |  subject 01")
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for yi, v in zip(y, vals[::-1]):
        ax.text(v + 0.001, yi, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, max(vals) * 1.25)
    fig.tight_layout()
    out = OUT / "brubotics-hyser-ndof-fingers.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig_table_v():
    d = sio.loadmat(str(MATS / "Err_tableV_subject01.mat"), squeeze_me=True)
    Err = np.asarray(d["Err"], dtype=float)
    overall = float(d["overall"])
    fingers = ["Thumb", "Index", "Middle", "Ring", "Little"]
    trials = ["Trial 1", "Trial 2", "Trial 3"]
    journal_rc()
    fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=220)
    im = ax.imshow(Err, cmap="Greys", aspect="auto", vmin=0.02, vmax=0.11)
    ax.set_xticks(range(3))
    ax.set_xticklabels(trials)
    ax.set_yticks(range(5))
    ax.set_yticklabels(fingers)
    for i in range(5):
        for j in range(3):
            val = Err[i, j]
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                color="white" if val > 0.065 else "black",
                fontsize=8,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("RMSE (normalized MVC)")
    ax.set_title(f"Table V reproduction — subject 01 (leave-one-trial-out)  |  overall = {overall:.3f}")
    fig.tight_layout()
    out = OUT / "brubotics-hyser-tableV-subject01.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def install(paths: list[Path]):
    OUT.mkdir(parents=True, exist_ok=True)
    PROJ.mkdir(parents=True, exist_ok=True)
    if PROGRESS.parent.exists():
        PROGRESS.mkdir(parents=True, exist_ok=True)
    for p in paths:
        dest = PROJ / p.name
        dest.write_bytes(p.read_bytes())
        print("portfolio:", dest)
        if PROGRESS.parent.exists():
            (PROGRESS / p.name).write_bytes(p.read_bytes())


def patch_html():
    html = INDEX.read_text(encoding="utf-8")
    html = re.sub(r"brubotics-hyser-[^\"?]+\.png\?v=[^\"]+", lambda m: m.group(0).split("?v=")[0] + "?v=20261011c", html)
    # ensure the four filenames
    for name in (
        "brubotics-hyser-emg-force-trial.png",
        "brubotics-hyser-force-estimate.png",
        "brubotics-hyser-ndof-fingers.png",
        "brubotics-hyser-tableV-subject01.png",
    ):
        if name not in html:
            raise SystemExit(f"missing {name} in index.html")
    INDEX.write_text(html, encoding="utf-8")
    print("cache-busted index.html")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Building publication-style figures from Hyser + Table V mats...")
    paths = [
        fig_emg_force(),
        fig_force_1dof(),
        fig_ndof_bars(),
        fig_table_v(),
    ]
    install(paths)
    patch_html()
    print("DONE")


if __name__ == "__main__":
    main()
