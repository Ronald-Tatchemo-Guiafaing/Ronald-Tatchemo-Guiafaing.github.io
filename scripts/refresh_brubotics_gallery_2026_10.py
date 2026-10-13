# -*- coding: utf-8 -*-
"""Obtain 4 unique Rossana/Hyser progress figures, keep progressive best 4 on portfolio."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

HOME = Path.home()
RESULTS = HOME / "Desktop" / "BruBotics_Results"
MATS = RESULTS / "mats"
LOGS = RESULTS / "logs"
IMG_SRC = RESULTS / "images"
OBTAINED = RESULTS / "images" / "portfolio_candidates_2026-10"
PORT = HOME / "Documents" / "2026-30" / "portfolio-site"
PROJ = PORT / "assets" / "img" / "projects"
INDEX = PORT / "index.html"
PROGRESS = Path(r"D:\Ronald_Data\BruBotics_Verstraten_VUB\Progress_by_Month\2026-10_October\figures_matlab")

FINAL_NAMES = [
    "brubotics-hyser-emg-force-trial.png",
    "brubotics-hyser-force-estimate.png",
    "brubotics-hyser-ndof-fingers.png",
    "brubotics-hyser-tableV-subject01.png",
]


def style_dark():
    plt.rcParams.update(
        {
            "figure.facecolor": "#0d0d0d",
            "axes.facecolor": "#0d0d0d",
            "axes.edgecolor": "#cccccc",
            "axes.labelcolor": "#dddddd",
            "xtick.color": "#bbbbbb",
            "ytick.color": "#bbbbbb",
            "text.color": "#eeeeee",
            "grid.color": "#333333",
            "font.size": 10,
        }
    )


def obtain_01_table_v_s01() -> Path:
    d = sio.loadmat(str(MATS / "Err_tableV_subject01.mat"), squeeze_me=True)
    Err = np.asarray(d["Err"], dtype=float)
    overall = float(d["overall"])
    fingers = ["Thumb", "Index", "Middle", "Ring", "Little"]
    trials = ["Trial 1", "Trial 2", "Trial 3"]

    style_dark()
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    im = ax.imshow(Err, cmap="viridis", aspect="auto", vmin=0.02, vmax=0.11)
    ax.set_xticks(range(3))
    ax.set_xticklabels(trials)
    ax.set_yticks(range(5))
    ax.set_yticklabels(fingers)
    for i in range(5):
        for j in range(3):
            ax.text(j, i, f"{Err[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("RMSE (norm. MVC)")
    ax.set_title(f"Hyser Table V — subject 01 (leave-one-trial-out)  |  overall = {overall:.4f}")
    fig.tight_layout()
    out = OBTAINED / "01_tableV_subject01.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def obtain_02_ndof_five_fingers() -> Path:
    """Unique viz from demo_ndof log: RMSE for all five fingers."""
    log = (LOGS / "demo_ndof_last.txt").read_text(encoding="utf-8", errors="ignore")
    vals = {}
    for name in ("thumb", "index", "middle", "ring", "little"):
        m = re.search(rf"Err_{name}=([0-9.]+)", log, re.I)
        if not m:
            raise SystemExit(f"missing Err_{name} in demo_ndof_last.txt")
        vals[name] = float(m.group(1))
    fingers = ["Thumb", "Index", "Middle", "Ring", "Little"]
    y = [vals[k] for k in ("thumb", "index", "middle", "ring", "little")]
    mean_e = float(np.mean(y))

    style_dark()
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    x = np.arange(5)
    bars = ax.bar(x, y, color="#F58518", edgecolor="#dddddd", width=0.65)
    ax.axhline(mean_e, color="#4C78A8", ls="--", lw=1.2, label=f"Mean = {mean_e:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(fingers)
    ax.set_ylabel("RMSE (norm. MVC)")
    ax.set_title("Hyser N-DoF — finger-wise force RMSE (subject 01 demo)")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(loc="upper right", framealpha=0.25)
    for b, v in zip(bars, y):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.001, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = OBTAINED / "02_ndof_five_fingers.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def obtain_03_sil() -> Path:
    d = sio.loadmat(str(MATS / "demo_decomp_subject01.mat"), squeeze_me=True)
    SIL = np.asarray(d["SIL"], dtype=float)
    mean_sil = float(d["MeanSIL"])
    num_mu = int(d["NumMU"])

    style_dark()
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    idx = np.arange(1, len(SIL) + 1)
    ax.plot(idx, SIL, "o-", color="#54A24B", lw=1.8, ms=7)
    ax.axhline(0.6, color="#888888", ls="--", lw=1, label="SIL = 0.6")
    ax.set_xlabel("Source index")
    ax.set_ylabel("SIL")
    ax.set_ylim(0.5, 1.02)
    ax.set_xticks(idx)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower left", framealpha=0.3)
    ax.set_title(f"EMG decomposition — subject 01, finger 2  |  MU={num_mu}, mean SIL={mean_sil:.3f}")
    fig.tight_layout()
    out = OBTAINED / "03_emg_decomposition_SIL.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def obtain_04_table_v_all20() -> Path:
    d = sio.loadmat(str(MATS / "Err_TABLE_V_all20_CHECKPOINT.mat"), squeeze_me=True)
    Err = np.asarray(d["Err"], dtype=float)  # (5, 3, 20, 2)
    per_finger = np.nanmean(Err, axis=(1, 2, 3))
    fingers = ["Thumb", "Index", "Middle", "Ring", "Little"]

    style_dark()
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    x = np.arange(5)
    bars = ax.bar(x, per_finger, color="#4C78A8", edgecolor="#dddddd", width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(fingers)
    ax.set_ylabel("Mean RMSE (norm. MVC)")
    ax.set_title("Hyser Table V — mean RMSE across subjects (checkpoint)")
    ax.grid(True, axis="y", alpha=0.35)
    for b, v in zip(bars, per_finger):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0015, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = OBTAINED / "04_tableV_all20_mean.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def copy_progress(paths: list[Path]):
    if PROGRESS.parent.exists():
        PROGRESS.mkdir(parents=True, exist_ok=True)
        for p in paths:
            shutil.copy2(p, PROGRESS / p.name)


def install_progressive_best(obtained: dict[str, Path]):
    """
    Pool = existing portfolio figures + 4 newly obtained.
    Best 4 (progressive, not 'today only'):
      1) EMG+force trial (existing) — pipeline context
      2) 1-DoF estimate (existing) — core curve result
      3) N-DoF five-finger RMSE (new) — multi-DoF progress
      4) Table V subject 01 (new) — paper Section IIIC metric
    SIL + all20 kept as candidates only.
    """
    # remove previous ndof-index if present (replaced by five-finger summary)
    old = PROJ / "brubotics-hyser-ndof-index.png"
    if old.exists():
        old.unlink()
        print("removed", old.name)

    shutil.copy2(obtained["ndof"], PROJ / FINAL_NAMES[2])
    shutil.copy2(obtained["table_v"], PROJ / FINAL_NAMES[3])
    print("kept", FINAL_NAMES[0])
    print("kept", FINAL_NAMES[1])
    print("installed", FINAL_NAMES[2])
    print("installed", FINAL_NAMES[3])


def patch_html():
    html = INDEX.read_text(encoding="utf-8")
    html = html.replace(
        "Research internship, BruBotics / R&amp;MM, VUB, Aug. 2026 to Present, Hybrid",
        "Research collaboration, BruBotics / R&amp;MM, VUB, Aug. 2026 to Present, Hybrid",
    )
    new_gallery = """          <div class="gallery">
            <figure>
              <img src="assets/img/projects/brubotics-hyser-emg-force-trial.png?v=20261011b" alt="Forearm HD-sEMG with simultaneous finger force" loading="lazy">
              <figcaption>256-channel forearm HD-sEMG with simultaneous finger force</figcaption>
            </figure>
            <figure>
              <img src="assets/img/projects/brubotics-hyser-force-estimate.png?v=20261011b" alt="Finger force estimate compared with measured force" loading="lazy">
              <figcaption>1-DoF finger force: estimate vs measured signal (Hyser)</figcaption>
            </figure>
            <figure>
              <img src="assets/img/projects/brubotics-hyser-ndof-fingers.png?v=20261011b" alt="N-DoF force RMSE for five fingers" loading="lazy">
              <figcaption>N-DoF finger-wise force RMSE (Hyser)</figcaption>
            </figure>
            <figure>
              <img src="assets/img/projects/brubotics-hyser-tableV-subject01.png?v=20261011b" alt="Table V RMSE by finger for subject 01" loading="lazy">
              <figcaption>Table V RMSE by finger, subject 01 (Hyser)</figcaption>
            </figure>
          </div>"""
    pat = re.compile(
        r'(<p class="meta">Research collaboration, BruBotics.*?</p>.*?)(<div class="gallery">.*?</div>)',
        re.S,
    )
    m = pat.search(html)
    if not m:
        raise SystemExit("BruBotics gallery not found")
    html = html[: m.start(2)] + new_gallery + html[m.end(2) :]
    INDEX.write_text(html, encoding="utf-8")
    print("patched", INDEX)


def main():
    OBTAINED.mkdir(parents=True, exist_ok=True)
    p1 = obtain_01_table_v_s01()
    p2 = obtain_02_ndof_five_fingers()
    p3 = obtain_03_sil()
    p4 = obtain_04_table_v_all20()
    obtained_list = [p1, p2, p3, p4]
    print("OBTAINED 4 unique visualizations:")
    for p in obtained_list:
        print(" ", p)
    copy_progress(obtained_list)
    install_progressive_best({"ndof": p2, "table_v": p1})
    patch_html()
    print("GALLERY best 4 = EMG trial + 1DoF + NDoF-5fingers + TableV s01")


if __name__ == "__main__":
    main()
