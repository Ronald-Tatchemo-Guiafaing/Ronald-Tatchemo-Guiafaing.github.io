# Portfolio BruBotics gallery updater (max 5 images).
# Run a few hours before each Rossana meeting, or after strong monthly figures.
# Picks the 2 newest/best PNGs from the current month figures* folders,
# copies into portfolio-site + github.io, keeps at most 5 brubotics-hyser-* files.

from __future__ import annotations
import re
import shutil
from pathlib import Path

HOME = Path.home()
MONTH_ROOT = HOME / "Desktop/BruBotics_Verstraten_VUB/Progress_by_Month"
PORTFOLIOS = [
    HOME / "Documents/2026-30/portfolio-site/assets/img/projects",
    HOME / "Documents/2026-30/Ronald-Tatchemo-Guiafaing.github.io/assets/img/projects",
]
HTMLS = [
    HOME / "Documents/2026-30/portfolio-site/index.html",
    HOME / "Documents/2026-30/Ronald-Tatchemo-Guiafaing.github.io/index.html",
]
MAX_IMAGES = 5
TAKE_PER_UPDATE = 2


def latest_month_dir() -> Path | None:
    if not MONTH_ROOT.exists():
        return None
    months = sorted([p for p in MONTH_ROOT.iterdir() if p.is_dir() and re.match(r"\d{4}-\d{2}_", p.name)])
    return months[-1] if months else None


def collect_candidates(month: Path) -> list[Path]:
    cands: list[Path] = []
    for sub in ("figures_matlab", "figures"):
        d = month / sub
        if d.is_dir():
            cands.extend(d.glob("*.png"))
    # prefer demo/estimate names, then newest mtime
    def score(p: Path):
        name = p.name.lower()
        bonus = 0
        if "estimate" in name or "demo" in name:
            bonus += 100
        if "trial1" in name or "sync" in name:
            bonus += 40
        return (bonus, p.stat().st_mtime)

    cands = sorted(cands, key=score, reverse=True)
    return cands[:TAKE_PER_UPDATE]


def main():
    month = latest_month_dir()
    if not month:
        print("No month folder")
        return
    picks = collect_candidates(month)
    if not picks:
        print("No figures in", month)
        return
    stamp = month.name[:7].replace("-", "")  # 202609
    new_names = []
    for i, src in enumerate(picks, 1):
        name = f"brubotics-hyser-{stamp}-{i}.png"
        new_names.append(name)
        for dest_dir in PORTFOLIOS:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest_dir / name)
            print("copied", src.name, "->", dest_dir / name)

    # prune oldest brubotics-hyser beyond MAX
    for dest_dir in PORTFOLIOS:
        existing = sorted(dest_dir.glob("brubotics-hyser-*.png"), key=lambda p: p.stat().st_mtime)
        while len(existing) > MAX_IMAGES:
            old = existing.pop(0)
            old.unlink()
            print("removed overflow", old)

    print("DONE picks:", new_names)
    print("Remember: update HTML gallery tags + git push for live site if needed.")


if __name__ == "__main__":
    main()
