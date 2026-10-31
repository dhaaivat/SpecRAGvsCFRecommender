"""Render the README figures (light and dark versions) from the CSVs in results/.

    python analysis/make_figures.py

The swap hit rates in fig_swaps come from analysis/paired_tests.py, which needs the MovieLens test split.
"""
import re
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  s1="#2a78d6", s2="#eb6834", band="#f0efec"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 s1="#3987e5", s2="#d95926", band="#2c2c2a"),
}
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10})

def ci(cell):
    m, lo, hi = map(float, re.findall(r"-?\d+\.\d+", cell)[:3])
    return m, lo, hi

def style(ax, t, grid_axis="x"):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["ink2"], length=0)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)

def figure(t, w=8, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    return fig, ax

def title(fig, t, head, sub):
    fig.text(0.02, 0.965, head, color=t["ink"], fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 0.905, sub, color=t["ink2"], fontsize=9.5, va="top")

def box(t):
    return dict(boxstyle="round,pad=0.15", facecolor=t["surface"], edgecolor="none")

def top_legend(fig, ax, t, y=0.8, **kw):
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, y), bbox_transform=fig.transFigure, ncol=2, frameon=False,
              fontsize=9, labelcolor=t["ink"], handletextpad=0.4, columnspacing=1.6, **kw)

def save(fig, name, mode):
    fig.savefig(FIG / f"{name}_{mode}.png", facecolor=fig.get_facecolor())
    plt.close(fig)

ab = pd.read_csv(RES / "ablation.csv").set_index("system")
CF = ci(ab.loc["CF alone (candidate order)", "NDCG@10"])
SYSTEMS = [("CF + Gemini (single call)", "Gemini Flash\n(single call)"),
           ("CF + drafter (single)", "Qwen-7B drafter\n(single call)"),
           ("Spec RAG (argmax, paper)", "Speculative RAG\n(paper's rule)"),
           ("Spec RAG + RRF with CF", "Speculative RAG\n+ fused with CF")]

# 1. headline: every LLM system, shuffled vs CF order, against CF alone
def fig_headline(mode):
    t = THEMES[mode]
    fig, ax = figure(t, 8, 4.4)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.76, bottom=0.13)
    style(ax, t)
    ax.axvspan(CF[1], CF[2], color=t["band"], zorder=0)
    ax.axvline(CF[0], color=t["ink2"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.text(CF[0] + 0.001, len(SYSTEMS) - 0.35, f"CF alone (EASE) {CF[0]:.3f}", color=t["ink2"], fontsize=9, ha="left", va="bottom")
    for y, (key, label) in enumerate(SYSTEMS[::-1]):
        a, b = ci(ab.loc[key, "NDCG@10"]), ci(ab.loc[f"[CF order] {key}", "NDCG@10"])
        for (m, lo, hi), c, dy in [(a, t["s1"], 0.12), (b, t["s2"], -0.12)]:
            ax.plot([lo, hi], [y + dy] * 2, color=c, linewidth=1.2, alpha=0.6, zorder=2)
            ax.scatter([m], [y + dy], s=70, color=c, edgecolor=t["surface"], linewidth=2, zorder=3)
        ax.text(a[0], y + 0.3, f"{a[0]:.3f}", color=t["ink2"], fontsize=8.5, ha="center", bbox=box(t), zorder=4)
        ax.text(b[0], y - 0.45, f"{b[0]:.3f}", color=t["ink"], fontsize=8.5, ha="center", fontweight="bold", bbox=box(t), zorder=4)
    ax.set_yticks(range(len(SYSTEMS)), [l for _, l in SYSTEMS[::-1]], color=t["ink"])
    ax.set_ylim(-0.7, len(SYSTEMS) - 0.1)
    ax.set_xlim(0.035, 0.13)
    ax.set_xlabel("NDCG@10 on 500 held-out users (higher is better; whiskers = 95% CI)", color=t["ink2"])
    title(fig, t, "Showing the CF order rescues the LLMs, but none beats CF",
          "Each LLM reranks the same 50 CF candidates for 500 users. Grey band = CF's 95% CI.")
    h1 = ax.scatter([], [], s=60, color=t["s1"], label="Candidates shuffled (run 1)")
    h2 = ax.scatter([], [], s=60, color=t["s2"], label="Candidates in CF order (run 2)")
    top_legend(fig, ax, t, handles=[h1, h2])
    save(fig, "fig_headline", mode)

# 2. position bias in the shuffled arm
def fig_position_bias(mode):
    t = THEMES[mode]
    ps = pd.read_csv(RES / "pick_sources.csv", index_col=0)
    rows = [(k, l.replace("\n", " ")) for k, l in SYSTEMS[:3]]
    fig, ax = figure(t, 8, 3.4)
    fig.subplots_adjust(left=0.27, right=0.95, top=0.72, bottom=0.16)
    style(ax, t)
    vals = [ps.loc[k, "first 10 shown"] * 100 for k, _ in rows][::-1]
    ax.barh(range(len(rows)), vals, height=0.5, color=t["s1"], zorder=2)
    for y, v in enumerate(vals):
        ax.text(v + 1, y, f"{v:.0f}%", color=t["ink"], va="center", fontsize=9.5, fontweight="bold")
    ax.axvline(20, color=t["ink2"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=3)
    ax.text(20.8, len(rows) - 0.45, "20% = no position bias", color=t["ink2"], fontsize=9, va="bottom")
    ax.set_yticks(range(len(rows)), [l for _, l in rows][::-1], color=t["ink"])
    ax.set_xlim(0, 65)
    ax.set_xlabel("Share of the top-10 picks taken from the first 10 of 50 candidates listed, shuffled order (%)", color=t["ink2"])
    title(fig, t, "The 7B model mostly picks whatever is listed first",
          "With a random candidate order, the first 10 titles hold no more liked films than any other 10.")
    save(fig, "fig_position_bias", mode)

# 3. does the verifier pick the best draft?
def fig_verifier(mode):
    t = THEMES[mode]
    arms = [("Candidates shuffled", pd.read_csv(RES / "draft_level_shuffled.csv"), t["s1"]),
            ("Candidates in CF order", pd.read_csv(RES / "draft_level_CF_order.csv"), t["s2"])]
    cols = [("mean_draft", "A random\ndraft"), ("by_draft_only", "Drafter's own\nconfidence"),
            ("chosen", "Verifier\n(paper's ρ)"), ("oracle", "Best of 5\n(hindsight)")]
    fig, ax = figure(t, 8, 4.2)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.74, bottom=0.17)
    style(ax, t, "y")
    w = 0.36
    for k, (name, d, c) in enumerate(arms):
        xs = [i + (k - 0.5) * (w + 0.03) for i in range(len(cols))]
        vals = [d[col].mean() for col, _ in cols]
        ax.bar(xs, vals, width=w, color=c, label=name, zorder=2)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.002, f"{v:.3f}", ha="center", color=t["ink"], fontsize=8.5, bbox=box(t), zorder=4)
    ax.axhline(CF[0], color=t["ink2"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=3)
    ax.text(-0.45, CF[0] + 0.003, f"CF alone {CF[0]:.3f}", color=t["ink2"], fontsize=9, ha="left", bbox=box(t), zorder=4)
    ax.set_xticks(range(len(cols)), [l for _, l in cols], color=t["ink"])
    ax.set_ylim(0, 0.155)
    ax.set_ylabel("Mean NDCG@10 of the chosen draft", color=t["ink2"])
    top_legend(fig, ax, t)
    title(fig, t, "The 14B verifier picks no better than the drafter's own confidence",
          "Speculative RAG writes 5 drafts per user and keeps one. Good drafts exist; the scores can't find them.")
    save(fig, "fig_verifier", mode)

# 4. what the LLM swaps into CF's top 10 vs what it pushes out (CF-order arm; from analysis/paired_tests.py)
SWAPS = {"Qwen-7B drafter (single call)": (0.0343, 0.0242), "Speculative RAG (paper's rule)": (0.0307, 0.0215)}
BASE_11_50 = 0.0185

def fig_swaps(mode):
    t = THEMES[mode]
    fig, ax = figure(t, 8, 4.0)
    fig.subplots_adjust(left=0.29, right=0.95, top=0.7, bottom=0.15)
    style(ax, t)
    h = 0.32
    for y, (name, (out, inn)) in enumerate(list(SWAPS.items())[::-1]):
        ax.barh(y + h / 2 + 0.02, out * 100, height=h, color=t["s1"], zorder=2, label="Pushed out of CF's top 10" if y == 0 else None)
        ax.barh(y - h / 2 - 0.02, inn * 100, height=h, color=t["s2"], zorder=2, label="Brought in from CF ranks 11–50" if y == 0 else None)
        ax.text(out * 100 + 0.05, y + h / 2 + 0.02, f"{out:.1%}", va="center", color=t["ink"], fontsize=9)
        ax.text(inn * 100 + 0.05, y - h / 2 - 0.02, f"{inn:.1%}", va="center", color=t["ink"], fontsize=9)
    ax.axvline(BASE_11_50 * 100, color=t["ink2"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=3)
    ax.text(BASE_11_50 * 100 + 0.05, 1.62, "average film at CF ranks 11–50", color=t["ink2"], fontsize=8.5, ha="left", bbox=box(t), zorder=4)
    ax.set_yticks(range(len(SWAPS)), list(SWAPS)[::-1], color=t["ink"])
    ax.set_ylim(-0.6, 1.85)
    ax.set_xlim(0, 4.2)
    ax.set_xlabel("How often the swapped film was one the user actually liked next (%)", color=t["ink2"])
    top_legend(fig, ax, t, y=0.78)
    title(fig, t, "Each swap the LLM makes trades a better film for a worse one",
          "With the CF order shown, the LLM keeps ~7 of CF's top 10 and swaps ~3 (run 2).")
    save(fig, "fig_swaps", mode)

for mode in THEMES:
    fig_headline(mode)
    fig_position_bias(mode)
    fig_verifier(mode)
    fig_swaps(mode)
print("figures written to", FIG)
