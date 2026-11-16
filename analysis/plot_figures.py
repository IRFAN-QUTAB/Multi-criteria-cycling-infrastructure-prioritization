import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "outputs")
FIG = os.path.join(os.path.dirname(HERE), "figures")
CITIES = ["modena", "ferrara", "pesaro"]
CT = {"modena": "Modena", "ferrara": "Ferrara", "pesaro": "Pesaro"}
CRITERIA = ["C1_pagerank", "C2_modal_impact", "C3_poi_loss",
            "C4_fragmentation", "C5_compensation_failure"]
CSHORT = {"C1_pagerank": "C1 Network importance", "C2_modal_impact":
          "C2 Detour", "C3_poi_loss": "C3 Service loss",
          "C4_fragmentation": "C4 Fragmentation",
          "C5_compensation_failure": "C5 Compensation failure"}
METHODS = ["NR_TOPSIS", "VIKOR", "PROMETHEE_II", "TOPSIS", "WSM", "EDAS",
           "MABAC", "MARCOS"]
LABEL = {"NR_TOPSIS": "NR-TOPSIS", "TOPSIS": "TOPSIS", "VIKOR": "VIKOR",
         "PROMETHEE_II": "PROMETHEE II", "WSM": "WSM", "EDAS": "EDAS",
         "MABAC": "MABAC", "MARCOS": "MARCOS", "Consensus": "MCDM consensus",
         "PageRank_only": "PageRank", "CarOnly_MCDM": "Car-only MCDM",
         "IPS_original": "Heuristic IPS", "Random": "Random"}
COL = {"NR_TOPSIS": "#0072B2", "TOPSIS": "#56B4E9", "VIKOR": "#D55E00",
       "PROMETHEE_II": "#009E73", "WSM": "#999999", "EDAS": "#CC79A7",
       "MABAC": "#E69F00", "MARCOS": "#F0E442", "Consensus": "#000000",
       "PageRank_only": "#E69F00", "CarOnly_MCDM": "#CC79A7",
       "IPS_original": "#999999", "Random": "#bbbbbb"}
MRK = {"NR_TOPSIS": "o", "VIKOR": "s", "PROMETHEE_II": "^",
       "PageRank_only": "D", "CarOnly_MCDM": "v", "IPS_original": "x",
       "Consensus": "*"}
LAYER_COL = {"car": "#D55E00", "bike_all": "#56B4E9", "bike_safe": "#009E73"}
LAYER_LAB = {"car": "Car", "bike_all": "All-cycling",
             "bike_safe": "Low-stress cycling"}
W_COL = {"critic": "#0072B2", "entropy": "#E69F00",
         "critic_entropy": "#009E73", "equal": "#999999"}
W_LAB = {"critic": "CRITIC", "entropy": "Entropy",
         "critic_entropy": "CRITIC-entropy", "equal": "Equal"}

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize": 11, "axes.titlesize": 12, "legend.fontsize": 9,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "lines.linewidth": 1.8,
    "lines.markersize": 5, "axes.spines.top": False,
    "axes.spines.right": False})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, "{}.{}".format(name, ext)))
    plt.close(fig)
    print("saved", name)


def read(name, city=None):
    return pd.read_csv(os.path.join(OUT, city, name) if city
                       else os.path.join(OUT, name))


def fig_asymmetry():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, c in zip(axes, CITIES):
        d = read("asym_summary.csv", c)
        d = d[d.rule == "iso"]
        for m in ("car", "bike_all", "bike_safe"):
            x = d[d.layer == m].sort_values("k")
            ax.plot(x.k, x.targeted_adj, marker="o", color=LAYER_COL[m],
                    label="{} - targeted".format(LAYER_LAB[m]))
            mu, sd = x.alternatives_adj_mean, x.alternatives_adj_sd
            ax.plot(x.k, mu, ls="--", color=LAYER_COL[m],
                    label="{} - random".format(LAYER_LAB[m]))
            ax.fill_between(x.k, mu - sd, mu + sd, color=LAYER_COL[m],
                            alpha=0.12, lw=0)
        ax.set_title(CT[c])
        ax.set_xlabel("Roads closed simultaneously ($k$)")
    axes[0].set_ylabel("Adjusted detour ratio $\\bar{r}$")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    save(fig, "fig1_asymmetry")


def fig_compensation():
    classes = ["none", "partial", "full", "bike_only_vulnerable",
               "no_impact"]
    lab = {"none": "Not compensated", "partial": "Partially compensated",
           "full": "Fully compensated",
           "bike_only_vulnerable": "Bicycle-only vulnerable",
           "no_impact": "No impact"}
    col = {"none": "#D55E00", "partial": "#E69F00", "full": "#009E73",
           "bike_only_vulnerable": "#CC79A7", "no_impact": "#dddddd"}
    s = read("summary_classes.csv")
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    y = np.arange(len(CITIES))
    left = np.zeros(len(CITIES))
    for k in classes:
        v = []
        for c in CITIES:
            d = s[s.city == c].set_index("compensation_class")["n"]
            v.append(d.get(k, 0) / d.sum())
        v = np.array(v)
        ax.barh(y, v, left=left, color=col[k], label=lab[k],
                edgecolor="white", lw=0.5)
        for i, (l0, vi) in enumerate(zip(left, v)):
            if vi > 0.05:
                ax.text(l0 + vi / 2, i, "{:.0f}%".format(100 * vi),
                        ha="center", va="center", fontsize=8)
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels([CT[c] for c in CITIES])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of closed roads")
    ax.legend(frameon=False, loc="lower center", ncol=3,
              bbox_to_anchor=(0.5, -0.55), fontsize=8)
    save(fig, "fig4_compensation")


def fig_coverage():
    rows = []
    for c in CITIES:
        ns = json.load(open(os.path.join(OUT, c, "network_stats.json")))
        st = json.load(open(os.path.join(OUT, c, "structure.json")))
        a = read("asym_summary.csv", c)
        a = a[(a.rule == "iso") & (a.layer == "bike_safe") & (a.k == 100)]
        rows.append((c, ns["protected_class1_km"], st["coverage_lowstress"],
                     st["edge_overlap_lowstress_car_mean"],
                     float(a.targeted_adj.iloc[0])))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for c, km, cov, ov, adj in rows:
        axes[0].scatter(km, cov, s=60, color="#0072B2", zorder=3)
        axes[0].annotate(CT[c], (km, cov), textcoords="offset points",
                         xytext=(6, 4), fontsize=9)
        axes[1].scatter(ov, adj, s=60, color="#D55E00", zorder=3)
        axes[1].annotate(CT[c], (ov, adj), textcoords="offset points",
                         xytext=(6, 4), fontsize=9)
    axes[0].set_xlabel("Protected cycling infrastructure (km)")
    axes[0].set_ylabel("Low-stress coverage")
    axes[0].set_title("(a) Coverage")
    axes[1].set_xlabel("Edge overlap of low-stress and car routes")
    axes[1].set_ylabel("Low-stress $\\bar{r}$, top-100 closed")
    axes[1].set_title("(b) Robustness")
    for ax in axes:
        ax.grid(color="#eeeeee", zorder=0)
    fig.tight_layout()
    save(fig, "fig5_coverage_independence")


def fig_weights():
    w = read("summary_weights.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    x = np.arange(len(CRITERIA))
    for ax, c in zip(axes, CITIES):
        d = w[w.city == c].set_index("criterion").loc[CRITERIA]
        for i, sc in enumerate(["critic", "entropy", "critic_entropy",
                                "equal"]):
            ax.bar(x + (i - 1.5) * 0.2, d[sc], 0.2, color=W_COL[sc],
                   label=W_LAB[sc])
        ax.set_xticks(x)
        ax.set_xticklabels([k.split("_")[0] for k in CRITERIA])
        ax.set_title(CT[c])
    axes[0].set_ylabel("Criterion weight")
    axes[0].legend(frameon=False, loc="upper right", fontsize=8)
    fig.text(0.5, -0.04, "  |  ".join(CSHORT[k] for k in CRITERIA),
             ha="center", fontsize=9)
    save(fig, "figM1_weights")


def fig_similarity():
    s = read("summary_rank_similarity_methods.csv")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5),
                             gridspec_kw={"wspace": 0.45})
    im = None
    for ax, c in zip(axes, CITIES):
        d = s[(s.city == c) & s.ranking_A.isin(METHODS)
              & s.ranking_B.isin(METHODS)]
        M = np.eye(len(METHODS))
        for _, r in d.iterrows():
            i, j = METHODS.index(r.ranking_A), METHODS.index(r.ranking_B)
            lo, hi = sorted((i, j))
            M[hi, lo], M[lo, hi] = r.spearman_rho, r.WS_A_ref
        im = ax.imshow(M, vmin=0.6, vmax=1.0, cmap="Blues")
        for i in range(len(METHODS)):
            for j in range(len(METHODS)):
                if i != j:
                    ax.text(j, i, "{:.2f}".format(M[i, j]), ha="center",
                            va="center", fontsize=7,
                            color="white" if M[i, j] > 0.9 else "black")
        lbl = [LABEL[n] for n in METHODS]
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(lbl, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels(lbl, fontsize=8)
        ax.set_title(CT[c])
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02)
    cb.set_label("Similarity")
    fig.text(0.45, -0.04, "Lower triangle: Spearman's rho. Upper triangle: "
             "WS coefficient (row ranking as reference).", ha="center",
             fontsize=9)
    save(fig, "figM2_rank_similarity")


def fig_weight_scenarios():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    for ax, c in zip(axes, CITIES):
        t = read("sensitivity_weight_scenarios_top10_ranks.csv", c)
        scen = [k for k in t.columns if k.startswith("S")]
        x = np.arange(len(scen) + 1)
        for i, (_, r) in enumerate(t.iterrows()):
            y = np.r_[r["base"], r[scen].values.astype(float)]
            ax.plot(x, y, marker="o", markersize=2.5, lw=1.1,
                    color=plt.cm.tab10(i % 10), label=r["road_name"])
        ax.set_yscale("log")
        ax.invert_yaxis()
        ax.set_yticks([1, 2, 5, 10, 20, 50, 100])
        ax.set_yticklabels(["1", "2", "5", "10", "20", "50", "100"])
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
        ax.set_xticklabels(["Base", "S5", "S10", "S15", "S20", "S25",
                            "S30"], fontsize=8)
        for b in (10.5, 20.5):
            ax.axvline(b, color="#cccccc", lw=0.8)
        ax.set_ylabel("Consensus rank (log scale)")
        ax.set_title(CT[c])
        ax.legend(frameon=False, fontsize=7, loc="upper center",
                  bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.text(0.5, -0.03, "S1-S10, S11-S20, S21-S30: weight of the first, "
             "second, and third most important criterion reduced by "
             "5%-95%.", ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, "figM4_weight_scenarios")


def fig_contribution():
    c = read("summary_criteria_contribution.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(len(CRITERIA))
    cols = ["#0072B2", "#D55E00", "#009E73"]
    for i, city in enumerate(CITIES):
        d = c[c.city == city].set_index("criterion").loc[CRITERIA]
        axes[0].bar(x + (i - 1) * 0.27, d["CFA_NR_TOPSIS"], 0.27,
                    color=cols[i], label=CT[city])
        axes[1].bar(x + (i - 1) * 0.27, d["spearman_with_Consensus"], 0.27,
                    color=cols[i])
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([k.split("_")[0] for k in CRITERIA])
    axes[0].set_ylabel("Contribution factor (NR-TOPSIS)")
    axes[1].set_ylabel("Spearman: criterion vs consensus")
    axes[0].legend(frameon=False)
    save(fig, "figM6_criteria_contribution")


def fig_rank_reversal():
    fx = read("summary_rr_fixed.csv")
    re = read("summary_rr_reest.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    x = np.arange(len(CITIES))
    bw = 0.1
    for ax, d, title in ((axes[0], fx, "Fixed weights and bounds"),
                         (axes[1], re, "Weights and bounds re-estimated")):
        for i, m in enumerate(METHODS):
            v = [float(d[(d.city == c) & (d.method == m)]
                       .steps_with_top10_order_change.iloc[0]
                       / d[(d.city == c) & (d.method == m)].steps.iloc[0])
                 for c in CITIES]
            ax.bar(x + (i - 3.5) * bw, v, bw, color=COL[m], label=LABEL[m])
        ax.set_xticks(x)
        ax.set_xticklabels([CT[c] for c in CITIES])
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Share of deletion steps\nthat change the top-10 order")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="lower center", ncol=8,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, "figM7_rank_reversal")


def fig_smaa(n_show=15, n_rank=10):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8),
                             gridspec_kw={"wspace": 0.6})
    im = None
    for ax, c in zip(axes, CITIES):
        t = read("smaa_NR_TOPSIS.csv", c).head(n_show)
        B = t[["b{}".format(r) for r in range(1, n_rank + 1)]].values
        im = ax.imshow(B, vmin=0, vmax=max(0.5, B.max()), cmap="Blues",
                       aspect="auto")
        ax.set_yticks(range(len(t)))
        ax.set_yticklabels(["{} ({:.2f})".format(n, a) for n, a in
                            zip(t.road_name, t.holistic_acceptability)],
                           fontsize=7.5)
        ax.set_xticks(range(n_rank))
        ax.set_xticklabels(range(1, n_rank + 1))
        ax.set_xlabel("Rank")
        ax.set_title(CT[c])
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02)
    cb.set_label("Rank acceptability $b_i^r$")
    save(fig, "figM8_smaa_NR_TOPSIS")


def fig_consensus_ranks(n_show=10):
    cols = {"NR_TOPSIS": "NR_TOPSIS_rank", "VIKOR": "VIKOR_rank",
            "PROMETHEE_II": "PROMETHEE_II_rank",
            "CarOnly_MCDM": "CarOnly_MCDM_rank",
            "PageRank_only": "PageRank_only_rank",
            "IPS_original": "IPS_original_rank"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, c in zip(axes, CITIES):
        d = read("scores_rankings.csv", c).head(n_show).iloc[::-1]
        y = np.arange(len(d))
        for m, k in cols.items():
            ax.scatter(d[k], y, marker=MRK[m], color=COL[m], label=LABEL[m],
                       zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(d.road_name, fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("Rank (log scale)")
        ax.set_title(CT[c])
        ax.grid(axis="x", color="#eeeeee", zorder=0)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    save(fig, "figM9_consensus_top10_ranks")


def fig_validation():
    order = ["Consensus", "CarOnly_MCDM", "PageRank_only", "IPS_original"]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.8))
    for col, c in enumerate(CITIES):
        v = read("validation_topk.csv", c)
        v = v[v.sets == "B"]
        for row, m in enumerate(("bike_safe", "car")):
            ax = axes[row, col]
            d = v[v.layer == m]
            mu = d[d.ranking == "Random_mean"].sort_values("k")
            sd = d[d.ranking == "Random_sd"].sort_values("k")
            ax.fill_between(mu.k, mu.adj.values - sd.adj.values,
                            mu.adj.values + sd.adj.values, color="#dddddd",
                            label="Random (mean $\\pm$ 1 s.d.)")
            ax.plot(mu.k, mu.adj, color="#999999", ls="--", lw=1)
            for r in order:
                x = d[d.ranking == r].sort_values("k")
                ax.plot(x.k, x.adj, marker=MRK[r], color=COL[r],
                        label=LABEL[r])
            ax.set_xlabel("Roads closed simultaneously ($k$)")
            ax.set_ylabel("$\\bar{r}$, " + ("low-stress cycling"
                                            if m == "bike_safe" else "car"))
            if row == 0:
                ax.set_title(CT[c])
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    save(fig, "figM10_validation_topk_closures")


def fig_redundancy():
    order = ["Consensus", "CarOnly_MCDM", "PageRank_only", "IPS_original"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    x = np.arange(len(CITIES))
    bw = 0.16
    for i, r in enumerate(order + ["Random_mean"]):
        v = []
        for c in CITIES:
            b = read("bypass_summary.csv", c)
            v.append(float(b[(b.ranking == r) & (b.k == 20)].avoided_sum
                           .iloc[0]))
        axes[0].bar(x + (i - 2) * bw, v, bw,
                    color=COL.get(r, "#bbbbbb"),
                    label=LABEL.get(r, "Random (mean)"))
    for j, c in enumerate(CITIES):
        b = read("bypass_summary.csv", c)
        p95 = float(b[(b.ranking == "Random_p95") & (b.k == 20)]
                    .avoided_sum.iloc[0])
        axes[0].plot([j - 0.45, j + 0.45], [p95, p95], color="#555555",
                     ls=":", lw=1)
    axes[0].set_ylabel("Avoided low-stress harm\n(sum over top-20 roads)")
    axes[0].set_title("(a) Parallel bypass (redundancy)")
    for i, r in enumerate(order + ["Random"]):
        v = []
        for c in CITIES:
            g = read("investment_gain.csv", c)
            v.append(100 * float(g[(g.ranking == r) & (g.k == 20)]
                                 .reach_gain_pop_share.mean()))
        axes[1].bar(x + (i - 2) * bw, v, bw, color=COL.get(r, "#bbbbbb"),
                    label=LABEL.get(r, r))
    axes[1].set_ylabel("Trips gaining low-stress access\n(%, population-weighted)")
    axes[1].set_title("(b) Upgrade of the road (coverage)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([CT[c] for c in CITIES])
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    save(fig, "figM11_redundancy")


def main():
    global OUT, FIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--fig", default=FIG)
    a = ap.parse_args()
    OUT, FIG = a.out, a.fig
    os.makedirs(FIG, exist_ok=True)
    for f in (fig_asymmetry, fig_compensation, fig_coverage, fig_weights,
              fig_similarity, fig_weight_scenarios, fig_contribution,
              fig_rank_reversal, fig_smaa, fig_consensus_ranks,
              fig_validation, fig_redundancy):
        f()


if __name__ == "__main__":
    main()
