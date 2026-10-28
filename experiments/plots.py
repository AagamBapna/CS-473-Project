"""Figures for Experiments 2 and 3 from results/*.csv."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results"); FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
C_A3, C_KNN, C_CL = "#2563eb", "#9333ea", "#16a34a"

def band(ax, g, col, label, color, x="n_train"):
    s = g.groupby("size").agg(n=(x, "mean"), m=(col, "mean"), sd=(col, "std")).reset_index()
    ax.plot(s.n, s.m, "-o", color=color, label=label, ms=4); ax.fill_between(s.n, s.m - s.sd.fillna(0), s.m + s.sd.fillna(0), color=color, alpha=0.15)
    return s

# ---------------- Experiment 2
a3 = pd.read_csv(os.path.join(RES, "exp2_approach3.csv")); r3 = pd.read_csv(os.path.join(RES, "exp3_results.csv"))
knn = r3[r3.experiment == "exp2_random"]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, col, title in zip(axes, ["mean_per_q_qwk", "large_q", "small_q"], ["All 91 questions (leaderboard metric)", "Questions 1-10 (ASAP)", "Questions 11-91 (CS, ~22 graded each)"]):
    band(ax, a3, col, "Approach 3 (no LLM)", C_A3); band(ax, knn[knn["size"] >= 0.1], col, "Approach 1: TF-IDF KNN", C_KNN)
    ax.set_title(title, fontsize=10); ax.set_xlabel("number of graded training answers"); ax.grid(alpha=.3)
axes[0].set_ylabel("mean per-question QWK on P"); axes[0].legend(fontsize=8)
fig.suptitle("Experiment 2: QWK vs. training-set size (T/P = 90/10; mean ± std over random orderings)", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp2_learning_curve.png"), dpi=160)

# ---------------- Experiment 3 heatmap
cl = r3[r3.experiment == "exp3_cluster"]
H = cl[cl.seed == 0].groupby(["N", "size"]).mean_per_q_qwk.mean().unstack("size")
rnd = knn.groupby("size").mean_per_q_qwk.mean()
H.loc["random"] = rnd[H.columns]
fig, ax = plt.subplots(figsize=(9, 3.8))
im = ax.imshow(H.values, aspect="auto", cmap="viridis")
ax.set_xticks(range(len(H.columns))); ax.set_xticklabels([f"{int(s*100)}%" for s in H.columns]); ax.set_yticks(range(len(H.index))); ax.set_yticklabels([f"N={i}" if i != "random" else "random" for i in H.index])
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        ax.text(j, i, f"{H.values[i, j]:.2f}", ha="center", va="center", color="w" if H.values[i, j] < H.values.max() * 0.75 else "k", fontsize=7)
ax.set_xlabel("training-set size (% of T)"); ax.set_ylabel("number of clusters N"); ax.set_title("Experiment 3: mean per-question QWK of TF-IDF KNN, examples chosen across K-Means++ clusters", fontsize=10)
fig.colorbar(im, ax=ax, label="QWK"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp3_heatmap.png"), dpi=160)
# choose N: best mean QWK over the small sizes (<= 30%)
small = cl[(cl["size"] <= 0.3) & (cl.seed == 0)].groupby("N").mean_per_q_qwk.mean(); bestN = int(small.idxmax())
print("mean QWK at sizes <=30% by N:", small.round(4).to_dict(), "-> chosen N =", bestN)

# ---------------- Experiment 3 curves
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, col, title in zip(axes, ["mean_per_q_qwk", "large_q", "small_q"], ["All 91 questions", "Questions 1-10", "Questions 11-91"]):
    band(ax, knn, col, "random selection", C_KNN); band(ax, cl[cl.N == bestN], col, f"K-Means++ selection (N={bestN})", C_CL)
    ax.set_title(title, fontsize=10); ax.set_xlabel("number of graded training answers"); ax.grid(alpha=.3)
axes[0].set_ylabel("mean per-question QWK on P"); axes[0].legend(fontsize=8)
fig.suptitle("Experiment 3: TF-IDF KNN with clustered vs. random training-set selection", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp3_curves.png"), dpi=160)
tab = pd.DataFrame({"size": H.columns, "n_train": knn.groupby("size").n_train.mean()[H.columns].round(0).values,
                    "random": rnd[H.columns].values, f"clustered_N{bestN}": H.loc[bestN].values})
tab["gain"] = tab[f"clustered_N{bestN}"] - tab["random"]; tab.to_csv(os.path.join(RES, "exp3_curve_table.csv"), index=False)
print(tab.round(4).to_string(index=False))
print(a3.groupby("size")[["n_train", "mean_per_q_qwk", "large_q", "small_q"]].mean().round(4).to_string())
