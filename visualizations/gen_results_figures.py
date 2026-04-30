import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ── Data ──────────────────────────────────────────────────────────────────────

dims       = [2, 4, 8, 16, 32]
brute_ms   = [0.0647, 0.0695, 0.0729, 0.0825, 0.1237]
kd_ms      = [0.0036, 0.0080, 0.0453, 0.1021, 0.1891]
brute_acc  = [0.9690, 0.9955, 1.0000, 1.0000, 1.0000]
kd_acc     = [0.9690, 0.9955, 1.0000, 1.0000, 1.0000]

epsilons   = [0.0,  0.1,  0.5,  1.0,  2.0,  5.0,  10.0]
eps_ms     = [0.0822, 0.0671, 0.0363, 0.0220, 0.0136, 0.0076, 0.0056]
eps_acc    = [0.9906, 0.9906, 0.9909, 0.9912, 0.9910, 0.9903, 0.9875]

# Compression data — bucket (pts, acc) sorted by pts ascending
bucket_pts = [83,  147,  287,  548,  1057, 2093, 3365]
bucket_acc = [0.9289, 0.9322, 0.9350, 0.9311, 0.9294, 0.9333, 0.9339]

midpt_pts  = [15,   27,   51,   96,   174,  313,  552,  964]
midpt_acc  = [0.9350, 0.9211, 0.9267, 0.9250, 0.9267, 0.9333, 0.9333, 0.9311]

baseline_pts = 7200
baseline_acc = 0.9272

# ── Figure 1: Dimension sweep ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(dims, brute_ms, marker="o", color="#e07b54", label="Brute force")
ax.plot(dims, kd_ms,    marker="s", color="#4c7dba", label="kD-tree")
ax.axvline(x=10, color="#aaaaaa", linewidth=0.8, linestyle="--")
ax.text(10.4, max(kd_ms)*0.92, "crossover\n≈ dim 10", fontsize=9, color="#888888")
ax.set_xlabel("Dimensions")
ax.set_ylabel("Avg query time (ms)")
ax.set_title("Query time vs dimensionality  (n=10k, k=5)")
ax.legend()
fig.tight_layout()
fig.savefig("fig_dim_sweep.png", bbox_inches="tight")
print("saved fig_dim_sweep.png")
plt.close()

# ── Figure 2: Epsilon sweep (side-by-side) ────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
fig.suptitle("Approximate kD-tree  (n=50k, 8D, k=5)", fontsize=12)

ax1.plot(epsilons, eps_ms, marker="o", color="#4c7dba")
ax1.set_xlabel(r"$\varepsilon$")
ax1.set_ylabel("Avg query time (ms)")
ax1.set_title("Query time")

ax2.plot(epsilons, eps_acc, marker="o", color="#e07b54")
ax2.set_xlabel(r"$\varepsilon$")
ax2.set_ylabel("Accuracy")
ax2.set_title("Classification accuracy")
ax2.set_ylim(0.983, 0.993)

fig.tight_layout()
fig.savefig("fig_epsilon_sweep.png", bbox_inches="tight")
print("saved fig_epsilon_sweep.png")
plt.close()

# ── Figure 3: Compression accuracy ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4))

ax.axhline(baseline_acc, color="#aaaaaa", linewidth=1.0,
           linestyle="--", label=f"No compression (acc={baseline_acc:.4f})")
ax.plot(bucket_pts, bucket_acc, marker="o", color="#52a06e",
        label="Bucket (uniform)")
ax.plot(midpt_pts,  midpt_acc,  marker="s", color="#9b52e0",
        label="Midpoint depth (density-aware)")

ax.set_xlabel("Training points kept")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs compression  (boundary dataset, k=5)")
ax.legend()
ax.set_xlim(left=0)

fig.tight_layout()
fig.savefig("fig_compression_accuracy.png", bbox_inches="tight")
print("saved fig_compression_accuracy.png")
plt.close()
