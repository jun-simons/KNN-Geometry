import matplotlib.pyplot as plt

# --- Data ---

epsilons   = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
eps_time   = [0.0915, 0.0741, 0.0392, 0.0238, 0.0138, 0.0076, 0.0056]
eps_acc    = [0.9796, 0.9796, 0.9794, 0.9788, 0.9779, 0.9758, 0.9722]

dims       = [2, 4, 8, 16, 32]
brute_time = [0.0650, 0.0695, 0.0725, 0.0818, 0.1233]
kd_time    = [0.0036, 0.0080, 0.0476, 0.0956, 0.1873]

# --- Style ---
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# ── Plot 1: epsilon sweep ──────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
fig.suptitle("Approximate kD-tree  (50k pts, 8D, k=5)", fontsize=12)

ax1.plot(epsilons, eps_time, marker="o", color="steelblue")
ax1.set_xlabel("ε")
ax1.set_ylabel("avg query time (ms)")
ax1.set_title("Query time vs ε")

ax2.plot(epsilons, eps_acc, marker="o", color="coral")
ax2.set_xlabel("ε")
ax2.set_ylabel("accuracy")
ax2.set_title("Accuracy vs ε")
ax2.set_ylim(0.96, 1.001)

fig.tight_layout()
fig.savefig("epsilon_sweep.png", dpi=150)
print("saved epsilon_sweep.png")

# ── Plot 2: dimension sweep ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.set_title("kD-tree vs brute force  (10k pts, k=5)")

ax.plot(dims, brute_time, marker="o", label="brute force", color="coral")
ax.plot(dims, kd_time,    marker="s", label="kD-tree",     color="steelblue")
ax.axhline(0, color="gray", linewidth=0.5)
ax.set_xlabel("dimensions")
ax.set_ylabel("avg query time (ms)")
ax.legend()

fig.tight_layout()
fig.savefig("dimension_sweep.png", dpi=150)
print("saved dimension_sweep.png")
