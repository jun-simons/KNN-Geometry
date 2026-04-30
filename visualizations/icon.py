import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BUCKET_SIZE = 6
SEED = 21

rng = random.Random(SEED)

centers = [(2.5, 2.0), (7.5, 2.5), (5.0, 7.5)]
colors  = ["#3a7abf", "#e05e45", "#52a06e"]
n_per   = 30

points = []
for ci, (cx, cy) in enumerate(centers):
    for _ in range(n_per):
        x = rng.gauss(cx, 1.1)
        y = rng.gauss(cy, 1.1)
        points.append((x, y, ci))

rng.shuffle(points)

def build_cells(pts_idx, all_pts, xmin, xmax, ymin, ymax, bucket_size):
    if len(pts_idx) <= bucket_size:
        return [{"pts": pts_idx, "bbox": (xmin, xmax, ymin, ymax)}]

    xs = [all_pts[i][0] for i in pts_idx]
    ys = [all_pts[i][1] for i in pts_idx]

    if (max(xs) - min(xs)) >= (max(ys) - min(ys)):
        median = sorted(xs)[len(xs) // 2]
        left  = [i for i in pts_idx if all_pts[i][0] <= median]
        right = [i for i in pts_idx if all_pts[i][0] >  median]
        return (build_cells(left,  all_pts, xmin,   median, ymin, ymax, bucket_size) +
                build_cells(right, all_pts, median, xmax,   ymin, ymax, bucket_size))
    else:
        median = sorted(ys)[len(ys) // 2]
        lower = [i for i in pts_idx if all_pts[i][1] <= median]
        upper = [i for i in pts_idx if all_pts[i][1] >  median]
        return (build_cells(lower, all_pts, xmin, xmax, ymin,   median, bucket_size) +
                build_cells(upper, all_pts, xmin, xmax, median, ymax,   bucket_size))

xs_all = [p[0] for p in points]
ys_all = [p[1] for p in points]
pad = 0.4
cells = build_cells(
    list(range(len(points))), points,
    min(xs_all) - pad, max(xs_all) + pad,
    min(ys_all) - pad, max(ys_all) + pad,
    BUCKET_SIZE
)

keep_idx = set()
for cell in cells:
    pts = cell["pts"]
    if not pts:
        continue
    mx = sum(points[i][0] for i in pts) / len(pts)
    my = sum(points[i][1] for i in pts) / len(pts)
    closest = min(pts, key=lambda i: (points[i][0]-mx)**2 + (points[i][1]-my)**2)
    keep_idx.add(closest)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("#f7f7f7")
ax.set_facecolor("#f7f7f7")

for cell in cells:
    x0, x1, y0, y1 = cell["bbox"]
    rect = patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=1.0, edgecolor="#bbbbbb", facecolor="none", zorder=1
    )
    ax.add_patch(rect)

for i, (x, y, ci) in enumerate(points):
    col = colors[ci]
    if i in keep_idx:
        ax.scatter(x, y, color=col, s=90, zorder=3, linewidths=0)
    else:
        ax.scatter(x, y, color=col, s=35, alpha=0.25, zorder=2, linewidths=0)

ax.set_xlim(min(xs_all) - pad, max(xs_all) + pad)
ax.set_ylim(min(ys_all) - pad, max(ys_all) + pad)

fig.tight_layout(pad=0.1)
fig.savefig("icon.png", dpi=200, bbox_inches="tight", facecolor="#f7f7f7")
print("saved icon.png")
