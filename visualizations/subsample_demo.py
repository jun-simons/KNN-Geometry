import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BUCKET_SIZE = 12
SEED = 7

# generate 2D points from overlapping Gaussians -
rng = random.Random(SEED)

centers = [(2, 3), (6, 7), (9, 3), (4, 8), (7, 2)]
colors  = ["#e05252", "#5287e0", "#52c06e", "#d4a017", "#9b52e0"]

points = []   # (x, y, class_idx)
for _ in range(400):
    ci = rng.randint(0, len(centers) - 1)
    cx, cy = centers[ci]
    x = rng.gauss(cx, 1.6)
    y = rng.gauss(cy, 1.6)
    points.append((x, y, ci))

# -- kD-tree partitioner --
# returning list of leaf cells where each cell is a list of point indices + bbox
def build_cells(pts_idx, all_pts, xmin, xmax, ymin, ymax, bucket_size):
    if len(pts_idx) <= bucket_size:
        return [{"pts": pts_idx, "bbox": (xmin, xmax, ymin, ymax)}]

    xs = [all_pts[i][0] for i in pts_idx]
    ys = [all_pts[i][1] for i in pts_idx]
    spread_x = max(xs) - min(xs)
    spread_y = max(ys) - min(ys)

    if spread_x >= spread_y:
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
cells = build_cells(
    list(range(len(points))), points,
    min(xs_all) - 0.3, max(xs_all) + 0.3,
    min(ys_all) - 0.3, max(ys_all) + 0.3,
    BUCKET_SIZE
)

# pick the centroid index thats closest point to mean of cell
keep_idx = set()
for cell in cells:
    pts = cell["pts"]
    if not pts:
        continue
    mx = sum(points[i][0] for i in pts) / len(pts)
    my = sum(points[i][1] for i in pts) / len(pts)
    closest = min(pts, key=lambda i: (points[i][0]-mx)**2 + (points[i][1]-my)**2)
    keep_idx.add(closest)

# create plot
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect("equal")
ax.axis("off")

# Draw cell boundaries
for cell in cells:
    x0, x1, y0, y1 = cell["bbox"]
    rect = patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=0.6, edgecolor="#aaaaaa", facecolor="none"
    )
    ax.add_patch(rect)

# Draw points — removed (light) first, kept (solid) on top
for i, (x, y, ci) in enumerate(points):
    base = colors[ci]
    if i in keep_idx:
        ax.scatter(x, y, color=base, s=38, zorder=3, linewidths=0)
    else:
        ax.scatter(x, y, color=base, s=18, alpha=0.33, zorder=2, linewidths=0)

ax.set_xlim(min(xs_all) - 0.5, max(xs_all) + 0.5)
ax.set_ylim(min(ys_all) - 0.5, max(ys_all) + 0.5)

fig.tight_layout(pad=0)
fig.savefig("subsample_demo.png", dpi=180, bbox_inches="tight")
print("saved subsample_demo.png")
