import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

MAX_DEPTH = 5
SEED = 6

# --- Generate 2D points from overlapping Gaussians ---
rng = random.Random(SEED)

centers = [(2, 3), (6, 7), (9, 3), (4, 8), (7, 2)]
colors  = ["#e05252", "#5287e0", "#52c06e", "#d4a017", "#9b52e0"]

points = []
for _ in range(400):
    ci = rng.randint(0, len(centers) - 1)
    cx, cy = centers[ci]
    x = rng.gauss(cx, 1.4)
    y = rng.gauss(cy, 1.4)
    points.append((x, y, ci))

# --- Fixed-depth kD-tree partitioner ---
# Always splits to max_depth regardless of how many points are in the cell.
# Returns list of cells: {"pts": [...indices...], "bbox": (xmin,xmax,ymin,ymax)}
def build_cells_midpoint(pts_idx, all_pts, xmin, xmax, ymin, ymax, depth, max_depth):
    if depth >= max_depth or len(pts_idx) == 0:
        return [{"pts": pts_idx, "bbox": (xmin, xmax, ymin, ymax)}]

    # Split at the spatial midpoint of the bounding box, not the data median.
    # Equal-volume cells: dense regions contain more points per cell -> higher compression.
    if (xmax - xmin) >= (ymax - ymin):
        mid = (xmin + xmax) / 2
        left  = [i for i in pts_idx if all_pts[i][0] <= mid]
        right = [i for i in pts_idx if all_pts[i][0] >  mid]
        return (build_cells_midpoint(left,  all_pts, xmin, mid,  ymin, ymax, depth+1, max_depth) +
                build_cells_midpoint(right, all_pts, mid,  xmax, ymin, ymax, depth+1, max_depth))
    else:
        mid = (ymin + ymax) / 2
        lower = [i for i in pts_idx if all_pts[i][1] <= mid]
        upper = [i for i in pts_idx if all_pts[i][1] >  mid]
        return (build_cells_midpoint(lower, all_pts, xmin, xmax, ymin, mid,  depth+1, max_depth) +
                build_cells_midpoint(upper, all_pts, xmin, xmax, mid,  ymax, depth+1, max_depth))

xs_all = [p[0] for p in points]
ys_all = [p[1] for p in points]
cells = build_cells_midpoint(
    list(range(len(points))), points,
    min(xs_all) - 0.3, max(xs_all) + 0.3,
    min(ys_all) - 0.3, max(ys_all) + 0.3,
    depth=0, max_depth=MAX_DEPTH
)

# One representative per non-empty cell (closest point to centroid)
keep_idx = set()
for cell in cells:
    pts = cell["pts"]
    if not pts:
        continue
    mx = sum(points[i][0] for i in pts) / len(pts)
    my = sum(points[i][1] for i in pts) / len(pts)
    closest = min(pts, key=lambda i: (points[i][0]-mx)**2 + (points[i][1]-my)**2)
    keep_idx.add(closest)

# --- Plot ---
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect("equal")
ax.axis("off")

# Cell boundaries
for cell in cells:
    x0, x1, y0, y1 = cell["bbox"]
    rect = patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=0.6, edgecolor="#aaaaaa", facecolor="none"
    )
    ax.add_patch(rect)

# Points — faded first, solid kept on top
for i, (x, y, ci) in enumerate(points):
    base = colors[ci]
    if i in keep_idx:
        ax.scatter(x, y, color=base, s=38, zorder=3, linewidths=0)
    else:
        ax.scatter(x, y, color=base, s=18, alpha=0.33, zorder=2, linewidths=0)

ax.set_xlim(min(xs_all) - 0.5, max(xs_all) + 0.5)
ax.set_ylim(min(ys_all) - 0.5, max(ys_all) + 0.5)

fig.tight_layout(pad=0)
fig.savefig("subsample_demo_midpoint.png", dpi=180, bbox_inches="tight")
print("saved subsample_demo_midpoint.png")
