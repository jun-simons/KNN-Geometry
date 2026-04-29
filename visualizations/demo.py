import numpy as np
import matplotlib.pyplot as plt


class KDNode:
    def __init__(self, point, axis, left=None, right=None):
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right


def build_kdtree(points, depth=0):
    if len(points) == 0:
        return None

    k = points.shape[1]
    axis = depth % k

    points = points[points[:, axis].argsort()]
    median = len(points) // 2

    return KDNode(
        point=points[median],
        axis=axis,
        left=build_kdtree(points[:median], depth + 1),
        right=build_kdtree(points[median + 1 :], depth + 1),
    )


def plot_kdtree(ax, node, xmin, xmax, ymin, ymax, depth=0):
    if node is None:
        return

    x, y = node.point
    axis = node.axis

    if axis == 0:
        ax.plot([x, x], [ymin, ymax], linestyle="--", linewidth=1)
        plot_kdtree(ax, node.left, xmin, x, ymin, ymax, depth + 1)
        plot_kdtree(ax, node.right, x, xmax, ymin, ymax, depth + 1)
    else:
        ax.plot([xmin, xmax], [y, y], linestyle="--", linewidth=1)
        plot_kdtree(ax, node.left, xmin, xmax, ymin, y, depth + 1)
        plot_kdtree(ax, node.right, xmin, xmax, y, ymax, depth + 1)


def knn_bruteforce(points, query, k):
    dists = np.sum((points - query) ** 2, axis=1)
    idx = np.argsort(dists)[:k]
    return idx


def main():
    # Example 2D dataset
    points = np.array([
        [2.0, 3.0],
        [5.0, 4.0],
        [9.0, 6.0],
        [4.0, 7.0],
        [8.0, 1.0],
        [7.0, 2.0],
        [1.5, 8.0],
        [3.0, 5.5],
        [6.5, 8.0],
        [8.5, 4.5],
    ])

    query = np.array([6.0, 3.5])
    k = 3

    tree = build_kdtree(points)

    pad = 0.8
    xmin, ymin = points.min(axis=0) - pad
    xmax, ymax = points.max(axis=0) + pad

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw partitions
    plot_kdtree(ax, tree, xmin, xmax, ymin, ymax)

    # Draw all points
    ax.scatter(points[:, 0], points[:, 1], s=60, label="Data points")

    # Label points
    for i, (x, y) in enumerate(points):
        ax.text(x + 0.08, y + 0.08, str(i), fontsize=9)

    # Query point
    ax.scatter([query[0]], [query[1]], s=140, marker="*", label="Query point")

    # Highlight k nearest neighbors
    nn_idx = knn_bruteforce(points, query, k)
    neighbors = points[nn_idx]
    ax.scatter(
        neighbors[:, 0],
        neighbors[:, 1],
        s=180,
        facecolors="none",
        linewidths=2,
        label=f"{k} nearest neighbors",
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title("2D kD-Tree Partition Plot")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()