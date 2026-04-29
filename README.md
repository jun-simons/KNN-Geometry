# KNN-Geometry

KNN classifier comparing brute force vs. kD-tree using CGAL. Includes approximate search, spatial subsampling experiments, and visualizations
## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
```

## Run

```bash
./build/knn_kdtree_test --file <csv> [options]

  --k <int>               number of neighbours (default: 3)
  --train <float>         train fraction (default: 0.8)
  --header                csv has a header row
  --no-brute              skip brute force, kD-tree only
  --epsilon <float>       approximate search factor (default: 0 = exact)
  --subsample <int>       subsample train set by kD-tree bucket size
  --subsample-depth <int> subsample by fixed tree depth (median split)
  --subsample-midpoint <int> subsample by fixed depth with spatial midpoint splits
```

Examples:

```bash
# exact, both methods
./build/knn_kdtree_test --file data/iris_numeric.csv --k 3

# approximate kD-tree only
./build/knn_kdtree_test --file data/bench_50k_8d.csv --k 5 --no-brute --epsilon 1.0

# spatial subsampling
./build/knn_kdtree_test --file data/boundary_hard.csv --k 5 --no-brute --subsample 20
```

## Data

Synthetic datasets are generated with `data/generate.py`. See `data/README.md`.

## Visualizations

Figures and demo scripts are in `visualizations/`. Run `visualizations/render.sh` to regenerate the kD-tree graphviz diagram.
