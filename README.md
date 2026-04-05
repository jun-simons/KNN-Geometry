# KNN-Geometry

Run KNN (both brute force and kD-tree):

First compile with cmake to build directory, then:

```bash
./knn_kdtree_test <dataset> <k> <train-split proportion> <has_header>
```

Iris dataset:

```bash
 ./knn_kdtree_test ../data/iris_numeric.csv 3 0.8 0
```

Generated big dataset (very slow):

```bash
./knn_kdtree_test ../data/bench_100k.csv 3 0.8 0
```
