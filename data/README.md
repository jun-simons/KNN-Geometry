# data/

## Dataset format

(for now) CSV, no header, each row is `feature1,...,featureN,label` where label is an integer

```csv
5.1,3.5,1.4,0.2,0
4.9,3.0,1.4,0.2,1
...
```

## generate.py

Generates synthetic Gaussian blobs with one blob per class, randomly placed in [-10, 10]^d

```python
python generate.py [--samples N] [--features N] [--classes N] [--spread F] [--seed N] [--out file.csv]
```

Defaults: 1000 samples, 4 features, 3 classes, spread=1.0

Lower `--spread` = tighter blobs = easier classification problem.
