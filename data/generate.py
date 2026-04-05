"""
generate.py — synthetic KNN benchmark dataset generator

Produces CSV files in the same format expected by the C++ KNN implementation:
  feature1,feature2,...,featureN,label   (no header, integer labels)

Data is generated as isotropic Gaussian blobs, one per class.

Usage:
  python generate.py [options]

Options:
  --samples    Total number of points          (default: 1000)
  --features   Number of feature dimensions    (default: 4)
  --classes    Number of classes               (default: 3)
  --spread     Std-dev of each blob            (default: 1.0)
  --seed       Random seed for reproducibility (default: 42)
  --out        Output file path                (default: synthetic.csv)

Examples:
  python generate.py --samples 5000 --features 8 --classes 5
  python generate.py --samples 100000 --out big_bench.csv
"""

import argparse
import csv
import random


def generate_blobs(n_samples, n_features, n_classes, spread, seed):
    rng = random.Random(seed)

    # Place class centers randomly in [-10, 10]^n_features
    centers = [
        [rng.uniform(-10, 10) for _ in range(n_features)]
        for _ in range(n_classes)
    ]

    points = []
    for i in range(n_samples):
        label = i % n_classes      
        center = centers[label]
        features = [rng.gauss(center[d], spread) for d in range(n_features)]
        points.append((features, label))

    rng.shuffle(points)
    return points


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic KNN benchmark dataset")
    parser.add_argument("--samples",  type=int,   default=1000,          help="Total number of points")
    parser.add_argument("--features", type=int,   default=4,             help="Number of feature dimensions")
    parser.add_argument("--classes",  type=int,   default=3,             help="Number of classes")
    parser.add_argument("--spread",   type=float, default=1.0,           help="Std-dev of each Gaussian blob")
    parser.add_argument("--seed",     type=int,   default=42,            help="Random seed")
    parser.add_argument("--out",      type=str,   default="synthetic.csv", help="Output CSV file path")
    args = parser.parse_args()

    if args.samples < args.classes:
        parser.error("--samples must be >= --classes")
    if args.features < 1:
        parser.error("--features must be >= 1")

    points = generate_blobs(args.samples, args.features, args.classes, args.spread, args.seed)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        for features, label in points:
            writer.writerow([f"{v:.6f}" for v in features] + [label])

    print(f"Wrote {args.samples} points  ({args.features}D, {args.classes} classes)  ->  {args.out}")


if __name__ == "__main__":
    main()
