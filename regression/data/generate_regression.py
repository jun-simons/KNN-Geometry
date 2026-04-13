import argparse
import csv
import random
import math


def generate(n, d, seed):
    rng = random.Random(seed)

    data = []
    for _ in range(n):
        x = [rng.uniform(-5, 5) for _ in range(d)]

        # nonlinear function + noise
        y = sum(xi**2 for xi in x) + math.sin(x[0]) + rng.gauss(0, 3.0)

        data.append((x, y))

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--features", type=int, default=4)
    parser.add_argument("--out", type=str, default="regression.csv")
    args = parser.parse_args()

    data = generate(args.samples, args.features, 42)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        for x, y in data:
            writer.writerow([f"{v:.6f}" for v in x] + [f"{y:.6f}"])

    print("Generated", args.out)


if __name__ == "__main__":
    main()
