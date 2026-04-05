import csv

label_map = {
    "Setosa": 0,
    "Versicolor": 1,
    "Virginica": 2
}

input_file = "iris.csv"
output_file = "iris_numeric.csv"

with open(input_file, "r", newline="") as fin, open(output_file, "w", newline="") as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)

    header = next(reader)  # skip header row

    for row in reader:
        if not row:
            continue

        features = row[:-1]
        label_str = row[-1].strip().strip('"')

        if label_str not in label_map:
            raise ValueError(f"Unknown label: {label_str}")

        label = label_map[label_str]
        writer.writerow(features + [label])

print(f"Wrote cleaned dataset to {output_file}")
