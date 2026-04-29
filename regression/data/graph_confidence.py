import pandas as pd
import matplotlib.pyplot as plt

# load your data (pick best k, e.g. k=9)
df = pd.read_csv("details_k9.csv")

# create scatter plot
plt.figure(figsize=(6, 5))
plt.scatter(df["confidence"], df["abs_error"], alpha=0.3)

# labels
plt.xlabel("Confidence")
plt.ylabel("Absolute Error")
plt.title("Confidence vs Error (Synthetic Data)")

# optional: add trend line
import numpy as np

z = np.polyfit(df["confidence"], df["abs_error"], 1)
p = np.poly1d(z)
plt.plot(df["confidence"], p(df["confidence"]), linestyle="--")

plt.tight_layout()
plt.savefig("confidence_vs_error_scatter.png")
plt.show()
