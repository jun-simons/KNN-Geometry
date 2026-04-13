import pandas as pd
import matplotlib.pyplot as plt

summary = pd.read_csv("summary_results.csv")

plt.figure()
plt.plot(summary["k"], summary["mse_weighted"], marker="o")
plt.xlabel("k")
plt.ylabel("Weighted MSE")
plt.title("Weighted KNN Regression Performance")
plt.savefig("mse_vs_k.png", bbox_inches="tight")
plt.close()

best_k = int(summary.loc[summary["mse_weighted"].idxmin(), "k"])
details = pd.read_csv(f"details_k{best_k}.csv")

plt.figure()
plt.scatter(details["confidence"], details["abs_error"], alpha=0.25)
plt.xlabel("Confidence")
plt.ylabel("Absolute Error")
plt.title(f"Confidence vs Error (k={best_k})")
plt.savefig("confidence_vs_error.png", bbox_inches="tight")
plt.close()

details["bin"] = pd.cut(details["confidence"], bins=10)
binned = details.groupby("bin", observed=False)["abs_error"].mean()

plt.figure()
binned.plot(kind="bar")
plt.ylabel("Average Absolute Error")
plt.xlabel("Confidence Bin")
plt.title(f"Binned Confidence vs Error (k={best_k})")
plt.tight_layout()
plt.savefig("binned_confidence_error.png", bbox_inches="tight")
plt.close()

print(f"Best k = {best_k}")
print(
    "Wrote mse_vs_k.png, confidence_vs_error.png, binned_confidence_error.png"
)
