import json
import os
import matplotlib.pyplot as plt

METRICS_FILE = "results/stage3_report.json"

with open(METRICS_FILE, "r", encoding="utf-8") as f:
    metrics = json.load(f)

chunk_sizes = []
overlaps = []
top_ks = []
exact_matches = []
avg_chunks = []

for filename, data in metrics.items():
    parts = filename.replace(".json", "").split("_")
    size = int(parts[3])
    overlap = int(parts[5])
    k = int(parts[7])

    chunk_sizes.append(size)
    overlaps.append(overlap)
    top_ks.append(k)
    exact_matches.append(data["exact_match"])
    avg_chunks.append(data["avg_chunks_used"])

def plot_metric(x, y, labels, title, ylabel, xlabel="Experiment"):
    plt.figure(figsize=(10,6))
    plt.bar(range(len(x)), y, color="skyblue")
    plt.xticks(range(len(x)), labels, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

labels = list(metrics.keys())

plot_metric(
    x=labels,
    y=exact_matches,
    labels=labels,
    title="RAG Experiments: Exact Match per Retrieval Setup",
    ylabel="Exact Match"
)

plot_metric(
    x=labels,
    y=avg_chunks,
    labels=labels,
    title="RAG Experiments: Average Chunks Used",
    ylabel="Avg Chunks Used"
)

plt.figure(figsize=(8,5))
plt.scatter(chunk_sizes, exact_matches, s=[k*20 for k in top_ks], c=overlaps, cmap="viridis", alpha=0.7)
for i, label in enumerate(labels):
    plt.text(chunk_sizes[i], exact_matches[i]+0.005, f"k={top_ks[i]}", fontsize=8, ha="center")
plt.colorbar(label="Overlap")
plt.xlabel("Chunk Size")
plt.ylabel("Exact Match")
plt.title("Exact Match vs Chunk Size (bubble ~ top_k, color ~ overlap)")
plt.show()