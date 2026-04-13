import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
REPORTS_DIR = os.path.join(PROJECT_ROOT, "results", "reports")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "results", "plots")

RETRIEVAL_REPORT = os.path.join(REPORTS_DIR, "retrieval_report.json")
RAG_REPORT = os.path.join(REPORTS_DIR, "rag_evaluation_report.json")
FAITH_REPORT = os.path.join(REPORTS_DIR, "faithfulness_report.json")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)

METHOD_COLORS = {
    "MiniLM": "#4C72B0",
    "BGE": "#DD8452",
    "BM25": "#55A868",
    "Hybrid": "#C44E52",
}

PROMPT_HATCH = {
    "default": "",
    "extraction": "///",
}

CHUNK_SIZES = [200, 500, 1000]

_INT = re.compile(r"\d+")


def parse_retrieval_name(filename: str):
    name = filename.replace(".json", "")

    if name.startswith("bge_"):
        method = "BGE"
        name = name[len("bge_") :]
    elif name.startswith("bm25_"):
        method = "BM25"
        name = name[len("bm25_") :]
    elif name.startswith("hybrid_"):
        method = "Hybrid"
        name = name[len("hybrid_") :]
    else:
        method = "MiniLM"

    nums = _INT.findall(name)
    chunk_size = int(nums[0])
    overlap = int(nums[1])
    k = int(nums[2])
    return method, chunk_size, overlap, k


def parse_rag_name(filename: str):
    name = filename.replace(".json", "")

    if name.startswith("extraction_rag_"):
        prompt = "extraction"
        name = name[len("extraction_rag_") :]
    elif name.startswith("rag_"):
        prompt = "default"
        name = name[len("rag_") :]
    else:
        return None

    method, chunk_size, overlap, k = parse_retrieval_name(name)
    return prompt, method, chunk_size, overlap, k


def load_retrieval():
    with open(RETRIEVAL_REPORT, encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for fname, data in raw.items():
        try:
            method, chunk_size, overlap, k = parse_retrieval_name(fname)
        except Exception:
            continue
        rows.append(
            {
                "method": method,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "k": k,
                "doc_recall": data["document_recall"],
                "answer_recall": data["answer_recall"],
            }
        )
    return rows


def load_rag():
    with open(RAG_REPORT, encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for fname, data in raw.items():
        parsed = parse_rag_name(fname)
        if parsed is None:
            continue
        prompt, method, chunk_size, overlap, k = parsed
        rows.append(
            {
                "prompt": prompt,
                "method": method,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "k": k,
                "contains": data["contains_match"],
                "f1": data["average_f1"],
                "sem_sim": data["average_semantic_similarity"],
                "refusal_rate": data["refusal_rate"],
            }
        )
    return rows


def load_faith():
    with open(FAITH_REPORT, encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for fname, data in raw.items():
        parsed = parse_rag_name(fname)
        if parsed is None:
            continue
        prompt, method, chunk_size, overlap, k = parsed
        rows.append(
            {
                "prompt": prompt,
                "method": method,
                "chunk_size": chunk_size,
                "k": k,
                "faithfulness": data["faithfulness_score"],
                "supported": data["supported"],
                "partial": data["partially_supported"],
                "unsupported": data["unsupported"],
            }
        )
    return rows


def lookup(rows, **filters):
    """Return the first row matching all keyword filters."""
    for row in rows:
        if all(row.get(k) == v for k, v in filters.items()):
            return row
    return None


def lookall(rows, **filters):
    return [r for r in rows if all(r.get(k) == v for k, v in filters.items())]


# plot 1 recall_by_method
def plot_recall_by_method(retrieval_rows):
    methods = ["MiniLM", "BGE", "BM25", "Hybrid"]
    n_sizes = len(CHUNK_SIZES)
    n_methods = len(methods)
    bar_w = 0.18
    x = np.arange(n_sizes)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, method in enumerate(methods):
        vals = []
        for cs in CHUNK_SIZES:
            row = lookup(retrieval_rows, method=method, chunk_size=cs, k=5)
            vals.append(row["answer_recall"] if row else 0.0)

        offset = (i - (n_methods - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset,
            vals,
            bar_w,
            label=method,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.6,
        )

        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"chunk={cs}" for cs in CHUNK_SIZES])
    ax.set_ylabel("Answer Recall")
    ax.set_ylim(0, 1.05)
    ax.set_title("Answer Recall at k=5 by Retrieval Method and Chunk Size")
    ax.legend(title="Method", loc="lower right")

    path = os.path.join(PLOTS_DIR, "01_answer_recall_by_method.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# plot 2 recall_vs_k
def plot_recall_vs_k(retrieval_rows):
    k_values = [1, 3, 5, 10]
    methods = ["MiniLM", "BGE", "BM25", "Hybrid"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in methods:
        vals = []
        for k in k_values:
            row = lookup(retrieval_rows, method=method, chunk_size=200, k=k)
            vals.append(row["answer_recall"] if row else None)

        valid_k = [k for k, v in zip(k_values, vals) if v is not None]
        valid_v = [v for v in vals if v is not None]

        ax.plot(
            valid_k,
            valid_v,
            marker="o",
            linewidth=2,
            markersize=7,
            color=METHOD_COLORS[method],
            label=method,
        )

        for k, v in zip(valid_k, valid_v):
            ax.annotate(
                f"{v:.2f}",
                (k, v),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )

    ax.set_xticks(k_values)
    ax.set_xlabel("Top-k retrieved chunks")
    ax.set_ylabel("Answer Recall")
    ax.set_ylim(0.3, 1.0)
    ax.set_title("Answer Recall vs k  (chunk size = 200)")
    ax.legend(title="Method")

    path = os.path.join(PLOTS_DIR, "02_recall_vs_k.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# plot 3 rag_quality
def plot_rag_quality(rag_rows):
    methods = ["MiniLM", "BGE", "BM25", "Hybrid"]
    metrics = [
        ("contains", "Contains-match"),
        ("f1", "Token F1"),
        ("sem_sim", "Semantic Similarity"),
    ]
    prompts = ["default", "extraction"]
    prompt_labels = {"default": "Default prompt", "extraction": "Extraction prompt"}
    prompt_colors = {"default": "#7DB8D7", "extraction": "#E8855A"}

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(14, 5), sharey=False)
    fig.suptitle(
        "RAG Answer Quality: Default vs Extraction Prompt  (chunk=200, k=5)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    bar_w = 0.35
    x = np.arange(len(methods))

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        for j, prompt in enumerate(prompts):
            vals = []
            for method in methods:
                row = lookup(
                    rag_rows, prompt=prompt, method=method, chunk_size=200, k=5
                )
                vals.append(row[metric_key] if row else 0.0)

            offset = (j - 0.5) * bar_w
            bars = ax.bar(
                x + offset,
                vals,
                bar_w,
                label=prompt_labels[prompt],
                color=prompt_colors[prompt],
                edgecolor="white",
                linewidth=0.6,
            )

            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=10)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)

    handles = [
        mpatches.Patch(color=prompt_colors[p], label=prompt_labels[p]) for p in prompts
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
    )

    path = os.path.join(PLOTS_DIR, "03_rag_quality_prompt_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# plot 4 refusal_rate
def plot_refusal_rate(rag_rows):
    methods = ["MiniLM", "BGE", "BM25", "Hybrid"]
    prompts = ["default", "extraction"]
    prompt_colors = {"default": "#7DB8D7", "extraction": "#E8855A"}
    prompt_labels = {"default": "Default prompt", "extraction": "Extraction prompt"}

    bar_w = 0.35
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(9, 5))

    for j, prompt in enumerate(prompts):
        vals = []
        for method in methods:
            row = lookup(rag_rows, prompt=prompt, method=method, chunk_size=200, k=5)
            vals.append(row["refusal_rate"] if row else 0.0)

        offset = (j - 0.5) * bar_w
        bars = ax.bar(
            x + offset,
            vals,
            bar_w,
            label=prompt_labels[prompt],
            color=prompt_colors[prompt],
            edgecolor="white",
            linewidth=0.6,
        )

        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.0%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Refusal Rate")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Model Refusal Rate: Default vs Extraction Prompt  (chunk=200, k=5)")
    ax.legend(frameon=False)

    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(len(methods) - 0.5, 0.515, "50%", color="grey", fontsize=9)

    path = os.path.join(PLOTS_DIR, "04_refusal_rate.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# plot 5 faithfulness
def plot_faithfulness(faith_rows):
    methods = ["MiniLM", "BGE", "BM25", "Hybrid"]
    prompts = ["default", "extraction"]
    n_total = 44

    configs = []
    for prompt in prompts:
        for method in methods:
            row = lookup(faith_rows, prompt=prompt, method=method, chunk_size=200, k=5)
            if row:
                configs.append(
                    {
                        "label": f"{method}\n({prompt[:3].capitalize()})",
                        "supported": row["supported"] / n_total,
                        "partial": row["partial"] / n_total,
                        "unsupported": row["unsupported"] / n_total,
                        "score": row["faithfulness"],
                        "prompt": prompt,
                    }
                )

    labels = [c["label"] for c in configs]
    sup = [c["supported"] for c in configs]
    part = [c["partial"] for c in configs]
    unsup = [c["unsupported"] for c in configs]
    scores = [c["score"] for c in configs]

    x = np.arange(len(configs))
    bar_w = 0.55

    fig, ax = plt.subplots(figsize=(12, 5.5))

    b1 = ax.bar(x, sup, bar_w, label="Supported", color="#4CAF50", edgecolor="white")
    b2 = ax.bar(
        x,
        part,
        bar_w,
        bottom=sup,
        label="Partially supported",
        color="#FFC107",
        edgecolor="white",
    )
    b3 = ax.bar(
        x,
        unsup,
        bar_w,
        bottom=[s + p for s, p in zip(sup, part)],
        label="Unsupported",
        color="#F44336",
        edgecolor="white",
    )

    for i, score in enumerate(scores):
        ax.text(
            i,
            1.02,
            f"F={score:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.axvline(
        len(methods) - 0.5, color="black", linewidth=1.2, linestyle="--", alpha=0.5
    )
    ax.text(
        len(methods) - 0.5, 0.5, "  default →", fontsize=8, color="grey", va="center"
    )
    ax.text(
        len(methods) - 0.5,
        0.5,
        "← extraction  ",
        fontsize=8,
        color="grey",
        va="center",
        ha="right",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Fraction of answers")
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title(
        "Faithfulness Breakdown: Default vs Extraction Prompt  (chunk=200, k=5)\n"
        "F = faithfulness score  (supported + 0.5 × partial) / total"
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    path = os.path.join(PLOTS_DIR, "05_faithfulness_stacked.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading reports...")
    retrieval_rows = load_retrieval()
    rag_rows = load_rag()
    faith_rows = load_faith()

    print(f"  retrieval entries : {len(retrieval_rows)}")
    print(f"  rag entries       : {len(rag_rows)}")
    print(f"  faithfulness entries: {len(faith_rows)}")
    print()

    print("Generating plots...")
    plot_recall_by_method(retrieval_rows)
    plot_recall_vs_k(retrieval_rows)
    plot_rag_quality(rag_rows)
    plot_refusal_rate(rag_rows)
    plot_faithfulness(faith_rows)

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
