import os
import json
import numpy as np
import matplotlib.pyplot as plt


# def visualize_all_metrics(dataset, base_path='/mnt/nas/users/mehar21541_3_59/HITL-ReID/outputs/pretrained_models',
def visualize_all_metrics(
    dataset,
    base_path="outputs/pretrained_models",
    thresh=0.2,
    seed=0,
    min_samples=5,
    num_splits=5,
    K=5,
):
    print("For Dataset:", dataset, "thresh:", thresh, "K:", K)
    images_per_id = 1
    metrics = [
        ("Top-1 Accuracy", "Top-1 Accuracy (%)"),
        ("Top-3 Accuracy", "Top-3 Accuracy (%)"),
        ("Top-5 Accuracy", "Top-5 Accuracy (%)"),
        ("Top-10 Accuracy", "Top-10 Accuracy (%)"),
        ("AP", "Average Precision"),
        ("INP", "INP"),
    ]
    methods = os.listdir(os.path.join(base_path, dataset))
    if "skyline" in methods:
        methods.remove("skyline")
    if "baseline" in methods:
        methods.remove("baseline")
    methods = np.random.permutation(methods)

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.flatten()

    def load_constant_metric(method, metric_key):
        log_path = os.path.join(
            base_path,
            dataset,
            method,
            f"mean_accuracy_{seed}_{min_samples}_{num_splits}_{K}.json",
        )
        if not os.path.exists(log_path):
            return None
        with open(log_path, "r") as f:
            logs = json.load(f)
        try:
            return logs["Mean"][metric_key]
        except:
            return None

    # Load skyline and baseline values once per metric
    skyline_vals = {k: load_constant_metric("skyline", k) for k, _ in metrics}
    baseline_vals = {k: load_constant_metric("baseline", k) for k, _ in metrics}

    for idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axs[idx]
        for method in methods:
            print("Method:", method)
            log_path = os.path.join(
                base_path,
                dataset,
                method,
                f"logs_{thresh}_{seed}_{min_samples}_{num_splits}_{images_per_id}.json",
            )
            if not os.path.exists(log_path):
                continue
            with open(log_path, "r") as f:
                logs = json.load(f)

            budgets = ["1", "2", "3", "4", "5"]
            means = []
            stds = []
            for budget in budgets:
                means.append(logs[budget][str(K)]["Mean"].get(metric_key, 0))
                stds.append(logs[budget][str(K)]["Std. Dev."].get(metric_key, 0))
            budgets = np.array(budgets, dtype=int)
            means = np.array(means)
            stds = np.array(stds)
            ax.plot(budgets, means, label=method, marker="o")
            ax.fill_between(budgets, means - stds, means + stds, alpha=0.2)

        # Add skyline and baseline as horizontal dotted lines
        skyline_val = skyline_vals.get(metric_key)
        if skyline_val is not None:
            ax.axhline(
                y=skyline_val,
                color="green",
                linestyle="--",
                label="Skyline",
                linewidth=2,
            )

        baseline_val = baseline_vals.get(metric_key)
        if baseline_val is not None:
            ax.axhline(
                y=baseline_val,
                color="red",
                linestyle="--",
                label="Baseline",
                linewidth=2,
            )

        ax.set_title(metric_label)
        ax.set_xlabel("Budget")
        ax.set_ylabel(metric_label)
        ax.grid(True)
        ax.legend()

    plt.suptitle(f"Performance on Dataset: {dataset} | K = {K}", fontsize=16)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{dataset}_{K}_{thresh}_metrics.png")
    print(f"Saved to plots/{dataset}_{K}_{thresh}_metrics.png")


# Example usage:
visualize_all_metrics("MPDD", K=1, thresh=0.2)
visualize_all_metrics("MPDD", K=2, thresh=0.2)
visualize_all_metrics("MPDD", K=5, thresh=0.2)

visualize_all_metrics("DogFaceNet", K=1, thresh=0.2)
visualize_all_metrics("DogFaceNet", K=2, thresh=0.2)
visualize_all_metrics("DogFaceNet", K=5, thresh=0.2)

visualize_all_metrics("SeaStarReID2023", K=1, thresh=0.2)
visualize_all_metrics("SeaStarReID2023", K=2, thresh=0.2)
visualize_all_metrics("SeaStarReID2023", K=5, thresh=0.2)

visualize_all_metrics("CatIndividualImages", K=1, thresh=0.2)
visualize_all_metrics("CatIndividualImages", K=2, thresh=0.2)
visualize_all_metrics("CatIndividualImages", K=5, thresh=0.2)

# visualize_all_metrics('MPDD', K=1, thresh=0.15)
# visualize_all_metrics('MPDD', K=3, thresh=0.15)
# visualize_all_metrics('MPDD', K=5, thresh=0.15)

# visualize_all_metrics('DogFaceNet', K=1, thresh=0.15)
# visualize_all_metrics('DogFaceNet', K=3, thresh=0.15)
# visualize_all_metrics('DogFaceNet', K=5, thresh=0.15)

# visualize_all_metrics('SeaStarReID2023', K=1, thresh=0.15)
# visualize_all_metrics('SeaStarReID2023', K=3, thresh=0.15)
# visualize_all_metrics('SeaStarReID2023', K=5, thresh=0.15)

# visualize_all_metrics('CatIndividualImages', K=1, thresh=0.15)
# visualize_all_metrics('CatIndividualImages', K=3, thresh=0.15)
# visualize_all_metrics('CatIndividualImages', K=5, thresh=0.15)
