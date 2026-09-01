import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Path to the base directory
base_dir = "outputs/pretrained_models"

# Metrics to extract
accuracy_keys = ["Top-1 Accuracy", "Top-3 Accuracy", "Top-5 Accuracy"]
identity_keys = ["Number of pseudo-identities", "Number of gt-identities:"]
annotation_keys = ["Number of +ve Annotations", "Number of -ve Annotations"]

# Plotting style
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
# linestyles = ['-', '--', '-.', ':']

# Loop over datasets
thresh = 0.2
for dataset_name in os.listdir(base_dir):
    # if dataset_name not in ['MPDD', 'DogFaceNet']:
    #     continue
    if dataset_name in ["skyline", "baseline", "plots"]:
        continue
    dataset_path = os.path.join(base_dir, dataset_name)
    if not os.path.isdir(dataset_path):
        continue

    method_results = {}

    # Loop over each method
    for method_name in os.listdir(dataset_path):
        method_path = os.path.join(dataset_path, method_name, f"logs_{thresh}_0_5.json")
        if not os.path.exists(method_path):
            continue

        # Load JSON
        with open(method_path, "r") as f:
            log = json.load(f)

        # Extract accuracy per step
        steps = sorted([int(k) for k in log.keys()])
        results = {k: [] for k in accuracy_keys}
        identities = {k: [] for k in identity_keys}
        annotations = {k: [] for k in annotation_keys}

        for step in steps:
            for acc_key in accuracy_keys:
                results[acc_key].append(log[str(step)][acc_key]["mean"])
            for ann_key in annotation_keys:
                if ann_key in log[str(step)]:
                    annotations[ann_key].append(log[str(step)][ann_key])
                else:
                    annotations[ann_key].append(None)  # If not available, mark None
            for id_key in identity_keys:
                if id_key in log[str(step)]:
                    identities[id_key].append(log[str(step)][id_key])
                else:
                    identities[id_key].append(None)  # If not available, mark None

        method_results[method_name] = {
            "steps": steps,
            "accuracies": results,
            "identities": identities,
            "annotations": annotations,
        }

    # Plot per accuracy type (Top-1, Top-3, Top-5)
    for acc_key in accuracy_keys:
        plt.figure(figsize=(8, 5))
        for i, method in enumerate(["random", "LBAS", "MinMax", "EC", "CC", "ConSam"]):
            try:
                res = method_results[method]
            except:
                method = method + "_no_outliers"
                try:
                    res = method_results[method]
                except:
                    continue
            # if method in ["LBAS", "GBAS"]:
            #     continue
            plt.plot(
                # np.array(res["annotations"]["Number of +ve Annotations"]) + np.array(res["annotations"]["Number of -ve Annotations"]),
                res["steps"],
                res["accuracies"][acc_key],
                label=method.replace("_no_outliers", ""),
                marker="o",
            )

        method_path = os.path.join(dataset_path, "skyline", "mean_accuracy_0_5.json")
        if not os.path.exists(method_path):
            continue
        # Load JSON
        with open(method_path, "r") as f:
            log = json.load(f)

        plt.axhline(
            y=log["Mean"][acc_key], label="Skyline", linestyle="--", color="green"
        )
        method_path = os.path.join(dataset_path, "baseline", "mean_accuracy_0_5.json")
        if not os.path.exists(method_path):
            continue
        # Load JSON
        with open(method_path, "r") as f:
            log = json.load(f)
        plt.axhline(
            y=log["Mean"][acc_key], label="Baseline", linestyle="--", color="red"
        )

        plt.title(f"{dataset_name} - {acc_key}")
        plt.xlabel("Annotation Step")
        plt.ylabel("Accuracy (%)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        os.makedirs(f"{base_dir}/plots/{thresh}/{acc_key}", exist_ok=True)
        plt.savefig(
            f"{base_dir}/plots/{thresh}/{acc_key}/{dataset_name}_{acc_key.replace(' ', '_')}.png"
        )
        plt.close()

    # Plot pseudo-identities vs ground-truth identities
    plt.figure(figsize=(10, 6))
    width = 0.35  # Bar width
    methods_for_bar = []

    pseudo_values = []
    gt_values = []

    for method in ["random", "LBAS", "MinMax", "EC", "CC", "ConSam"]:
        if method_results.get(method) is None:
            continue
        pseudo = method_results[method]["identities"]["Number of pseudo-identities"]
        gt = method_results[method]["identities"]["Number of gt-identities:"]

        # Only take the last annotation step value
        if pseudo and gt and pseudo[-1] is not None and gt[-1] is not None:
            pseudo_values.append(pseudo[-1])
            gt_values.append(gt[-1])
            methods_for_bar.append(method)

    x = np.arange(len(methods_for_bar))
    plt.bar(x - width / 2, gt_values, width, label="GT-Identities")
    plt.bar(x + width / 2, pseudo_values, width, label="Pseudo-Identities")

    plt.xticks(x, methods_for_bar, rotation=45)
    plt.ylabel("Number of Identities")
    plt.title(f"{dataset_name} - Count of Sampled Identities at 5n Budget")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    os.makedirs(f"{base_dir}/plots/{thresh}/identities", exist_ok=True)
    plt.savefig(f"{base_dir}/plots/{thresh}/identities/{dataset_name}_identities.png")
    plt.close()

print("✅ Plots saved!")
