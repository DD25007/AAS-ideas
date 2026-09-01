import os
import json
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

def parse_log_file(filepath, K):
    with open(filepath, 'r') as f:
        logs = json.load(f)

    if str(K) not in logs:
        return None

    metrics = ['Top-1 Accuracy', 'Top-3 Accuracy', 'Top-5 Accuracy', 'Top-10 Accuracy', 'AP', 'INP']
    parsed = {}
    for metric in metrics:
        try:
            mean = logs[str(K)]['Mean'][metric]
            std = logs[str(K)]['Std. Dev.'][metric]
            parsed[metric] = (mean, std)
        except KeyError:
            continue
    return parsed

def visualize_metrics(dataset, K, root_dir='outputs/pretrained_models', save_path='visualization.png'):
    dataset_dir = os.path.join(root_dir, dataset)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found.")

    metrics = ['Top-1 Accuracy', 'Top-3 Accuracy', 'Top-5 Accuracy', 'Top-10 Accuracy', 'AP', 'INP']
    method_logs = {}

    for method in sorted(os.listdir(dataset_dir)):
        method_path = os.path.join(dataset_dir, method)
        if not os.path.isdir(method_path):
            continue
        log_files = sorted(glob(os.path.join(method_path, 'logs_*.json')))
        if not log_files:
            continue

        budgets = []
        metric_data = {m: {'mean': [], 'std': []} for m in metrics}

        for log_file in log_files:
            filename = os.path.basename(log_file)
            try:
                _, budget_str, *_ = filename.split('_')
                budget = int(budget_str)
            except:
                continue

            parsed = parse_log_file(log_file, K)
            if parsed is None:
                continue

            budgets.append(budget)
            for m in metrics:
                mean, std = parsed.get(m, (None, None))
                metric_data[m]['mean'].append(mean)
                metric_data[m]['std'].append(std)

        if budgets:
            sort_idx = np.argsort(budgets)
            budgets = np.array(budgets)[sort_idx]
            for m in metrics:
                metric_data[m]['mean'] = np.array(metric_data[m]['mean'])[sort_idx]
                metric_data[m]['std'] = np.array(metric_data[m]['std'])[sort_idx]
            method_logs[method] = (budgets, metric_data)

    # Plotting
    ncols = 3
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axs = axs.flatten()

    for i, metric in enumerate(metrics):
        ax = axs[i]
        for method, (budgets, metric_data) in method_logs.items():
            mean = metric_data[metric]['mean']
            std = metric_data[metric]['std']
            ax.plot(budgets, mean, label=method)
            ax.fill_between(budgets, mean - std, mean + std, alpha=0.2)
        ax.set_title(metric)
        ax.set_xlabel('Budget')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)

    for j in range(len(metrics), len(axs)):
        fig.delaxes(axs[j])

    plt.suptitle(f'Dataset: {dataset} | K = {K}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")

# Example usage
if __name__ == "__main__":
    visualize_metrics(dataset='CUB200', K=5)