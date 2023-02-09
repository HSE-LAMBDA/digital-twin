import matplotlib.pyplot as plt
from pathlib import Path

class Figures:
    def __init__(self, data, filter_outliers=True):
        self.data = data
        if filter_outliers:
            upper_threshold = data.quantile(0.95)
            lower_threshold = data.quantile(0.05)
            self.data = data[(data > lower_threshold) & (data < upper_threshold)]

    def plot_iops_latency(self, title, save_path=None, save_kwargs=None):
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        axs[0].scatter(
            self.data["iops"],
            self.data["lat"],
            color="blue",
            label="Real Data",
            marker="x",
            alpha=0.5,
        )
        axs[0].scatter(
            self.data["generated_iops"],
            self.data["generated_lat"],
            color="darkred",
            label="Generated Data",
            marker="o",
            alpha=0.5,
        )
        axs[0].set_xlabel("IOPS")
        axs[0].set_ylabel("Latency (ms)")
        axs[0].legend()

        axs[1].hist(
            self.data["generated_iops"],
            color="darkred",
            label="Generated IOPS",
            bins=60,
            alpha=0.5,
        )
        axs[1].hist(
            self.data["iops"],
            color="blue",
            label="Ground Truth IOPS",
            bins=60,
            alpha=0.5,
        )

        axs[1].set_xlabel("IOPS")
        axs[1].legend()

        axs[2].hist(
            self.data["generated_lat"],
            color="darkred",
            label="Generated Latency",
            bins=60,
            alpha=0.5,
        )
        axs[2].hist(
            self.data["lat"],
            color="blue",
            label="Ground Truth Latency",
            bins=60,
            alpha=0.5,
        )
        axs[2].set_xlabel("Latency (ms)")
        axs[2].legend()
        fig.suptitle(title)
        if save_path is not None:
            if not Path(save_path).parent.exists():
                Path(save_path).parent.mkdir(parents=True)
            fig.savefig(save_path, **save_kwargs)
