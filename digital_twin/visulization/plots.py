import matplotlib.pyplot as plt
from pathlib import Path
import textwrap as twp


class Figures:
    def __init__(self, data, filter_outliers=True):
        self.data = data
        if filter_outliers:
            upper_threshold = data.quantile(0.95)
            lower_threshold = data.quantile(0.05)
            self.data = data[(data > lower_threshold) & (data < upper_threshold)]

    def plot_iops_latency(self, title, conds, save_path=None, save_kwargs={}):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        axs[0].scatter(
            self.data["iops"],
            self.data["lat"],
            color="blue",
            label="Real Data",
            marker="x",
            alpha=0.5,
        )
        axs[0].scatter(
            self.data["gen_iops"],
            self.data["gen_lat"],
            color="darkred",
            label="Generated Data",
            marker="o",
            alpha=0.5,
        )
        axs[0].set_xlabel("IOPS")
        axs[0].set_ylabel("Latency (ms)")
        axs[0].legend()

        axs[1].hist(
            self.data["gen_iops"],
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
            self.data["gen_lat"],
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
        # axs[2].legend()
        for ax in axs:
            ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        fig.suptitle(title)
        ax4 = fig.add_axes([0.86, 0.22, 0.2, 0.2], alpha=0.8)
        ax4.axis("off")
        if conds is not None:
            table = ax4.table(
                [[str(v)] for v in conds.values()],
                rowLabels=list(conds.keys()),
                loc="top",
                colWidths=[0.3] * len(conds),
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            ax4.legend().set_visible(False)

        if save_path is not None:
            if not Path(save_path).parent.exists():
                Path(save_path).parent.mkdir(parents=True)
            fig.savefig(save_path, **save_kwargs)
        return fig
