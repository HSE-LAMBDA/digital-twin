import argparse
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from digital_twin.performance_metrics import ape, meape, seape, aggregate_loads
from digital_twin.performance_metrics import mmd_rbf, frechet_distance as fd
from digital_twin.visulization.plots import Figures
from tqdm import tqdm

logger = logging.getLogger(__name__)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# save logs to file and console
if not Path("logs").exists():
    Path("logs").mkdir(parents=True)
logger.addHandler(logging.FileHandler("results/summary.log"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default="results/predictions")
    return parser.parse_args()


# meape_iops_mean, meape_iops_std = aggregate_loads(
#         ids[ids.isin(test_ids)].values, y_test.values[:, 0], y_pred[:, 0], meape
#     )
#     meape_lat_mean, meape_lat_std = aggregate_loads(
#         ids[ids.isin(test_ids)].values, y_test.values[:, 1], y_pred[:, 1], meape
#     )

#     seape_iops_mean, seape_iops_std = aggregate_loads(
#         ids[ids.isin(test_ids)].values, y_test.values[:, 0], y_pred[:, 0], seape
#     )
#     seape_lat_mean, seape_lat_std = aggregate_loads(
#         ids[ids.isin(test_ids)].values, y_test.values[:, 1], y_pred[:, 1], seape
#     )

#     with open(model_checkpoint_path, "wb") as f:
#         pickle.dump(model, f)

#     return {
#         "MEAPE_IOPS": {"mean": meape_iops_mean, "std": meape_iops_std},
#         "MEAPE_LAT": {"mean": meape_lat_mean, "std": meape_lat_std},
#         "SEAPE_IOPS": {"mean": seape_iops_mean, "std": seape_iops_std},
#         "SEAPE_LAT": {"mean": seape_lat_mean, "std": seape_lat_std},
#     }


def nice_format(x: dict):
    return {k: f"{mean:.2f} ± {std:.2f}" for k, (mean, std) in x.items()}


def metrics_eval(target: np.array, prediction: np.array):
    target_iops, target_lat, gen_iops, gen_lat = (
        target[:, 0],
        target[:, 1],
        prediction[:, 0],
        prediction[:, 1],
    )
    return {
        "MMD (RBF)": mmd_rbf(target, prediction),
        "FD": fd(target, prediction),
        "MEAPE_IOPS": meape(target_iops, gen_iops),
        "MEAPE_LAT": meape(target_lat, gen_lat),
        "SEAPE_IOPS": seape(target_iops, gen_iops),
        "SEAPE_LAT": seape(target_lat, gen_lat),
    }


def caclulate_stats(df, title=None, save_path=None):
    metrics = nice_format(
        metrics_eval(df[["iops", "lat"]].values, df[["gen_iops", "gen_lat"]].values)
    )
    plotter = Figures(df, filter_outliers=False)
    fig = plotter.plot_iops_latency(title=title, save_path=save_path)
    return metrics


def main(files: list[Path]):
    metrics = {}
    dfs = [pd.read_csv(f) for f in files]
    groupded_dfs = [
        df.groupby(
            df.drop(["iops", "lat", "gen_iops", "gen_lat"], axis=1).columns.tolist()
        )
        for df in dfs
    ]
    

    for df, file in tqdm(zip(dfs, files)):
        metrics[file.stem] = caclulate_stats(
            df,
            title=file.stem,
            save_path=file.parent.parent.parent
            / "figures"
            / file.parent.name
            / f"{file.stem}.pdf",
        )
    metrics = pd.DataFrame.from_records(metrics).T
    metrics.to_csv("results/metrics.csv")


if __name__ == "__main__":
    args = parse_args()
    files = list(args.results.rglob("*.csv"))
    metrics = main(files)
    logger.info(metrics)
