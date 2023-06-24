import argparse
from pathlib import Path
import logging
import json
import numpy as np
import pandas as pd
from digital_twin.performance_metrics import ape, meape, seape, aggregate_loads
from digital_twin.performance_metrics import mmd_rbf, frechet_distance as fd
from digital_twin.visulization.plots import Figures
from tqdm import tqdm
import uuid

NICE_NAMES = {
    "block_size": "Block size",
    "n_jobs": "Number of jobs",
    "iodepth": "IO depth",
    "read_fraction": "Read fraction",
    "load_type": "Load type",
    "io_type": "IO type",
    "raid": "RAID",
    "n_disks": "Number of disks",
    "device_type": "Device type",
    "offset": "Offset",
    "id": "ID",
    "sequential": "Sequential",
    "read": "Read",
    "rand": "Rand",
    "random": "Random",
    "ssd": "SSD",
    "hdd": "HDD",
}

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


def transpose(l):
    return list(map(list, zip(*l)))


def nice_format(x: dict):
    return {k: f"{mean:.2f} ± {std:.2f}" for k, (mean, std) in x.items()}


from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
def remove_outliers(x, y, q=0.05):
    ee = EllipticEnvelope(contamination=q)
    #ee = IsolationForest(contamination=q)
    xy = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    lab = ee.fit_predict(xy)
    xy = xy[lab==1]
    # print(len(y), len(xy))
    return xy[:, 0], xy[:, 1]


def metrics_eval(target: np.array, prediction: np.array):
    target_iops, target_lat, gen_iops, gen_lat = (
        target[:, 0],
        target[:, 1],
        prediction[:, 0],
        prediction[:, 1],
    )
    target_iops, target_lat = remove_outliers(target_iops, target_lat)
    gen_iops, gen_lat = remove_outliers(gen_iops, gen_lat)
    n = min(len(target_iops), len(gen_iops))
    target_iops, target_lat = target_iops[:n], target_lat[:n]
    gen_iops, gen_lat = gen_iops[:n], gen_lat[:n]
    target = np.concatenate((target_iops.reshape(-1, 1), target_lat.reshape(-1, 1)), axis=1)
    prediction = np.concatenate((gen_iops.reshape(-1, 1), gen_lat.reshape(-1, 1)), axis=1)
    return {
        "MMD (RBF)": mmd_rbf(target, prediction),
        "FD": fd(target, prediction),
        "MEAPE_IOPS": meape(target_iops, gen_iops),
        "MEAPE_LAT": meape(target_lat, gen_lat),
        "SEAPE_IOPS": seape(target_iops, gen_iops),
        "SEAPE_LAT": seape(target_lat, gen_lat),
    }


def caclulate_stats(df, conds, title=None, save_path=None, no_stats=False):
    plotter = Figures(df, filter_outliers=True)
    fig = plotter.plot_iops_latency(title=title, conds=conds, save_path=save_path)
    if not no_stats:
        metrics = metrics_eval(
            df[["iops", "lat"]].values, df[["gen_iops", "gen_lat"]].values
        )
        return metrics


def get_aggregated_stats(dfs):
    metrics = {}
    for df, file in tqdm(zip(dfs, files)):
        metrics[file.stem] = caclulate_stats(
            df,
            conds=None,
            title=file.stem,
            save_path=file.parent.parent.parent
            / "figures"
            / file.parent.name
            / f"agg_{file.stem}.pdf",
            no_stats=True
        )


def get_name_from_cond(cond: dict):
    return {NICE_NAMES.get(k, k): NICE_NAMES.get(v, str(v)) for k, v in cond.items()}


from sklearn.utils import resample
def _bootstrap_metric(x, n_iters=1000):
    scores = []
    for i in range(n_iters):
        x_boot = resample(x, random_state=i+1)
        scores.append(x_boot.mean())
    scores = np.array(scores)
    return scores.mean(axis=0), scores.std(axis=0)


def agg(x):
    t_x = np.array(transpose(x))[0]
    q = np.quantile(t_x, 0.95)
    mu, std = _bootstrap_metric(t_x[t_x<=q])
    return mu, std


def main(files: list[Path]):
    dfs = [pd.read_csv(f) for f in files]
    groupded_dfs = [
        df.groupby(
            columns := df.drop(
                ["iops", "lat", "gen_iops", "gen_lat"], axis=1
            ).columns.tolist()
        )
        for df in dfs
    ]
    
    # get_aggregated_stats(dfs)
    for grouped_df, file in zip(groupded_dfs, files):
        results = []
        for g, df in grouped_df:
            conditions = get_name_from_cond(dict(zip(columns, g)))
            this_uuid = uuid.uuid5(
                uuid.NAMESPACE_OID, json.dumps(conditions, sort_keys=True)
            )
            metrics = caclulate_stats(
                df,
                conditions,
                title=file.stem,
                save_path=file.parent.parent.parent
                / "figures"
                / file.parent.name
                / file.stem
                / f"{this_uuid}.pdf",
            )
            results.append(dict(UUID=str(this_uuid), **metrics, **conditions))
        this_group_df = pd.DataFrame.from_dict(results)
        root_path = file.parent.parent.parent / "metrics" / file.parent.name
        root_path.mkdir(parents=True, exist_ok=True)
        this_group_df.to_csv(root_path / f"metrics_{file.stem}.csv", index=False)
        this_group_df.agg(
            {
                "MMD (RBF)": agg,
                "FD": agg,
                "MEAPE_IOPS": agg,
                "MEAPE_LAT": agg,
                "SEAPE_IOPS": agg,
                "SEAPE_LAT": agg,
            },
        ).to_csv(
            root_path / f"metrics_agg_{file.stem}.csv",
            index=True,
            header=["(mean, std)"],
        )


if __name__ == "__main__":
    args = parse_args()
    files = list(args.results.rglob("*.csv"))
    metrics = main(files)
    logger.info(metrics)