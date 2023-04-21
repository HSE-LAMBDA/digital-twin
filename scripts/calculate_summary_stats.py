import argparse
import uuid

import logging
import json
import numpy as np
import pandas as pd

from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path

from digital_twin.performance_metrics import ape, meape, seape, aggregate_loads
from digital_twin.performance_metrics import mmd_rbf, frechet_distance as fd
from digital_twin.visulization.plots import Figures



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
    parser.add_argument("--make-plots", type=bool, default=True)
    parser.add_argument("--save-path", type=Path, default=None)
    return parser.parse_args()


def transpose(l):
    return list(map(list, zip(*l)))


def nice_format(x: dict):
    return {k: f"{mean:.2f} ± {std:.2f}" for k, (mean, std) in x.items()}


def metrics_eval(target: np.array, prediction: np.array, skip: list = []):
    target_iops, target_lat, gen_iops, gen_lat = (
        target[:, 0],
        target[:, 1],
        prediction[:, 0],
        prediction[:, 1],
    )
    metrics_fn = {
        "MMD (RBF)": (mmd_rbf, (target, prediction)),
        "FD": (fd, (target, prediction)),
        "MEAPE_IOPS": (meape, (target_iops, gen_iops)),
        "MEAPE_LAT": (meape, (target_lat, gen_lat)),
        "SEAPE_IOPS": (seape, (target_iops, gen_iops)),
        "SEAPE_LAT": (seape, (target_lat, gen_lat)),
    }
    return {name: fn(*args) for (name, (fn, args))  in metrics_fn.items() if not name in skip}


def caclulate_stats(df, conds, title=None, save_path=None, no_stats=False, plot=True):
    if plot:
        plotter = Figures(df, filter_outliers=True)
        fig = plotter.plot_iops_latency(title=title, conds=conds, save_path=save_path)
    if not no_stats:
        metrics = metrics_eval(
            df[["iops", "lat"]].values, df[["gen_iops", "gen_lat"]].values
        )
        return metrics


def get_aggregated_stats(dfs, save_path=None):
    metrics = {}
    for df, file in tqdm(zip(dfs, files)):
        _save_path = save_path or file.parent.parent.parent
        metrics[file.stem] = caclulate_stats(
            df,
            conds=None,
            title=file.stem,
            save_path=_save_path
            / "figures"
            / file.parent.name
            / f"agg_{file.stem}.pdf",
            no_stats=True
        )


def get_name_from_cond(cond: dict):
    return {NICE_NAMES.get(k, k): NICE_NAMES.get(v, str(v)) for k, v in cond.items()}


def agg(x):
    return np.mean(x[0]), np.sqrt(np.mean(np.power(x[1], 2)))


def main(files: list[Path]):
    dfs = [pd.read_csv(f) for f in files]
    grouped_dfs = [
        df.groupby(
            columns := df.drop(
                ["iops", "lat", "gen_iops", "gen_lat"], axis=1
            ).columns.tolist()
        )
        for df in dfs
    ]
    
    get_aggregated_stats(dfs, args.save_path)
    for grouped_df, file in zip(grouped_dfs, files):
        _save_path = args.save_path or file.parent.parent.parent
        _save_path.mkdir(exist_ok=True, parents=True)
        def _inner(g, df):
            conditions = get_name_from_cond(dict(zip(columns, g)))
            this_uuid = uuid.uuid5(
                uuid.NAMESPACE_OID, json.dumps(conditions, sort_keys=True)
            )
            metrics = caclulate_stats(
                df,
                conditions,
                title=file.stem,
                save_path=_save_path
                / "figures"
                / file.parent.name
                / file.stem
                / f"{this_uuid}.pdf",
                plot=args.make_plots
            )
            return dict(UUID=str(this_uuid), **metrics, **conditions)
        results = Parallel(n_jobs=-1)(delayed(_inner)(g, df) for g, df in grouped_df)

        this_group_df = pd.DataFrame.from_dict(results)
        root_path = _save_path / "metrics" / file.parent.name
        root_path.mkdir(parents=True, exist_ok=True)
        this_group_df.to_csv(root_path / f"metrics_{file.stem}.csv", index=False)
        this_group_df[['MMD (RBF)', 'FD', 'MEAPE_IOPS', 'MEAPE_LAT', 'SEAPE_IOPS', 'SEAPE_LAT']].agg(agg).T.to_csv(
            root_path / f"metrics_agg_{file.stem}.csv",
            index=True,
            header=["mean", "std"],
        )


if __name__ == "__main__":
    args = parse_args()
    files = list(args.results.rglob("*_pred_*.csv"))
    metrics = main(files)
    logger.info(metrics)