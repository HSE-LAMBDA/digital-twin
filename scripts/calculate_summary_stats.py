from digital_twin.performance_metrics import ape, meape, seape, aggregate_loads
from digital_twin.performance_metrics import mmd_rbf, frechet_distance  as fd
from digital_twin.visulization.plots import Figures
meape_iops_mean, meape_iops_std = aggregate_loads(
        ids[ids.isin(test_ids)].values, y_test.values[:, 0], y_pred[:, 0], meape
    )
    meape_lat_mean, meape_lat_std = aggregate_loads(
        ids[ids.isin(test_ids)].values, y_test.values[:, 1], y_pred[:, 1], meape
    )

    seape_iops_mean, seape_iops_std = aggregate_loads(
        ids[ids.isin(test_ids)].values, y_test.values[:, 0], y_pred[:, 0], seape
    )
    seape_lat_mean, seape_lat_std = aggregate_loads(
        ids[ids.isin(test_ids)].values, y_test.values[:, 1], y_pred[:, 1], seape
    )

    with open(model_checkpoint_path, "wb") as f:
        pickle.dump(model, f)
    
    return {
        "MEAPE_IOPS": {"mean": meape_iops_mean, "std": meape_iops_std},
        "MEAPE_LAT": {"mean": meape_lat_mean, "std": meape_lat_std},
        "SEAPE_IOPS": {"mean": seape_iops_mean, "std": seape_iops_std},
        "SEAPE_LAT": {"mean": seape_lat_mean, "std": seape_lat_std},
    }
   
   def metrics_eval(target: np.array, prediction: np.array):
    metrics = {"mmd_rbf": mmd_rbf(target, prediction), "FD": fd(target, prediction)}
    return metrics


def plot_figures(df, grouped_indices):
    for i, idx in enumerate(grouped_indices):
        sub_df = df.loc[idx][POOL_INDEPENDENT_VARS].drop_duplicates().to_dict('record')
        assert len(sub_df) == 1, "More than one unique combination of independent variables"
        title = ", ".join([f'{k}={v}' for k, v in sub_df[0].items()])
        pgf = Figures(df.loc[idx])
        pgf.plot_iops_latency(
            title,
            save_path=f"figures/{title}.png",
            save_kwargs={"dpi": 300},
        ) 

metrics = metrics_eval(
        test_set_predictions[["iops", "lat"]].values,
        test_set_predictions[["generated_iops", "generated_lat"]].values,
    )
    logger.info(metrics)
    plot_figures(test_set_predictions, test_set_indices)