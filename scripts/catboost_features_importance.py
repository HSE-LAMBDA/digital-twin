import argparse
from pathlib import Path
import logging
from catboost import CatBoostRegressor
import pandas as pd
from digital_twin.models.density_estimation.grouper import Grouper
import matplotlib.pyplot as plt

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
    "n_raid_parity_disks": "Num RAID parity pisks",
    "total_disk": "Total disk",
    "n_raid_disks": "Num RAID disks"
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, default="models_checkpoints")
    parser.add_argument(
        "--data", type=Path, default="dataset", help="Path to data (csv file)."
    )
    parser.add_argument(
        "--results", type=Path, default="results", help="Path to save Shap plots"
    )

    return parser.parse_args()

def find_corres_data(checkpoint: Path, data: Path):
    """
    Finds the correspondings data to a given checkpoint
    """
    name = checkpoint.stem.split('catboost_')[-1]
    return list(data.rglob(f'{name}.*'))[-1]

def preprocess_X(df):
    df.load_type = df.load_type.factorize()[0]
    df.io_type = df.io_type.factorize()[0]
    df = df.drop(['offset', 'raid', 'device_type'], axis=1)
    return df

def main(file, df, save_dir):
    model = CatBoostRegressor()
    model.load_model(file)
    X = preprocess_X(Grouper.featurize(df.drop(['iops', 'lat', 'id'], axis=1)))
    
    def shap_values():
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, show=False)
        save_dir = save_dir / 'shap_plots'
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f'shap_{file.stem}.pdf')
    
    def plot_feature_importance(save_dir):
        
        feature_importances = model.get_feature_importance(prettified=True).sort_values('Importances', ascending=False)
        feature_importances['Feature Id'] = feature_importances['Feature Id'].map(lambda x: NICE_NAMES.get(x, x))
        plt.figure(tight_layout=True)
        plt.barh(feature_importances['Feature Id'], feature_importances['Importances'], color='darkred', zorder=99)
        ax = plt.gca()
        ax.invert_yaxis()
        ax.yaxis.grid(color='gray', linestyle='-', linewidth=0.1, alpha=0.75)

        plt.xticks(rotation=90)
        plt.xlabel(r'Importance (%)')
        plt.title('Feature Importance')
        save_dir = save_dir / 'features_importance_plots'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_dir / f'feat_imp_{file.stem}.pdf')
    plot_feature_importance(save_dir)

    


if __name__ == "__main__":
    args = parse_args()
    checkpoints = list(args.checkpoints.rglob("catboost_*.cbm"))
    train_files = list(args.data.rglob("train_*.csv")) 
    test_files = list(args.data.rglob("test_*.csv"))
    
    for checkpoint in checkpoints:
        data_df = pd.read_csv(find_corres_data(checkpoint, args.data))
        main(checkpoint, data_df, args.results)

