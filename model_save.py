import json
from pathlib import Path
import pandas as pd


SAVE_ROOT = Path("saved_runs")


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_run(run_name: str, results: dict, config: dict):
    run_dir = SAVE_ROOT / run_name
    _ensure_dir(run_dir)

    # Save config + metrics
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(results["metrics"], f, indent=2)

    # Save dataframes
    results["predictions_df"].to_csv(run_dir / "predictions.csv", index=False)
    results["top_picks_df"].to_csv(run_dir / "top_picks.csv", index=False)
    results["trade_log_df"].to_csv(run_dir / "trade_log.csv", index=False)
    results["portfolio_history_df"].to_csv(run_dir / "portfolio_history.csv", index=False)
    results["feature_importance_df"].to_csv(run_dir / "feature_importance.csv", index=False)
    results["spy_benchmark_df"].to_csv(run_dir / "spy_benchmark.csv", index=False)
    results["training_history_df"].to_csv(run_dir / "training_history.csv", index=False)

    if "holdings_df" in results and results["holdings_df"] is not None:
        results["holdings_df"].to_csv(run_dir / "holdings.csv", index=False)


def load_run(run_name: str):
    run_dir = SAVE_ROOT / run_name
    if not run_dir.exists():
        return None

    with open(run_dir / "config.json", "r") as f:
        config = json.load(f)

    with open(run_dir / "metrics.json", "r") as f:
        metrics = json.load(f)

    results = {
        "metrics": metrics,
        "predictions_df": pd.read_csv(run_dir / "predictions.csv"),
        "top_picks_df": pd.read_csv(run_dir / "top_picks.csv"),
        "trade_log_df": pd.read_csv(run_dir / "trade_log.csv"),
        "portfolio_history_df": pd.read_csv(run_dir / "portfolio_history.csv"),
        "feature_importance_df": pd.read_csv(run_dir / "feature_importance.csv"),
        "spy_benchmark_df": pd.read_csv(run_dir / "spy_benchmark.csv"),
        "training_history_df": pd.read_csv(run_dir / "training_history.csv"),
        "holdings_df": pd.read_csv(run_dir / "holdings.csv") if (run_dir / "holdings.csv").exists() else pd.DataFrame(),
        "config": config,
    }
    return results


def list_saved_runs():
    if not SAVE_ROOT.exists():
        return []
    return sorted([p.name for p in SAVE_ROOT.iterdir() if p.is_dir()])


def load_default_run():
    runs = list_saved_runs()
    if not runs:
        return None
    if "default_model" in runs:
        return load_run("default_model")
    return load_run(runs[-1])