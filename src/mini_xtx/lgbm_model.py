from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from .schema import feature_columns


def _mask_dates(df, start=None, end=None):
    x = df
    d = pl.col("timestamp").dt.date()
    if start:
        x = x.filter(d >= pl.lit(start).str.to_date())
    if end:
        x = x.filter(d <= pl.lit(end).str.to_date())
    return x


def _purge_tail(df: pl.DataFrame, bars: int) -> pl.DataFrame:
    """Drop the last N rows per symbol so targets cannot cross a split boundary."""
    if bars <= 0 or df.is_empty():
        return df
    return (
        df.with_columns([
            pl.int_range(0, pl.len()).over("symbol").alias("__row"),
            pl.len().over("symbol").alias("__n"),
        ])
        .filter(pl.col("__row") < pl.col("__n") - bars)
        .drop(["__row", "__n"])
    )


def _target_horizon(name: str) -> int:
    return int(name.split("_")[-1])


def _target_specs(columns) -> list[tuple[str, str, str]]:
    """Return (target_column, prediction_column, model_stem) specs."""
    specs = []
    for c in set(columns):
        if c.startswith("target_mfe_"):
            h = c.split("_")[-1]
            specs.append((c, f"pred_mfe_{h}", f"model_mfe_h{h}"))
        elif c.startswith("target_mae_"):
            h = c.split("_")[-1]
            specs.append((c, f"pred_mae_{h}", f"model_mae_h{h}"))
        elif c.startswith("target_"):
            h = c.split("_")[-1]
            specs.append((c, f"pred_{h}", f"model_h{h}"))

    order = {"return": 0, "mfe": 1, "mae": 2}

    def key(spec):
        c = spec[0]
        kind = "mfe" if "_mfe_" in c else "mae" if "_mae_" in c else "return"
        return (_target_horizon(c), order[kind])

    return sorted(specs, key=key)


def train_lgbm(
    data: str,
    outdir: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_start: str,
    purge_bars: int | None = None,
    auxiliary_targets: bool = True,
):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(data).sort(["timestamp", "symbol"])
    features = feature_columns(df.columns)
    specs = _target_specs(df.columns)
    if not auxiliary_targets:
        specs = [s for s in specs if "_mfe_" not in s[0] and "_mae_" not in s[0]]

    return_targets = [
        s[0] for s in specs if "_mfe_" not in s[0] and "_mae_" not in s[0]
    ]
    if not return_targets:
        raise ValueError("No return targets found")

    max_h = max(map(_target_horizon, return_targets))
    purge = max_h if purge_bars is None else purge_bars

    train = _purge_tail(_mask_dates(df, end=train_end), purge)
    valid = _purge_tail(_mask_dates(df, start=valid_start, end=valid_end), purge)
    test = _mask_dates(df, start=test_start)

    prediction_cols = [
        c for c in test.columns
        if c.startswith(("ret_fwd_", "ret_exec_fwd_", "mfe_fwd_", "mae_fwd_"))
    ]
    pred = test.select(["symbol", "timestamp"] + prediction_cols)
    metrics = {
        "_meta": {
            "features": features,
            "purge_bars": purge,
            "auxiliary_targets": auxiliary_targets,
        }
    }

    for target, pred_col, model_stem in specs:
        cols = features + [target]
        tr = train.select(cols).drop_nulls()
        va = valid.select(cols).drop_nulls()
        te = test.select(cols).drop_nulls()
        if tr.is_empty() or va.is_empty() or te.is_empty():
            metrics[target] = {"skipped": "empty split after null filtering"}
            continue

        Xtr, ytr = tr.select(features).to_pandas(), tr[target].to_numpy()
        Xva, yva = va.select(features).to_pandas(), va[target].to_numpy()
        Xte, yte = te.select(features).to_pandas(), te[target].to_numpy()

        model = lgb.LGBMRegressor(
            objective="huber",
            n_estimators=4000,
            learning_rate=0.02,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=300,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=7,
            n_jobs=-1,
        )
        model.fit(
            Xtr,
            ytr,
            eval_set=[(Xva, yva)],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        model.booster_.save_model(str(out / f"{model_stem}.txt"))

        pv = model.predict(Xva)
        pt = model.predict(Xte)
        metrics[target] = {
            "best_iteration": int(model.best_iteration_ or model.n_estimators),
            "val_mse": float(mean_squared_error(yva, pv)),
            "test_mse": float(mean_squared_error(yte, pt)),
            "val_ic": float(np.corrcoef(pv, yva)[0, 1]) if len(yva) > 2 else np.nan,
            "test_ic": float(np.corrcoef(pt, yte)[0, 1]) if len(yte) > 2 else np.nan,
        }

        te_keys = (
            test.select(["symbol", "timestamp"] + cols)
            .drop_nulls(cols)
            .select(["symbol", "timestamp"])
            .with_columns(pl.Series(pred_col, pt))
        )
        pred = pred.join(te_keys, on=["symbol", "timestamp"], how="left")

    pred.write_parquet(out / "predictions.parquet")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
