from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .schema import feature_columns


def fit_ridge(train: pl.DataFrame, target: str, alpha: float = 10.0):
    features = feature_columns(train.columns)
    cols = features + [target]
    x = train.select(cols).drop_nulls([target])
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])
    model.fit(x.select(features).to_pandas(), x[target].to_numpy())
    return model, features


def predict_ridge(model, features: list[str], df: pl.DataFrame) -> np.ndarray:
    return model.predict(df.select(features).to_pandas())


def train_ridge(
    data: str,
    outdir: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_start: str,
    alpha: float = 10.0,
):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(data).sort(["timestamp", "symbol"])
    d = pl.col("timestamp").dt.date()
    train = df.filter(d <= pl.lit(train_end).str.to_date())
    valid = df.filter(
        (d >= pl.lit(valid_start).str.to_date()) &
        (d <= pl.lit(valid_end).str.to_date())
    )
    test = df.filter(d >= pl.lit(test_start).str.to_date())

    targets = sorted(
        [c for c in df.columns if c.startswith("target_") and "_mfe_" not in c and "_mae_" not in c],
        key=lambda x: int(x.split("_")[-1]),
    )
    pred = test.select(
        ["symbol", "timestamp"] +
        [c for c in test.columns if c.startswith(("ret_fwd_", "ret_exec_fwd_", "mfe_fwd_", "mae_fwd_"))]
    )
    metrics = {}
    for target in targets:
        h = target.split("_")[-1]
        model, features = fit_ridge(train, target, alpha)
        va = valid.drop_nulls([target])
        te = test.drop_nulls([target])
        pv = predict_ridge(model, features, va)
        pt = predict_ridge(model, features, te)
        yv, yt = va[target].to_numpy(), te[target].to_numpy()
        metrics[target] = {
            "val_ic": float(np.corrcoef(pv, yv)[0, 1]) if len(yv) > 2 else float("nan"),
            "test_ic": float(np.corrcoef(pt, yt)[0, 1]) if len(yt) > 2 else float("nan"),
        }
        joblib.dump({"model": model, "features": features}, out / f"ridge_h{h}.joblib")
        keys = te.select(["symbol", "timestamp"]).with_columns(pl.Series(f"pred_{h}", pt))
        pred = pred.join(keys, on=["symbol", "timestamp"], how="left")

    pred.write_parquet(out / "predictions.parquet")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
