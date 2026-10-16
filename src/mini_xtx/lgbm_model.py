from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from .schema import FEATURES

def _mask_dates(df, start=None, end=None):
    x = df
    d = pl.col("timestamp").dt.date()
    if start:
        x = x.filter(d >= pl.lit(start).str.to_date())
    if end:
        x = x.filter(d <= pl.lit(end).str.to_date())
    return x

def train_lgbm(
    data: str, outdir: str,
    train_end: str, valid_start: str, valid_end: str, test_start: str,
):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(data).sort(["timestamp","symbol"])
    target_cols = sorted([c for c in df.columns if c.startswith("target_")],
                         key=lambda x: int(x.split("_")[-1]))

    train = _mask_dates(df, end=train_end)
    valid = _mask_dates(df, start=valid_start, end=valid_end)
    test = _mask_dates(df, start=test_start)

    pred = test.select(["symbol","timestamp"] + [c for c in test.columns if c.startswith("ret_fwd_")])

    metrics = {}
    for target in target_cols:
        h = target.split("_")[-1]
        cols = FEATURES + [target]
        tr = train.select(cols).drop_nulls()
        va = valid.select(cols).drop_nulls()
        te = test.select(cols).drop_nulls()

        Xtr, ytr = tr.select(FEATURES).to_pandas(), tr[target].to_numpy()
        Xva, yva = va.select(FEATURES).to_pandas(), va[target].to_numpy()
        Xte, yte = te.select(FEATURES).to_pandas(), te[target].to_numpy()

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
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            callbacks=[lgb.early_stopping(200, verbose=False)]
        )
        model.booster_.save_model(str(out / f"model_h{h}.txt"))

        pv = model.predict(Xva)
        pt = model.predict(Xte)
        ic_val = np.corrcoef(pv, yva)[0,1] if len(yva) > 2 else np.nan
        ic_test = np.corrcoef(pt, yte)[0,1] if len(yte) > 2 else np.nan
        metrics[target] = {
            "best_iteration": int(model.best_iteration_ or model.n_estimators),
            "val_mse": float(mean_squared_error(yva, pv)),
            "test_mse": float(mean_squared_error(yte, pt)),
            "val_ic": float(ic_val),
            "test_ic": float(ic_test),
        }

        # Align predictions by symbol/timestamp, not by row assumption.
        te_keys = test.select(["symbol","timestamp",target]).drop_nulls([target]).select(["symbol","timestamp"])
        te_keys = te_keys.with_columns(pl.Series(f"pred_{h}", pt))
        pred = pred.join(te_keys, on=["symbol","timestamp"], how="left")

    pred.write_parquet(out / "predictions.parquet")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
