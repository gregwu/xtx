from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import (
    cross_sectional_metrics,
    decile_summary,
    nonoverlap_sample,
    tail_metrics,
)
from .schema import feature_columns


@dataclass(frozen=True)
class Fold:
    train_end: str
    test_start: str
    test_end: str


def monthly_folds(df: pl.DataFrame, start: str, end: str | None = None) -> list[Fold]:
    dates = (
        df.select(pl.col("timestamp").dt.date().alias("d"))
        .unique()
        .sort("d")
        .filter(pl.col("d") >= pl.lit(start).str.to_date())
    )
    if end:
        dates = dates.filter(pl.col("d") <= pl.lit(end).str.to_date())
    if dates.is_empty():
        return []

    months = (
        dates.with_columns(pl.col("d").dt.truncate("1mo").alias("m"))
        .group_by("m")
        .agg([pl.col("d").min().alias("start"), pl.col("d").max().alias("end")])
        .sort("m")
    )
    folds = []
    for row in months.iter_rows(named=True):
        train_end = row["start"] - __import__("datetime").timedelta(days=1)
        folds.append(Fold(str(train_end), str(row["start"]), str(row["end"])))
    return folds


def _purge_tail(df: pl.DataFrame, bars: int) -> pl.DataFrame:
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


def _fit_predict_ridge(train, test, features, target, alpha):
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])
    tr = train.select(features + [target]).drop_nulls([target])
    te = test.select(features + [target]).drop_nulls([target])
    model.fit(tr.select(features).to_pandas(), tr[target].to_numpy())
    return te, model.predict(te.select(features).to_pandas())


def _fit_predict_lgbm(train, test, features, target):
    tr = train.select(features + [target]).drop_nulls()
    te = test.select(features + [target]).drop_nulls()
    model = lgb.LGBMRegressor(
        objective="huber",
        n_estimators=1200,
        learning_rate=0.025,
        num_leaves=63,
        min_child_samples=300,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=7,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(tr.select(features).to_pandas(), tr[target].to_numpy())
    return te, model.predict(te.select(features).to_pandas())


def run_walkforward(
    data: str,
    outdir: str,
    horizon: int,
    start: str,
    end: str | None = None,
    models: tuple[str, ...] = ("ridge", "lgbm"),
    purge_bars: int | None = None,
    ridge_alpha: float = 10.0,
    eval_stride: int | None = None,
):
    """
    Anchored monthly walk-forward.

    Every prediction is genuinely OOS: a month is scored only by a model trained
    on dates before that month. The training tail is purged by H bars by default.
    Final economic diagnostics use a non-overlapping timestamp stream by default.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(data).sort(["timestamp", "symbol"])
    features = feature_columns(df.columns)
    target = f"target_{horizon}"
    exec_ret = f"ret_exec_fwd_{horizon}"
    if target not in df.columns:
        raise ValueError(f"{target} missing")
    if exec_ret not in df.columns:
        exec_ret = f"ret_fwd_{horizon}"

    folds = monthly_folds(df, start, end)
    purge = horizon if purge_bars is None else purge_bars
    stride = horizon if eval_stride is None else max(1, eval_stride)
    all_outputs = {m: [] for m in models}
    fold_meta = []

    date_expr = pl.col("timestamp").dt.date()
    for i, fold in enumerate(folds, 1):
        train = df.filter(date_expr <= pl.lit(fold.train_end).str.to_date())
        train = _purge_tail(train, purge)
        test = df.filter(
            (date_expr >= pl.lit(fold.test_start).str.to_date()) &
            (date_expr <= pl.lit(fold.test_end).str.to_date())
        )
        if train.is_empty() or test.is_empty():
            continue

        rec = {
            "fold": i,
            "train_end": fold.train_end,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            "train_rows": train.height,
            "test_rows": test.height,
        }

        for name in models:
            if name == "ridge":
                scored, pred = _fit_predict_ridge(train, test, features, target, ridge_alpha)
            elif name == "lgbm":
                scored, pred = _fit_predict_lgbm(train, test, features, target)
            else:
                raise ValueError(f"Unsupported walk-forward model: {name}")

            keep = ["symbol", "timestamp", exec_ret]
            for c in (f"mfe_fwd_{horizon}", f"mae_fwd_{horizon}"):
                if c in scored.columns:
                    keep.append(c)
            fold_pred = scored.select(keep).with_columns(pl.Series("prediction", pred))
            all_outputs[name].append(fold_pred)

            fold_eval = nonoverlap_sample(fold_pred, stride)
            summary, _, _ = cross_sectional_metrics(
                fold_eval, "prediction", exec_ret
            )
            rec[name] = summary
        fold_meta.append(rec)

    summary_all = {
        "horizon": horizon,
        "purge_bars": purge,
        "eval_stride_bars": stride,
        "overlap_avoided": stride >= horizon,
        "folds": fold_meta,
        "models": {},
    }

    for name, chunks in all_outputs.items():
        if not chunks:
            continue
        pred = pl.concat(chunks).sort(["timestamp", "symbol"])
        pcol = f"pred_{horizon}"
        pred = pred.rename({"prediction": pcol})
        pred.write_parquet(out / f"{name}_predictions.parquet")

        full_metric, full_by_time, full_dec = cross_sectional_metrics(pred, pcol, exec_ret)
        nonoverlap = nonoverlap_sample(pred, stride)
        metric, by_time, dec = cross_sectional_metrics(nonoverlap, pcol, exec_ret)
        metric["tails"] = tail_metrics(nonoverlap, pcol, exec_ret)
        metric["fold_count"] = len(chunks)
        metric["full_sample"] = full_metric
        summary_all["models"][name] = metric

        full_by_time.write_parquet(out / f"{name}_ic_by_time_full.parquet")
        full_dec.write_parquet(out / f"{name}_deciles_full.parquet")
        by_time.write_parquet(out / f"{name}_ic_by_time_nonoverlap.parquet")
        dec.write_parquet(out / f"{name}_deciles_nonoverlap.parquet")
        decile_summary(dec).write_parquet(out / f"{name}_decile_summary.parquet")

    (out / "walkforward_summary.json").write_text(json.dumps(summary_all, indent=2))
    return summary_all
