from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import polars as pl


def _zscore(expr: pl.Expr, group: str = "timestamp") -> pl.Expr:
    mu = expr.mean().over(group)
    sd = expr.std().over(group)
    return ((expr - mu) / (sd + 1e-12)).clip(-8, 8)


def build_ensemble(
    inputs: list[str],
    output: str,
    horizon: int,
    weights: list[float] | None = None,
    risk_lambda: float = 0.35,
    mfe_weight: float = 0.25,
):
    """Rank-average model predictions, then form a risk-adjusted opportunity score."""
    if not inputs:
        raise ValueError("At least one prediction file is required")
    if weights is None:
        weights = [1.0 / len(inputs)] * len(inputs)
    if len(weights) != len(inputs):
        raise ValueError("weights must match inputs")
    s = sum(weights)
    weights = [w / s for w in weights]

    base = None
    pred_cols = []
    for i, path in enumerate(inputs):
        df = pl.read_parquet(path)
        p = f"pred_{horizon}"
        if p not in df.columns:
            raise ValueError(f"{p} missing from {path}")
        keep = ["symbol", "timestamp", p]
        if base is None:
            extra = [
                c for c in df.columns
                if c.startswith(("ret_fwd_", "ret_exec_fwd_", "mfe_fwd_", "mae_fwd_"))
            ]
            keep += extra
            base = df.select(list(dict.fromkeys(keep))).rename({p: f"model_pred_{i}"})
        else:
            base = base.join(
                df.select(["symbol", "timestamp", p]).rename({p: f"model_pred_{i}"}),
                on=["symbol", "timestamp"],
                how="inner",
            )
        pred_cols.append(f"model_pred_{i}")

    rank_terms = []
    for c, w in zip(pred_cols, weights):
        rank = pl.col(c).rank("average").over("timestamp") / pl.len().over("timestamp")
        rank_terms.append(rank * w)
    base = base.with_columns(sum(rank_terms).alias("ensemble_rank"))
    base = base.with_columns(
        _zscore(pl.col("ensemble_rank")).alias(f"pred_{horizon}")
    )

    mfe_cols, mae_cols = [], []
    for i, path in enumerate(inputs):
        df = pl.read_parquet(path)
        for kind, acc in (("mfe", mfe_cols), ("mae", mae_cols)):
            c = f"pred_{kind}_{horizon}"
            if c in df.columns:
                alias = f"model_{kind}_{i}"
                base = base.join(
                    df.select(["symbol", "timestamp", c]).rename({c: alias}),
                    on=["symbol", "timestamp"],
                    how="left",
                )
                acc.append(alias)

    score = pl.col(f"pred_{horizon}")
    if mfe_cols:
        mfe = sum(_zscore(pl.col(c)) for c in mfe_cols) / len(mfe_cols)
        base = base.with_columns(mfe.alias("ensemble_mfe_z"))
        score = score + mfe_weight * pl.col("ensemble_mfe_z")
    if mae_cols:
        # MAE predictions are negative; more negative = more risk.
        adverse = sum(_zscore(-pl.col(c)) for c in mae_cols) / len(mae_cols)
        base = base.with_columns(adverse.alias("ensemble_adverse_z"))
        score = score - risk_lambda * pl.col("ensemble_adverse_z")

    base = base.with_columns(score.alias("opportunity_score"))
    base.write_parquet(output)
    meta = {
        "inputs": inputs,
        "weights": weights,
        "horizon": horizon,
        "risk_lambda": risk_lambda,
        "mfe_weight": mfe_weight,
    }
    Path(output).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta
