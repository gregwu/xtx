from __future__ import annotations
import numpy as np
import polars as pl

def backtest_ranked(predictions: str, horizon: str, top_frac: float=0.02,
                    cost_bps: float=2.0):
    df = pl.read_parquet(predictions)
    h = horizon.split("_")[-1]
    pcol = f"pred_{h}"
    if pcol not in df.columns:
        raise ValueError(f"{pcol} not found")
    if horizon not in df.columns:
        raise ValueError(f"{horizon} not found")

    # Rank by timestamp. Long top fraction, short bottom fraction.
    x = df.drop_nulls([pcol,horizon]).with_columns([
        pl.col(pcol).rank("average").over("timestamp")
            .truediv(pl.len().over("timestamp")).alias("rank")
    ])
    x = x.with_columns(
        pl.when(pl.col("rank") >= 1-top_frac).then(1.0)
          .when(pl.col("rank") <= top_frac).then(-1.0)
          .otherwise(0.0).alias("side")
    )
    # Simplified turnover assumption: each signal incurs round-trip cost once.
    x = x.with_columns(
        (pl.col("side")*pl.col(horizon) - pl.col("side").abs()*cost_bps/10000.0)
        .alias("net_ret")
    )
    pnl = x.filter(pl.col("side") != 0).group_by("timestamp").agg([
        pl.col("net_ret").mean().alias("portfolio_ret"),
        pl.len().alias("n_positions"),
        pl.corr(pcol,horizon).alias("cs_ic"),
    ]).sort("timestamp")
    arr = pnl["portfolio_ret"].to_numpy()
    mean = float(np.nanmean(arr)) if len(arr) else np.nan
    std = float(np.nanstd(arr, ddof=1)) if len(arr)>1 else np.nan
    summary = {
        "timestamps": int(len(pnl)),
        "mean_return_per_rebalance": mean,
        "std_return_per_rebalance": std,
        "sharpe_per_sqrt_rebalance": float(mean/std) if std and std>0 else np.nan,
        "mean_cs_ic": float(pnl["cs_ic"].mean()) if len(pnl) else np.nan,
        "top_fraction_each_side": top_frac,
        "cost_bps": cost_bps,
    }
    return summary, pnl
