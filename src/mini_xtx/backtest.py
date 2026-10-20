from __future__ import annotations
import numpy as np
import polars as pl


def backtest_ranked(
    predictions: str,
    horizon: str,
    top_frac: float = 0.02,
    cost_bps: float = 2.0,
    stride: int | None = None,
    executable: bool = True,
):
    df = pl.read_parquet(predictions)
    h = int(horizon.split("_")[-1])
    pcol = f"pred_{h}"
    ret_col = (
        f"ret_exec_fwd_{h}"
        if executable and f"ret_exec_fwd_{h}" in df.columns
        else horizon
    )

    if pcol not in df.columns:
        raise ValueError(f"{pcol} not found")
    if ret_col not in df.columns:
        raise ValueError(f"{ret_col} not found")

    # By default sample every h bars, so h-bar holding periods do not overlap.
    step = h if stride is None else max(1, stride)
    timestamps = (
        df.select("timestamp")
        .unique()
        .sort("timestamp")
        .with_row_index("__t")
    )
    keep = timestamps.filter(pl.col("__t") % step == 0).select("timestamp")
    df = df.join(keep, on="timestamp", how="inner")

    x = (
        df.drop_nulls([pcol, ret_col])
        .with_columns(
            pl.col(pcol)
            .rank("average")
            .over("timestamp")
            .truediv(pl.len().over("timestamp"))
            .alias("rank")
        )
        .with_columns(
            pl.when(pl.col("rank") >= 1 - top_frac)
            .then(1.0)
            .when(pl.col("rank") <= top_frac)
            .then(-1.0)
            .otherwise(0.0)
            .alias("side")
        )
    )

    selected = x.filter(pl.col("side") != 0).with_columns([
        (pl.col("side") * pl.col(ret_col)).alias("gross_ret"),
        (pl.col("side") * pl.col(ret_col) - cost_bps / 10000.0).alias("net_ret"),
    ])

    pnl = (
        selected.group_by("timestamp")
        .agg([
            pl.col("net_ret").mean().alias("portfolio_ret"),
            pl.col("gross_ret").mean().alias("gross_portfolio_ret"),
            pl.when(pl.col("side") > 0)
            .then(pl.col(ret_col))
            .otherwise(None)
            .mean()
            .alias("long_return"),
            pl.when(pl.col("side") < 0)
            .then(-pl.col(ret_col))
            .otherwise(None)
            .mean()
            .alias("short_return"),
            (pl.col("side") > 0).sum().alias("n_long"),
            (pl.col("side") < 0).sum().alias("n_short"),
            pl.corr(pcol, ret_col).alias("cs_ic"),
        ])
        .sort("timestamp")
    )

    arr = pnl["portfolio_ret"].to_numpy()
    mean = float(np.nanmean(arr)) if len(arr) else np.nan
    std = float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else np.nan

    summary = {
        "timestamps": int(len(pnl)),
        "return_column": ret_col,
        "rebalance_stride_bars": step,
        "overlap_avoided": step >= h,
        "mean_return_per_rebalance": mean,
        "std_return_per_rebalance": std,
        "sharpe_per_sqrt_rebalance": (
            float(mean / std) if std and std > 0 else np.nan
        ),
        "mean_long_return": (
            float(pnl["long_return"].mean()) if len(pnl) else np.nan
        ),
        "mean_short_return": (
            float(pnl["short_return"].mean()) if len(pnl) else np.nan
        ),
        "mean_cs_ic": float(pnl["cs_ic"].mean()) if len(pnl) else np.nan,
        "top_fraction_each_side": top_frac,
        "cost_bps_roundtrip_per_position": cost_bps,
    }
    return summary, pnl
