from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import polars as pl

from .metrics import nonoverlap_sample, performance_metrics


def simulate_portfolio(
    predictions: str,
    output: str,
    horizon: int,
    score_col: str | None = None,
    top_frac: float = 0.02,
    gross_exposure: float = 1.0,
    max_position: float = 0.05,
    cost_bps: float = 2.0,
    max_turnover: float = 1.0,
    stride: int | None = None,
    periods_per_year: float | None = None,
    min_price: float | None = None,
    min_dollar_volume: float | None = None,
):
    """
    Position-level research simulator.

    Default behavior is deliberately conservative:
    - signal at bar t
    - entry at t+1 open through ret_exec_fwd_H when available
    - rebalance every H bars, so H-bar holding periods do not overlap
    - transaction costs charged on actual portfolio turnover
    """
    df = pl.read_parquet(predictions).sort(["timestamp", "symbol"])
    if score_col is None:
        score_col = (
            "opportunity_score"
            if "opportunity_score" in df.columns
            else f"pred_{horizon}"
        )

    ret_col = f"ret_exec_fwd_{horizon}"
    if ret_col not in df.columns:
        ret_col = f"ret_fwd_{horizon}"
    if score_col not in df.columns:
        raise ValueError(f"{score_col} not found")
    if ret_col not in df.columns:
        raise ValueError(f"{ret_col} not found")

    step = horizon if stride is None else max(1, int(stride))
    df = nonoverlap_sample(df, step)

    if min_price is not None and "close" in df.columns:
        df = df.filter(pl.col("close") >= min_price)
    if min_dollar_volume is not None and {"close", "volume"}.issubset(df.columns):
        df = df.filter((pl.col("close") * pl.col("volume")) >= min_dollar_volume)

    prev = {}
    rows = []
    half_gross = gross_exposure / 2.0

    for _, g in df.partition_by("timestamp", as_dict=True, maintain_order=True).items():
        g = g.drop_nulls([score_col, ret_col]).sort(score_col)
        n = len(g)
        if n < 2:
            continue

        k = max(1, int(np.ceil(n * top_frac)))
        short_syms = g.head(k)["symbol"].to_list()
        long_syms = g.tail(k)["symbol"].to_list()

        lw = min(max_position, half_gross / max(len(long_syms), 1))
        sw = min(max_position, half_gross / max(len(short_syms), 1))
        target = {s: lw for s in long_syms}
        target.update({s: -sw for s in short_syms})

        all_syms = set(prev) | set(target)
        raw_turnover = sum(
            abs(target.get(s, 0.0) - prev.get(s, 0.0)) for s in all_syms
        )
        turnover = raw_turnover
        if turnover > max_turnover and turnover > 0:
            scale = max_turnover / turnover
            target = {
                s: prev.get(s, 0.0)
                + scale * (target.get(s, 0.0) - prev.get(s, 0.0))
                for s in all_syms
            }
            turnover = max_turnover

        ret_map = dict(zip(g["symbol"].to_list(), g[ret_col].to_list()))
        long_ret = sum(
            w * float(ret_map.get(s, 0.0))
            for s, w in target.items() if w > 0
        )
        short_ret = sum(
            w * float(ret_map.get(s, 0.0))
            for s, w in target.items() if w < 0
        )
        gross_ret = long_ret + short_ret
        cost = turnover * cost_bps / 10000.0
        net_ret = gross_ret - cost

        long_exp = sum(w for w in target.values() if w > 0)
        short_exp = -sum(w for w in target.values() if w < 0)
        rows.append((
            g["timestamp"][0],
            gross_ret,
            net_ret,
            long_ret,
            short_ret,
            raw_turnover,
            turnover,
            cost,
            long_exp,
            short_exp,
            len([w for w in target.values() if w != 0]),
        ))
        prev = target

    pnl = pl.DataFrame(
        rows,
        schema=[
            "timestamp", "gross_return", "net_return", "long_contribution",
            "short_contribution", "raw_turnover", "turnover", "cost",
            "long_exposure", "short_exposure", "positions",
        ],
        orient="row",
    )

    if pnl.height:
        pnl = pnl.with_columns([
            (pl.col("net_return") + 1.0).cum_prod().alias("equity"),
            (pl.col("gross_return") + 1.0).cum_prod().alias("gross_equity"),
        ])
        pnl = pnl.with_columns(
            (pl.col("equity") / pl.col("equity").cum_max() - 1.0).alias("drawdown")
        )

    pnl.write_parquet(output)

    perf = performance_metrics(
        pnl["net_return"].to_numpy() if pnl.height else np.array([]),
        periods_per_year=periods_per_year,
    )
    gross_perf = performance_metrics(
        pnl["gross_return"].to_numpy() if pnl.height else np.array([]),
        periods_per_year=periods_per_year,
    )

    summary = {
        **perf,
        "gross_performance": gross_perf,
        "return_column": ret_col,
        "score_column": score_col,
        "rebalance_stride_bars": step,
        "overlap_avoided": step >= horizon,
        "mean_turnover": float(pnl["turnover"].mean()) if pnl.height else float("nan"),
        "mean_raw_turnover": float(pnl["raw_turnover"].mean()) if pnl.height else float("nan"),
        "mean_cost": float(pnl["cost"].mean()) if pnl.height else float("nan"),
        "mean_long_contribution": (
            float(pnl["long_contribution"].mean()) if pnl.height else float("nan")
        ),
        "mean_short_contribution": (
            float(pnl["short_contribution"].mean()) if pnl.height else float("nan")
        ),
        "max_turnover": max_turnover,
        "gross_exposure_target": gross_exposure,
        "max_position": max_position,
        "top_fraction_each_side": top_frac,
        "cost_bps": cost_bps,
    }
    Path(output).with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return summary
