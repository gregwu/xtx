from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import polars as pl


def simulate_portfolio(
    predictions: str,
    output: str,
    horizon: int,
    score_col: str = "opportunity_score",
    top_frac: float = 0.02,
    gross_exposure: float = 1.0,
    max_position: float = 0.05,
    cost_bps: float = 2.0,
    max_turnover: float = 1.0,
    min_price: float | None = None,
    min_dollar_volume: float | None = None,
):
    """
    Position-level research simulator.

    Targets equal-dollar long/short tail portfolios. Costs are charged on actual
    absolute weight changes, so unchanged positions do not repeatedly pay a full
    round trip. Execution return defaults to next-open-to-horizon-close.
    """
    df = pl.read_parquet(predictions).sort(["timestamp", "symbol"])
    ret_col = f"ret_exec_fwd_{horizon}"
    if ret_col not in df.columns:
        ret_col = f"ret_fwd_{horizon}"
    if score_col not in df.columns:
        raise ValueError(f"{score_col} not found")
    if ret_col not in df.columns:
        raise ValueError(f"{ret_col} not found")

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
        turnover = sum(abs(target.get(s, 0.0) - prev.get(s, 0.0)) for s in all_syms)
        if turnover > max_turnover and turnover > 0:
            scale = max_turnover / turnover
            target = {
                s: prev.get(s, 0.0) + scale * (target.get(s, 0.0) - prev.get(s, 0.0))
                for s in all_syms
            }
            turnover = max_turnover

        ret_map = dict(zip(g["symbol"].to_list(), g[ret_col].to_list()))
        gross_ret = sum(w * float(ret_map.get(s, 0.0)) for s, w in target.items())
        cost = turnover * cost_bps / 10000.0
        net_ret = gross_ret - cost
        long_exp = sum(w for w in target.values() if w > 0)
        short_exp = -sum(w for w in target.values() if w < 0)
        rows.append((
            g["timestamp"][0], gross_ret, net_ret, turnover, cost,
            long_exp, short_exp, len([w for w in target.values() if w != 0])
        ))
        prev = target

    pnl = pl.DataFrame(
        rows,
        schema=[
            "timestamp", "gross_return", "net_return", "turnover", "cost",
            "long_exposure", "short_exposure", "positions"
        ],
        orient="row",
    )
    pnl.write_parquet(output)

    arr = pnl["net_return"].to_numpy() if pnl.height else np.array([])
    summary = {
        "periods": int(pnl.height),
        "mean_net_return": float(np.mean(arr)) if len(arr) else float("nan"),
        "std_net_return": float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan"),
        "sharpe_per_sqrt_period": (
            float(np.mean(arr) / np.std(arr, ddof=1))
            if len(arr) > 1 and np.std(arr, ddof=1) > 0 else float("nan")
        ),
        "mean_turnover": float(pnl["turnover"].mean()) if pnl.height else float("nan"),
        "max_turnover": max_turnover,
        "gross_exposure_target": gross_exposure,
        "max_position": max_position,
        "cost_bps": cost_bps,
    }
    Path(output).with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return summary
