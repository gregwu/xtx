from __future__ import annotations

import math
import numpy as np
import pandas as pd
import polars as pl


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    aa, bb = a[ok], b[ok]
    if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    return _corr(
        pd.Series(a).rank(method="average").to_numpy(),
        pd.Series(b).rank(method="average").to_numpy(),
    )


def nonoverlap_sample(
    df: pl.DataFrame,
    stride: int,
    timestamp_col: str = "timestamp",
) -> pl.DataFrame:
    """Keep every Nth decision timestamp. Used as the default economic evaluation."""
    stride = max(1, int(stride))
    times = (
        df.select(timestamp_col)
        .unique()
        .sort(timestamp_col)
        .with_row_index("__idx")
        .filter(pl.col("__idx") % stride == 0)
        .select(timestamp_col)
    )
    return df.join(times, on=timestamp_col, how="inner")


def cross_sectional_metrics(
    df: pl.DataFrame,
    pred_col: str,
    ret_col: str,
    timestamp_col: str = "timestamp",
    deciles: int = 10,
) -> tuple[dict, pl.DataFrame, pl.DataFrame]:
    """IC, RankIC, ICIR, hit-rate, and cross-sectional decile returns."""
    rows = []
    decile_rows = []
    for _, g in (
        df.select([timestamp_col, pred_col, ret_col])
        .drop_nulls()
        .partition_by(timestamp_col, as_dict=True)
        .items()
    ):
        a = g[pred_col].to_numpy()
        b = g[ret_col].to_numpy()
        ic = _corr(a, b)
        ric = _rank_corr(a, b)
        sign_hit = float(np.mean(np.sign(a) == np.sign(b))) if len(a) else float("nan")
        rows.append((g[timestamp_col][0], ic, ric, sign_hit, len(g)))

        if len(g) >= deciles:
            pdf = g.to_pandas()
            r = pdf[pred_col].rank(method="first")
            pdf["decile"] = pd.qcut(r, deciles, labels=False, duplicates="drop") + 1
            for d, z in pdf.groupby("decile"):
                decile_rows.append(
                    (g[timestamp_col][0], int(d), float(z[ret_col].mean()), len(z))
                )

    by_time = (
        pl.DataFrame(
            rows,
            schema=[timestamp_col, "ic", "rank_ic", "sign_hit_rate", "n"],
            orient="row",
        )
        if rows
        else pl.DataFrame(
            {timestamp_col: [], "ic": [], "rank_ic": [], "sign_hit_rate": [], "n": []}
        )
    )
    dec = (
        pl.DataFrame(
            decile_rows,
            schema=[timestamp_col, "decile", "mean_return", "n"],
            orient="row",
        )
        if decile_rows
        else pl.DataFrame(
            {timestamp_col: [], "decile": [], "mean_return": [], "n": []}
        )
    )

    ic = by_time["ic"].drop_nulls().to_numpy() if by_time.height else np.array([])
    ric = by_time["rank_ic"].drop_nulls().to_numpy() if by_time.height else np.array([])
    ric_std = float(np.nanstd(ric, ddof=1)) if len(ric) > 1 else float("nan")
    ic_std = float(np.nanstd(ic, ddof=1)) if len(ic) > 1 else float("nan")

    summary = {
        "timestamps": int(by_time.height),
        "mean_ic": float(np.nanmean(ic)) if len(ic) else float("nan"),
        "ic_std": ic_std,
        "mean_rank_ic": float(np.nanmean(ric)) if len(ric) else float("nan"),
        "rank_ic_std": ric_std,
        "icir_per_sqrt_period": (
            float(np.nanmean(ric) / ric_std)
            if np.isfinite(ric_std) and ric_std > 0
            else float("nan")
        ),
        "mean_sign_hit_rate": (
            float(by_time["sign_hit_rate"].mean()) if by_time.height else float("nan")
        ),
        "positive_ic_fraction": (
            float(np.mean(ic > 0)) if len(ic) else float("nan")
        ),
        "positive_rank_ic_fraction": (
            float(np.mean(ric > 0)) if len(ric) else float("nan")
        ),
    }
    return summary, by_time, dec


def decile_summary(deciles: pl.DataFrame) -> pl.DataFrame:
    if deciles.is_empty():
        return pl.DataFrame({"decile": [], "mean_return": [], "median_return": [], "observations": []})
    return (
        deciles.group_by("decile")
        .agg([
            pl.col("mean_return").mean().alias("mean_return"),
            pl.col("mean_return").median().alias("median_return"),
            pl.len().alias("observations"),
        ])
        .sort("decile")
    )


def tail_metrics(
    df: pl.DataFrame,
    pred_col: str,
    ret_col: str,
    fractions: tuple[float, ...] = (0.01, 0.02, 0.05),
) -> dict:
    out = {}
    for frac in fractions:
        vals_long, vals_short, vals_ls = [], [], []
        for _, g in (
            df.select(["timestamp", pred_col, ret_col])
            .drop_nulls()
            .partition_by("timestamp", as_dict=True)
            .items()
        ):
            n = len(g)
            k = max(1, int(math.ceil(n * frac)))
            z = g.sort(pred_col)
            s = float((-z.head(k)[ret_col]).mean())
            l = float(z.tail(k)[ret_col].mean())
            vals_long.append(l)
            vals_short.append(s)
            vals_ls.append((l + s) / 2.0)

        tag = f"{int(frac * 10000):04d}bp_universe"
        out[tag] = {
            "fraction": frac,
            "mean_long": float(np.mean(vals_long)) if vals_long else float("nan"),
            "mean_short": float(np.mean(vals_short)) if vals_short else float("nan"),
            "mean_long_short": float(np.mean(vals_ls)) if vals_ls else float("nan"),
            "long_win_rate": float(np.mean(np.asarray(vals_long) > 0)) if vals_long else float("nan"),
            "short_win_rate": float(np.mean(np.asarray(vals_short) > 0)) if vals_short else float("nan"),
            "long_short_win_rate": float(np.mean(np.asarray(vals_ls) > 0)) if vals_ls else float("nan"),
        }
    return out


def performance_metrics(
    returns: np.ndarray,
    periods_per_year: float | None = None,
) -> dict:
    """Economic metrics from a non-overlapping portfolio return stream."""
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            "periods": 0,
            "total_return": float("nan"),
            "mean_return": float("nan"),
            "volatility": float("nan"),
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "max_drawdown": float("nan"),
            "ulcer_index": float("nan"),
            "win_rate": float("nan"),
        }

    equity = np.cumprod(1.0 + x)
    peaks = np.maximum.accumulate(equity)
    dd = equity / peaks - 1.0
    downside = x[x < 0]
    std = float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")
    down_std = float(np.sqrt(np.mean(downside * downside))) if len(downside) else float("nan")
    scale = math.sqrt(periods_per_year) if periods_per_year and periods_per_year > 0 else 1.0

    return {
        "periods": int(len(x)),
        "total_return": float(equity[-1] - 1.0),
        "mean_return": float(np.mean(x)),
        "median_return": float(np.median(x)),
        "volatility": std,
        "sharpe": float(np.mean(x) / std * scale) if np.isfinite(std) and std > 0 else float("nan"),
        "sortino": (
            float(np.mean(x) / down_std * scale)
            if np.isfinite(down_std) and down_std > 0
            else float("nan")
        ),
        "max_drawdown": float(np.min(dd)),
        "ulcer_index": float(np.sqrt(np.mean(np.square(dd)))),
        "win_rate": float(np.mean(x > 0)),
        "profit_factor": (
            float(x[x > 0].sum() / abs(x[x < 0].sum()))
            if np.any(x < 0) and np.any(x > 0)
            else float("nan")
        ),
    }
