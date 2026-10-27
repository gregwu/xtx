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
    return _corr(pd.Series(a).rank(method="average").to_numpy(),
                 pd.Series(b).rank(method="average").to_numpy())


def cross_sectional_metrics(
    df: pl.DataFrame,
    pred_col: str,
    ret_col: str,
    timestamp_col: str = "timestamp",
    deciles: int = 10,
) -> tuple[dict, pl.DataFrame, pl.DataFrame]:
    """IC, RankIC, ICIR, and cross-sectional decile returns."""
    rows = []
    decile_rows = []
    for ts, g in df.select([timestamp_col, pred_col, ret_col]).drop_nulls().partition_by(
        timestamp_col, as_dict=True
    ).items():
        a = g[pred_col].to_numpy()
        b = g[ret_col].to_numpy()
        ic = _corr(a, b)
        ric = _rank_corr(a, b)
        rows.append((g[timestamp_col][0], ic, ric, len(g)))

        if len(g) >= deciles:
            pdf = g.to_pandas()
            # rank(method=first) avoids qcut failures from identical predictions.
            r = pdf[pred_col].rank(method="first")
            pdf["decile"] = pd.qcut(r, deciles, labels=False, duplicates="drop") + 1
            for d, z in pdf.groupby("decile"):
                decile_rows.append((g[timestamp_col][0], int(d), float(z[ret_col].mean()), len(z)))

    by_time = pl.DataFrame(
        rows, schema=[timestamp_col, "ic", "rank_ic", "n"], orient="row"
    ) if rows else pl.DataFrame({timestamp_col: [], "ic": [], "rank_ic": [], "n": []})

    dec = pl.DataFrame(
        decile_rows,
        schema=[timestamp_col, "decile", "mean_return", "n"],
        orient="row",
    ) if decile_rows else pl.DataFrame(
        {timestamp_col: [], "decile": [], "mean_return": [], "n": []}
    )

    ic = by_time["ic"].drop_nulls().to_numpy() if by_time.height else np.array([])
    ric = by_time["rank_ic"].drop_nulls().to_numpy() if by_time.height else np.array([])
    ric_std = float(np.nanstd(ric, ddof=1)) if len(ric) > 1 else float("nan")
    summary = {
        "timestamps": int(by_time.height),
        "mean_ic": float(np.nanmean(ic)) if len(ic) else float("nan"),
        "mean_rank_ic": float(np.nanmean(ric)) if len(ric) else float("nan"),
        "rank_ic_std": ric_std,
        "icir_per_sqrt_period": (
            float(np.nanmean(ric) / ric_std) if np.isfinite(ric_std) and ric_std > 0 else float("nan")
        ),
    }
    return summary, by_time, dec


def tail_metrics(
    df: pl.DataFrame,
    pred_col: str,
    ret_col: str,
    fractions: tuple[float, ...] = (0.01, 0.02, 0.05),
) -> dict:
    out = {}
    for frac in fractions:
        vals_long, vals_short, vals_ls = [], [], []
        for _, g in df.select(["timestamp", pred_col, ret_col]).drop_nulls().partition_by(
            "timestamp", as_dict=True
        ).items():
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
        }
    return out
