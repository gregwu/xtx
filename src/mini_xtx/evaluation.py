from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import polars as pl

from .metrics import (
    cross_sectional_metrics,
    decile_summary,
    nonoverlap_sample,
    performance_metrics,
    tail_metrics,
)
from .portfolio import simulate_portfolio


def evaluate_predictions(
    predictions: str,
    outdir: str,
    horizon: int,
    pred_col: str | None = None,
    top_frac: float = 0.02,
    cost_bps: float = 2.0,
    stride: int | None = None,
    periods_per_year: float | None = None,
    gross_exposure: float = 1.0,
    max_position: float = 0.05,
    max_turnover: float = 1.0,
):
    """
    One-command evaluation for an OOS prediction file.

    Produces:
    - full-sample IC / RankIC diagnostics
    - non-overlapping IC / RankIC / ICIR
    - decile monotonicity table
    - 1%, 2%, 5% tail returns and win rates
    - turnover-aware long/short portfolio simulation
    - Sharpe, Sortino, max drawdown, Ulcer index, profit factor
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(predictions).sort(["timestamp", "symbol"])

    if pred_col is None:
        pred_col = (
            "opportunity_score"
            if "opportunity_score" in df.columns
            else f"pred_{horizon}"
        )
    if pred_col not in df.columns:
        raise ValueError(f"{pred_col} not found")

    ret_col = f"ret_exec_fwd_{horizon}"
    if ret_col not in df.columns:
        ret_col = f"ret_fwd_{horizon}"
    if ret_col not in df.columns:
        raise ValueError(f"No executable or close-to-close return found for h={horizon}")

    step = horizon if stride is None else max(1, int(stride))
    required = ["symbol", "timestamp", pred_col, ret_col]
    optional = [
        f"mfe_fwd_{horizon}", f"mae_fwd_{horizon}",
        f"pred_mfe_{horizon}", f"pred_mae_{horizon}",
    ]
    cols = required + [c for c in optional if c in df.columns]
    eval_df = df.select(cols).drop_nulls([pred_col, ret_col])

    full_summary, full_ic, full_deciles = cross_sectional_metrics(
        eval_df, pred_col, ret_col
    )

    nonoverlap = nonoverlap_sample(eval_df, step)
    no_summary, no_ic, no_deciles = cross_sectional_metrics(
        nonoverlap, pred_col, ret_col
    )
    no_summary["tails"] = tail_metrics(nonoverlap, pred_col, ret_col)
    no_summary["rebalance_stride_bars"] = step
    no_summary["overlap_avoided"] = step >= horizon

    dec_summary = decile_summary(no_deciles)
    if dec_summary.height >= 2:
        vals = dec_summary["mean_return"].to_numpy()
        no_summary["decile_top_minus_bottom"] = float(vals[-1] - vals[0])
        # Spearman-like monotonicity of decile number vs mean return.
        x = np.arange(1, len(vals) + 1, dtype=float)
        no_summary["decile_monotonicity"] = (
            float(np.corrcoef(x, vals)[0, 1])
            if np.std(vals) > 1e-12 else float("nan")
        )

    # Additional path-quality diagnostics when MFE/MAE exist.
    mfe_col = f"mfe_fwd_{horizon}"
    mae_col = f"mae_fwd_{horizon}"
    path_quality = {}
    if mfe_col in nonoverlap.columns:
        z = nonoverlap.drop_nulls([mfe_col])
        path_quality["mean_mfe"] = float(z[mfe_col].mean()) if z.height else float("nan")
    if mae_col in nonoverlap.columns:
        z = nonoverlap.drop_nulls([mae_col])
        path_quality["mean_mae"] = float(z[mae_col].mean()) if z.height else float("nan")
    if mfe_col in nonoverlap.columns and mae_col in nonoverlap.columns:
        ranked = (
            nonoverlap.with_columns(
                pl.col(pred_col)
                .rank("average")
                .over("timestamp")
                .truediv(pl.len().over("timestamp"))
                .alias("__rank")
            )
            .filter(pl.col("__rank") >= 1.0 - top_frac)
        )
        if ranked.height:
            path_quality["top_tail_mean_mfe"] = float(ranked[mfe_col].mean())
            path_quality["top_tail_mean_mae"] = float(ranked[mae_col].mean())
            denom = abs(float(ranked[mae_col].mean())) + 1e-12
            path_quality["top_tail_mfe_to_mae"] = float(ranked[mfe_col].mean()) / denom

    pnl_path = out / "portfolio.parquet"
    portfolio_summary = simulate_portfolio(
        predictions=predictions,
        output=str(pnl_path),
        horizon=horizon,
        score_col=pred_col,
        top_frac=top_frac,
        gross_exposure=gross_exposure,
        max_position=max_position,
        cost_bps=cost_bps,
        max_turnover=max_turnover,
        stride=step,
        periods_per_year=periods_per_year,
    )

    full_ic.write_parquet(out / "ic_by_time_full.parquet")
    full_deciles.write_parquet(out / "deciles_full.parquet")
    no_ic.write_parquet(out / "ic_by_time_nonoverlap.parquet")
    no_deciles.write_parquet(out / "deciles_nonoverlap.parquet")
    dec_summary.write_parquet(out / "decile_summary.parquet")

    report = {
        "prediction_file": predictions,
        "prediction_column": pred_col,
        "return_column": ret_col,
        "horizon": horizon,
        "full_sample_signal_metrics": full_summary,
        "nonoverlap_signal_metrics": no_summary,
        "path_quality": path_quality,
        "portfolio": portfolio_summary,
    }
    (out / "evaluation.json").write_text(json.dumps(report, indent=2))
    return report
