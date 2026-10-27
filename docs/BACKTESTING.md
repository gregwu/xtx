# Backtesting and evaluation

This branch makes non-overlapping, executable-entry evaluation the default.

## 1. Walk-forward OOS predictions

Use anchored monthly retraining. Each test month is scored only by models trained on dates before that month.

```bash
python -m mini_xtx.cli walkforward \
  --data /data/features_5m_v02.parquet \
  --outdir artifacts/wf_h6 \
  --horizon 6 \
  --start 2025-01-01 \
  --end 2026-10-31 \
  --models ridge,lgbm
```

Default safeguards:
- purge = forecast horizon
- evaluation stride = forecast horizon
- executable return = next-open to t+H close when available

For a 30-minute model on 5m bars, `horizon=6` means economic metrics use every sixth decision timestamp by default.

## 2. Evaluate any OOS prediction file

```bash
python -m mini_xtx.cli evaluate \
  --predictions artifacts/wf_h6/lgbm_predictions.parquet \
  --outdir artifacts/eval_h6 \
  --horizon 6 \
  --top-frac 0.02 \
  --cost-bps 2
```

Outputs:
- `evaluation.json`
- `portfolio.parquet`
- `ic_by_time_full.parquet`
- `ic_by_time_nonoverlap.parquet`
- `deciles_full.parquet`
- `deciles_nonoverlap.parquet`
- `decile_summary.parquet`

## 3. Signal diagnostics

The evaluator reports:
- Pearson IC
- Spearman-style RankIC
- RankIC standard deviation
- ICIR
- positive IC fraction
- sign hit rate
- decile monotonicity
- top-minus-bottom decile spread
- top/bottom 1%, 2%, and 5% tail returns
- tail win rates
- MFE / MAE path quality when available

## 4. Portfolio diagnostics

The position simulator reports:
- gross and net return
- actual turnover
- trading cost
- long contribution
- short contribution
- Sharpe
- Sortino
- max drawdown
- Ulcer index
- win rate
- profit factor
- equity and drawdown series

Costs are applied to absolute weight changes rather than charging a full round trip on every signal.

## 5. Intraday target safety

5m targets are invalidated when t+H falls on another calendar session. This prevents a late-session "30m" or "60m" label from unintentionally including overnight return.

## 6. Annualization

The evaluator does not guess annualization. Pass `--periods-per-year` only when the sampling schedule is well defined.

Examples:
- daily non-overlapping 5-day strategy: roughly 252 / 5
- 5m RTH, 30m non-overlap: roughly 252 * 78 / 6

Raw non-annualized Sharpe-per-period is safer when sessions include pre/post market or irregular timestamps.
