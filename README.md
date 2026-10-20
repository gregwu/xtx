# Mini-XTX: XTX-inspired multi-asset forecasting system

This project is an **XTX-inspired research architecture**, not a reproduction of XTX Markets'
proprietary system. It is designed for existing US equity **5-minute + daily OHLCV data**.

Core ideas:
1. Pool many symbols into one model instead of training one model per ticker.
2. Predict multiple future horizons simultaneously.
3. Predict volatility-normalized, market-residual returns instead of only UP/DOWN classes.
4. Use cross-sectional ranks and market context.
5. Enforce strictly causal feature generation and walk-forward validation.
6. Start with LightGBM; add a shared Transformer sequence model.
7. Evaluate economic value after slippage/costs, not accuracy alone.

## Expected data

Parquet files can be one file or a directory of files.

Required columns:
- symbol
- timestamp
- open
- high
- low
- close
- volume

Optional:
- sector
- vwap
- trades

Timestamps must be parseable by Polars. 5-minute data should include only rows that you want
the model to see. If you have extended-hours data, keep it and use `session` features; otherwise
RTH-only data works.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Build 5-minute dataset

```bash
python -m mini_xtx.cli build \
  --input /data/us_5m \
  --output /data/features_5m.parquet \
  --freq 5m
```

## 2. Build daily dataset

```bash
python -m mini_xtx.cli build \
  --input /data/us_daily \
  --output /data/features_daily.parquet \
  --freq 1d
```

## 3. Train LightGBM multi-horizon models

```bash
python -m mini_xtx.cli train-lgbm \
  --data /data/features_5m.parquet \
  --outdir artifacts/lgbm_5m \
  --train-end 2025-12-31 \
  --valid-start 2026-01-01 \
  --valid-end 2026-06-30 \
  --test-start 2026-07-01
```

## 4. Train Transformer

```bash
python -m mini_xtx.cli train-transformer \
  --data /data/features_5m.parquet \
  --outdir artifacts/transformer_5m \
  --train-end 2025-12-31 \
  --valid-start 2026-01-01 \
  --valid-end 2026-06-30 \
  --test-start 2026-07-01 \
  --seq-len 78
```

78 five-minute bars ~= one regular US session.

## 5. Backtest ranked signals

```bash
python -m mini_xtx.cli backtest \
  --predictions artifacts/lgbm_5m/predictions.parquet \
  --horizon ret_fwd_6 \
  --top-frac 0.02 \
  --cost-bps 2.0
```

## Example run: daily bars (2026-10-20)

First end-to-end run of the pipeline on US equity daily bars (Alpaca), to check it works.

Data: 532 symbols, 1.59M rows, 2014-06-02 to 2026-10-16. No `sector` column, so the sector features are zero.
Split: train through 2023-12-31, validate on 2024, test from 2025-01-01.

```bash
python -m mini_xtx.cli build --input daily.parquet --output features_daily.parquet --freq 1d
python -m mini_xtx.cli train-lgbm --data features_daily.parquet --outdir artifacts/lgbm_daily \
  --train-end 2023-12-31 --valid-start 2024-01-01 --valid-end 2024-12-31 --test-start 2025-01-01
python -m mini_xtx.cli backtest --predictions artifacts/lgbm_daily/predictions.parquet \
  --horizon ret_fwd_10 --top-frac 0.05 --cost-bps 2.0
```

Building the features took about 3 s. Training LightGBM took about 50 s across the four horizons.

### LightGBM: correlation between predictions and the vol-normalized residual target

| Target | Best iteration | Validation IC | Test IC |
|---|---|---|---|
| `target_1` | 1 | 0.013 | 0.005 |
| `target_3` | 1 | 0.007 | 0.014 |
| `target_5` | 81 | 0.022 | 0.032 |
| `target_10` | 355 | 0.031 | 0.054 |

The models find no usable signal at 1 and 3 days: early stopping keeps only one tree.
The signal gets stronger with the horizon, and the test IC is higher than the validation IC at 5 and 10 days.

### Backtest (long top 5%, short bottom 5%, 2 bps cost)

| Horizon | Rebalances | Mean return per rebalance | Std | Sharpe per rebalance | Mean cross-sectional IC |
|---|---|---|---|---|---|
| `ret_fwd_5` | 444 | 0.17% | 1.76% | 0.098 | 0.031 |
| `ret_fwd_10` | 439 | 0.32% | 2.15% | 0.150 | 0.036 |

Caveat: the backtest rebalances on every bar but holds each position for 5 or 10 days, so the holding periods overlap.
These figures overstate an achievable return and cannot be annualized directly.

### Transformer

This was only a smoke run: 2 epochs, `--seq-len 20`, on a CUDA GPU (about 2 min).
Validation loss went from 1.240 to 1.254, so the result says nothing about performance.
`train-transformer` saves `best.pt` but does not yet write test-period predictions, so it cannot be backtested.

### Fix found during the run

`train-lgbm` crashed when it joined the test predictions back onto `symbol` and `timestamp`.
The model scored test rows after dropping nulls in the features and the target, but the keys only dropped null targets.
Both now drop nulls over the same columns.

The 5-minute pipeline has not been run yet.

## Recommended production path

Research:
`raw bars -> causal features -> pooled models -> walk-forward -> calibration -> ensemble`

Live:
`5m bar close -> same features -> model inference -> ranked opportunities -> execution/risk`

For a Rust execution stack, export the LightGBM models as text and implement inference using a
Rust LightGBM-compatible parser/library. The Transformer can later be distilled into a smaller
tree model using teacher predictions as targets.

## Important limitation

With OHLCV bars only, this is **not a true market-making system**. XTX's real edge operates on
much richer data, including microstructure/order-book and cross-venue information. This project
copies the *research philosophy*: massive pooled data, multi-asset learning, multi-horizon
forecasting, low-SNR discipline, strict validation, and economic evaluation.
