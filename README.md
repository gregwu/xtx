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
