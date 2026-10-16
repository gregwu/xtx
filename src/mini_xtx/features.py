from __future__ import annotations
import polars as pl

BASE_COLS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]

def _read_any(path: str) -> pl.DataFrame:
    p = path.lower()
    if p.endswith(".parquet"):
        return pl.read_parquet(path)
    if p.endswith(".csv"):
        return pl.read_csv(path, try_parse_dates=True)
    # directory / glob
    return pl.read_parquet(f"{path.rstrip('/')}/*.parquet")

def load_bars(path: str) -> pl.DataFrame:
    df = _read_any(path)
    missing = [c for c in BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.schema["timestamp"] == pl.String:
        df = df.with_columns(pl.col("timestamp").str.to_datetime(strict=False))
    return df.sort(["symbol", "timestamp"])

def _ema(col: str, span: int) -> pl.Expr:
    return pl.col(col).ewm_mean(span=span, adjust=False).over("symbol")

def add_features(df: pl.DataFrame, freq: str) -> pl.DataFrame:
    """
    Strictly causal bar features. No future data is used.
    Features are deliberately generic so the same code works for daily and 5m bars.
    """
    df = df.sort(["symbol", "timestamp"])

    # primitive returns / ranges
    df = df.with_columns([
        pl.col("close").pct_change().over("symbol").alias("ret_1"),
        pl.col("close").pct_change(2).over("symbol").alias("ret_2"),
        pl.col("close").pct_change(3).over("symbol").alias("ret_3"),
        pl.col("close").pct_change(6).over("symbol").alias("ret_6"),
        pl.col("close").pct_change(12).over("symbol").alias("ret_12"),
        ((pl.col("high") - pl.col("low")) / pl.col("close").shift(1).over("symbol"))
            .alias("range_frac"),
        ((pl.col("close") - pl.col("open")) / pl.col("open")).alias("body_frac"),
        (pl.col("volume") / pl.col("volume").rolling_mean(20).over("symbol"))
            .alias("rel_volume_20"),
    ])

    # realized vol
    for w in (6, 12, 20, 39, 78):
        df = df.with_columns(
            pl.col("ret_1").rolling_std(w).over("symbol").alias(f"rv_{w}")
        )

    # moving-average distance & trend
    for w in (5, 10, 20, 50, 100, 200):
        ma = pl.col("close").rolling_mean(w).over("symbol")
        df = df.with_columns(((pl.col("close") / ma) - 1.0).alias(f"dist_sma_{w}"))

    df = df.with_columns([
        ((_ema("close", 12) / _ema("close", 26)) - 1.0).alias("ema12_26"),
        (pl.col("close").rolling_max(20).over("symbol") / pl.col("close") - 1.0)
            .alias("dist_20_high"),
        (pl.col("close") / pl.col("close").rolling_min(20).over("symbol") - 1.0)
            .alias("dist_20_low"),
    ])

    # RSI-like normalized momentum, robust and causal
    delta = pl.col("close").diff().over("symbol")
    gain = delta.clip(lower_bound=0).rolling_mean(14).over("symbol")
    loss = (-delta.clip(upper_bound=0)).rolling_mean(14).over("symbol")
    rs = gain / (loss + 1e-12)
    df = df.with_columns((100 - 100 / (1 + rs)).alias("rsi_14"))

    # Time context
    df = df.with_columns([
        pl.col("timestamp").dt.weekday().alias("dow"),
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.minute().alias("minute"),
    ])

    # Cross-sectional context at each timestamp. This is one of the most important
    # "XTX-like" ideas: learn relative state across many assets.
    df = df.with_columns([
        pl.col("ret_1").mean().over("timestamp").alias("mkt_ret_1"),
        pl.col("ret_6").mean().over("timestamp").alias("mkt_ret_6"),
        pl.col("ret_1").rank("average").over("timestamp")
            .truediv(pl.len().over("timestamp")).alias("cs_rank_ret_1"),
        pl.col("ret_6").rank("average").over("timestamp")
            .truediv(pl.len().over("timestamp")).alias("cs_rank_ret_6"),
        pl.col("rel_volume_20").rank("average").over("timestamp")
            .truediv(pl.len().over("timestamp")).alias("cs_rank_volume"),
    ])

    if "sector" in df.columns:
        df = df.with_columns([
            pl.col("ret_1").mean().over(["timestamp", "sector"]).alias("sector_ret_1"),
            pl.col("ret_6").mean().over(["timestamp", "sector"]).alias("sector_ret_6"),
        ])
    else:
        df = df.with_columns([
            pl.lit(0.0).alias("sector_ret_1"),
            pl.lit(0.0).alias("sector_ret_6"),
        ])

    # residual returns relative to contemporaneous cross-section
    df = df.with_columns([
        (pl.col("ret_1") - pl.col("mkt_ret_1")).alias("resid_ret_1"),
        (pl.col("ret_6") - pl.col("mkt_ret_6")).alias("resid_ret_6"),
    ])
    return df

def add_targets(df: pl.DataFrame, freq: str) -> pl.DataFrame:
    """
    Multi-horizon future targets.
    We forecast residual future return divided by trailing volatility.
    """
    horizons = (1, 3, 6, 12) if freq == "5m" else (1, 3, 5, 10)
    out = df
    for h in horizons:
        fut = pl.col("close").shift(-h).over("symbol") / pl.col("close") - 1.0
        # Future market move is computed cross-sectionally at the same timestamp;
        # because it is a TARGET, future information here is allowed.
        name = f"ret_fwd_{h}"
        out = out.with_columns(fut.alias(name))
        out = out.with_columns(
            pl.col(name).mean().over("timestamp").alias(f"mkt_fwd_{h}")
        )
        out = out.with_columns([
            (pl.col(name) - pl.col(f"mkt_fwd_{h}")).alias(f"resid_fwd_{h}"),
            ((pl.col(name) - pl.col(f"mkt_fwd_{h}")) / (pl.col("rv_20") + 1e-6))
                .clip(-8, 8).alias(f"target_{h}")
        ])
    return out

def build_dataset(input_path: str, output_path: str, freq: str) -> None:
    df = load_bars(input_path)
    df = add_features(df, freq)
    df = add_targets(df, freq)
    df.write_parquet(output_path, compression="zstd")
