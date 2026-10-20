from __future__ import annotations
import polars as pl
from .schema import DAILY_STATE_FEATURES

BASE_COLS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]


def _read_any(path: str) -> pl.DataFrame:
    p = path.lower()
    if p.endswith(".parquet"):
        return pl.read_parquet(path)
    if p.endswith(".csv"):
        return pl.read_csv(path, try_parse_dates=True)
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
    """Strictly causal bar features. No future data is used."""
    df = df.sort(["symbol", "timestamp"])

    df = df.with_columns([
        pl.col("close").pct_change().over("symbol").alias("ret_1"),
        pl.col("close").pct_change(2).over("symbol").alias("ret_2"),
        pl.col("close").pct_change(3).over("symbol").alias("ret_3"),
        pl.col("close").pct_change(6).over("symbol").alias("ret_6"),
        pl.col("close").pct_change(12).over("symbol").alias("ret_12"),
        ((pl.col("high") - pl.col("low")) / pl.col("close").shift(1).over("symbol")).alias("range_frac"),
        ((pl.col("close") - pl.col("open")) / pl.col("open")).alias("body_frac"),
        (pl.col("volume") / pl.col("volume").rolling_mean(20).over("symbol")).alias("rel_volume_20"),
    ])

    for w in (6, 12, 20, 39, 78):
        df = df.with_columns(pl.col("ret_1").rolling_std(w).over("symbol").alias(f"rv_{w}"))

    for w in (5, 10, 20, 50, 100, 200):
        ma = pl.col("close").rolling_mean(w).over("symbol")
        df = df.with_columns(((pl.col("close") / ma) - 1.0).alias(f"dist_sma_{w}"))

    df = df.with_columns([
        ((_ema("close", 12) / _ema("close", 26)) - 1.0).alias("ema12_26"),
        (pl.col("close").rolling_max(20).over("symbol") / pl.col("close") - 1.0).alias("dist_20_high"),
        (pl.col("close") / pl.col("close").rolling_min(20).over("symbol") - 1.0).alias("dist_20_low"),
    ])

    delta = pl.col("close").diff().over("symbol")
    gain = delta.clip(lower_bound=0).rolling_mean(14).over("symbol")
    loss = (-delta.clip(upper_bound=0)).rolling_mean(14).over("symbol")
    rs = gain / (loss + 1e-12)
    df = df.with_columns((100 - 100 / (1 + rs)).alias("rsi_14"))

    df = df.with_columns([
        pl.col("timestamp").dt.weekday().alias("dow"),
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.minute().alias("minute"),
    ])

    df = df.with_columns([
        pl.col("ret_1").mean().over("timestamp").alias("mkt_ret_1"),
        pl.col("ret_6").mean().over("timestamp").alias("mkt_ret_6"),
        pl.col("ret_1").rank("average").over("timestamp").truediv(pl.len().over("timestamp")).alias("cs_rank_ret_1"),
        pl.col("ret_6").rank("average").over("timestamp").truediv(pl.len().over("timestamp")).alias("cs_rank_ret_6"),
        pl.col("rel_volume_20").rank("average").over("timestamp").truediv(pl.len().over("timestamp")).alias("cs_rank_volume"),
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

    return df.with_columns([
        (pl.col("ret_1") - pl.col("mkt_ret_1")).alias("resid_ret_1"),
        (pl.col("ret_6") - pl.col("mkt_ret_6")).alias("resid_ret_6"),
    ])


def _future_extreme(col: str, h: int, op: str) -> pl.Expr:
    exprs = [pl.col(col).shift(-k).over("symbol") for k in range(1, h + 1)]
    if op == "max":
        return pl.max_horizontal(exprs)
    if op == "min":
        return pl.min_horizontal(exprs)
    raise ValueError(op)


def add_targets(df: pl.DataFrame, freq: str) -> pl.DataFrame:
    """
    Multi-horizon targets:
    - close-to-future-close residual return (research target)
    - next-open-to-future-close executable return
    - next-open MFE / MAE path targets
    """
    horizons = (1, 3, 6, 12) if freq == "5m" else (1, 3, 5, 10)
    out = df

    for h in horizons:
        close_fut = pl.col("close").shift(-h).over("symbol")
        next_open = pl.col("open").shift(-1).over("symbol")
        future_high = _future_extreme("high", h, "max")
        future_low = _future_extreme("low", h, "min")
        name = f"ret_fwd_{h}"

        out = out.with_columns([
            (close_fut / pl.col("close") - 1.0).alias(name),
            (close_fut / next_open - 1.0).alias(f"ret_exec_fwd_{h}"),
            (future_high / next_open - 1.0).alias(f"mfe_fwd_{h}"),
            (future_low / next_open - 1.0).alias(f"mae_fwd_{h}"),
        ])
        out = out.with_columns(
            pl.col(name).mean().over("timestamp").alias(f"mkt_fwd_{h}")
        )
        out = out.with_columns([
            (pl.col(name) - pl.col(f"mkt_fwd_{h}")).alias(f"resid_fwd_{h}"),
            ((pl.col(name) - pl.col(f"mkt_fwd_{h}")) / (pl.col("rv_20") + 1e-6))
                .clip(-8, 8).alias(f"target_{h}"),
            (pl.col(f"mfe_fwd_{h}") / (pl.col("rv_20") + 1e-6))
                .clip(0, 12).alias(f"target_mfe_{h}"),
            (pl.col(f"mae_fwd_{h}") / (pl.col("rv_20") + 1e-6))
                .clip(-12, 0).alias(f"target_mae_{h}"),
        ])

    return out


def add_daily_state(intraday: pl.DataFrame, daily: pl.DataFrame) -> pl.DataFrame:
    """
    Join prior-session daily state onto every intraday bar.

    The daily row for session D is shifted by one symbol-row before the join, so
    intraday bars on D can only see daily information that was known after D-1.
    """
    daily = add_features(daily, "1d").sort(["symbol", "timestamp"])
    present = [c for c in DAILY_STATE_FEATURES if c in daily.columns]

    daily_state = (
        daily.select(["symbol", "timestamp"] + present)
        .with_columns(pl.col("timestamp").dt.date().alias("session_date"))
        .with_columns([
            pl.col(c).shift(1).over("symbol").alias(f"daily_{c}") for c in present
        ])
        .select(["symbol", "session_date"] + [f"daily_{c}" for c in present])
    )

    return (
        intraday.with_columns(pl.col("timestamp").dt.date().alias("session_date"))
        .join(daily_state, on=["symbol", "session_date"], how="left")
        .drop("session_date")
    )


def build_dataset(
    input_path: str,
    output_path: str,
    freq: str,
    daily_path: str | None = None,
) -> None:
    df = load_bars(input_path)
    df = add_features(df, freq)
    if freq == "5m" and daily_path:
        df = add_daily_state(df, load_bars(daily_path))
    df = add_targets(df, freq)
    df.write_parquet(output_path, compression="zstd")
