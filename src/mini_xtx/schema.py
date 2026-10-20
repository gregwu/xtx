BASE_FEATURES = [
    "ret_1","ret_2","ret_3","ret_6","ret_12",
    "range_frac","body_frac","rel_volume_20",
    "rv_6","rv_12","rv_20","rv_39","rv_78",
    "dist_sma_5","dist_sma_10","dist_sma_20","dist_sma_50",
    "dist_sma_100","dist_sma_200","ema12_26",
    "dist_20_high","dist_20_low","rsi_14",
    "dow","hour","minute",
    "mkt_ret_1","mkt_ret_6",
    "cs_rank_ret_1","cs_rank_ret_6","cs_rank_volume",
    "sector_ret_1","sector_ret_6",
    "resid_ret_1","resid_ret_6",
]

DAILY_STATE_FEATURES = [
    "ret_1","ret_3","ret_6","ret_12",
    "range_frac","body_frac","rel_volume_20",
    "rv_6","rv_12","rv_20","rv_39","rv_78",
    "dist_sma_20","dist_sma_50","dist_sma_100","dist_sma_200",
    "ema12_26","dist_20_high","dist_20_low","rsi_14",
    "mkt_ret_1","mkt_ret_6",
    "cs_rank_ret_1","cs_rank_ret_6","cs_rank_volume",
    "sector_ret_1","sector_ret_6","resid_ret_1","resid_ret_6",
]

# Backward-compatible alias for the original fast-only model.
FEATURES = BASE_FEATURES


def feature_columns(columns) -> list[str]:
    """Return model features available in a dataset, including optional daily state."""
    available = set(columns)
    out = [c for c in BASE_FEATURES if c in available]
    out.extend(f"daily_{c}" for c in DAILY_STATE_FEATURES if f"daily_{c}" in available)
    return out
