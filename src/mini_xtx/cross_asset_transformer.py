from __future__ import annotations

from pathlib import Path
import json
import math
import random
import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from .schema import feature_columns


class CrossAssetDataset(Dataset):
    """
    Fixed-size [assets, time, features] samples.

    Each sample is one decision timestamp. For every selected symbol, the model
    receives that symbol's trailing sequence. The endpoint is synchronized
    cross-sectionally, while each history is causal within the symbol.
    """
    def __init__(
        self,
        df: pl.DataFrame,
        features: list[str],
        horizon: int,
        seq_len: int,
        max_assets: int,
        normalization: dict[str, tuple[float, float]],
    ):
        self.features = features
        self.seq_len = seq_len
        self.max_assets = max_assets
        self.horizon = horizon
        self.targets = [f"target_{horizon}", f"target_mfe_{horizon}", f"target_mae_{horizon}"]
        self.symbol_data = {}
        endpoint = {}

        for key, g in df.sort(["symbol", "timestamp"]).partition_by("symbol", as_dict=True).items():
            sym = key[0] if isinstance(key, tuple) else key
            g = g.sort("timestamp")
            X = g.select(features).to_numpy().astype(np.float32)
            for j, c in enumerate(features):
                m, s = normalization[c]
                X[:, j] = np.clip((X[:, j] - m) / s, -10, 10)
            Y = g.select(self.targets).to_numpy().astype(np.float32)
            times = g["timestamp"].to_list()
            volume_rank = (
                g["cs_rank_volume"].fill_null(0.0).to_numpy()
                if "cs_rank_volume" in g.columns else np.zeros(len(g), dtype=np.float32)
            )
            self.symbol_data[sym] = (X, Y, times, volume_rank)
            for i, ts in enumerate(times):
                if i >= seq_len - 1 and np.isfinite(Y[i]).all() and np.isfinite(X[i-seq_len+1:i+1]).all():
                    endpoint.setdefault(ts, []).append((sym, i, float(volume_rank[i])))

        self.samples = []
        for ts in sorted(endpoint):
            names = sorted(endpoint[ts], key=lambda x: x[2], reverse=True)[:max_assets]
            if len(names) >= 2:
                self.samples.append((ts, names))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        _, names = self.samples[index]
        A, T, F = self.max_assets, self.seq_len, len(self.features)
        x = np.zeros((A, T, F), dtype=np.float32)
        y = np.zeros((A, 3), dtype=np.float32)
        mask = np.zeros(A, dtype=np.bool_)
        for a, (sym, i, _) in enumerate(names):
            X, Y, _, _ = self.symbol_data[sym]
            x[a] = X[i-T+1:i+1]
            y[a] = Y[i]
            mask[a] = True
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)

    def metadata(self, index):
        ts, names = self.samples[index]
        return ts, [x[0] for x in names]


class XTXStyleTransformer(nn.Module):
    """
    Slow/fast encoder -> temporal transformer -> cross-asset transformer ->
    multi-task return/MFE/MAE heads.
    """
    def __init__(
        self,
        n_features: int,
        fast_idx: list[int],
        daily_idx: list[int],
        d_model: int = 128,
        nhead: int = 8,
        temporal_layers: int = 3,
        cross_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.fast_idx = fast_idx
        self.daily_idx = daily_idx
        self.fast_proj = nn.Linear(max(len(fast_idx), 1), d_model)
        self.daily_proj = nn.Sequential(
            nn.Linear(max(len(daily_idx), 1), d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.01)

        temporal = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        cross = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.temporal = nn.TransformerEncoder(temporal, num_layers=temporal_layers)
        self.cross = nn.TransformerEncoder(cross, num_layers=cross_layers)
        self.norm = nn.LayerNorm(d_model)
        self.return_head = nn.Linear(d_model, 1)
        self.mfe_head = nn.Linear(d_model, 1)
        self.mae_head = nn.Linear(d_model, 1)

    def _slice_or_zero(self, x, idx):
        if idx:
            return x[..., idx]
        return torch.zeros(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)

    def forward(self, x, asset_mask):
        B, A, T, _ = x.shape
        fast = self._slice_or_zero(x, self.fast_idx)
        daily = self._slice_or_zero(x, self.daily_idx)
        z = self.fast_proj(fast) + self.daily_proj(daily)
        z = z + self.pos[:, :T].unsqueeze(1)
        z = z.reshape(B * A, T, -1)
        z = self.temporal(z)[:, -1].reshape(B, A, -1)
        z = self.cross(z, src_key_padding_mask=~asset_mask)
        z = self.norm(z)
        return torch.cat([
            self.return_head(z),
            self.mfe_head(z),
            self.mae_head(z),
        ], dim=-1)


def _normalization(train: pl.DataFrame, features: list[str]):
    stats = {}
    for c in features:
        m = train[c].mean()
        s = train[c].std()
        m = float(m) if m is not None and np.isfinite(m) else 0.0
        s = float(s) if s is not None and np.isfinite(s) and s > 1e-12 else 1.0
        stats[c] = (m, s)
    return stats


def _masked_huber(pred, target, mask, weights=(1.0, 0.35, 0.35)):
    valid = mask.unsqueeze(-1).expand_as(pred)
    loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
    w = torch.tensor(weights, device=pred.device, dtype=pred.dtype).view(1, 1, 3)
    loss = loss * w
    return loss[valid].mean()


def train_cross_asset_transformer(
    data: str,
    outdir: str,
    horizon: int,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_start: str,
    seq_len: int = 24,
    max_assets: int = 128,
    epochs: int = 10,
    batch_size: int = 8,
    d_model: int = 128,
):
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(data).sort(["timestamp", "symbol"])
    features = feature_columns(df.columns)
    required = [f"target_{horizon}", f"target_mfe_{horizon}", f"target_mae_{horizon}"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing targets: {missing}")

    d = pl.col("timestamp").dt.date()
    train = df.filter(d <= pl.lit(train_end).str.to_date())
    valid = df.filter(
        (d >= pl.lit(valid_start).str.to_date()) &
        (d <= pl.lit(valid_end).str.to_date())
    )
    test = df.filter(d >= pl.lit(test_start).str.to_date())
    stats = _normalization(train, features)
    (out / "normalization.json").write_text(json.dumps(
        {k: {"mean": v[0], "std": v[1]} for k, v in stats.items()}, indent=2
    ))

    tr_ds = CrossAssetDataset(train, features, horizon, seq_len, max_assets, stats)
    va_ds = CrossAssetDataset(valid, features, horizon, seq_len, max_assets, stats)
    te_ds = CrossAssetDataset(test, features, horizon, seq_len, max_assets, stats)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    daily_idx = [i for i, c in enumerate(features) if c.startswith("daily_")]
    fast_idx = [i for i, c in enumerate(features) if not c.startswith("daily_")]
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = XTXStyleTransformer(
        len(features), fast_idx, daily_idx, d_model=d_model, max_seq_len=max(seq_len, 256)
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)

    best = math.inf
    history = []
    for epoch in range(epochs):
        model.train()
        total = count = 0
        for x, y, mask in tr_dl:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x, mask)
            loss = _masked_huber(pred, y, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(x)
            count += len(x)

        model.eval()
        vt = vc = 0
        with torch.no_grad():
            for x, y, mask in va_dl:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                loss = _masked_huber(model(x, mask), y, mask)
                vt += float(loss.item()) * len(x)
                vc += len(x)
        rec = {
            "epoch": epoch + 1,
            "train_loss": total / max(count, 1),
            "valid_loss": vt / max(vc, 1),
        }
        history.append(rec)
        if rec["valid_loss"] < best:
            best = rec["valid_loss"]
            torch.save({
                "state_dict": model.state_dict(),
                "features": features,
                "horizon": horizon,
                "seq_len": seq_len,
                "max_assets": max_assets,
                "fast_idx": fast_idx,
                "daily_idx": daily_idx,
                "d_model": d_model,
            }, out / "best.pt")

    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    rows = []
    with torch.no_grad():
        for i in range(len(te_ds)):
            x, _, mask = te_ds[i]
            pred = model(x.unsqueeze(0).to(device), mask.unsqueeze(0).to(device))[0].cpu().numpy()
            ts, syms = te_ds.metadata(i)
            for a, sym in enumerate(syms):
                rows.append((sym, ts, float(pred[a, 0]), float(pred[a, 1]), float(pred[a, 2])))

    preds = pl.DataFrame(
        rows,
        schema=["symbol", "timestamp", f"pred_{horizon}", f"pred_mfe_{horizon}", f"pred_mae_{horizon}"],
        orient="row",
    )
    truth_cols = [
        "symbol", "timestamp", f"ret_fwd_{horizon}", f"ret_exec_fwd_{horizon}",
        f"mfe_fwd_{horizon}", f"mae_fwd_{horizon}"
    ]
    truth_cols = [c for c in truth_cols if c in test.columns]
    preds = preds.join(test.select(truth_cols), on=["symbol", "timestamp"], how="left")
    preds.write_parquet(out / "predictions.parquet")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    return history
