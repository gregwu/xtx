from __future__ import annotations
from pathlib import Path
import json, math, random
import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from .schema import FEATURES

class SequenceDataset(Dataset):
    def __init__(self, df: pl.DataFrame, seq_len: int, targets: list[str]):
        self.seq_len = seq_len
        self.targets = targets
        self.items = []
        self.feature_count = len(FEATURES)
        # A simple in-memory implementation for research. For huge datasets,
        # replace with memory-mapped shards.
        for _, g in df.partition_by("symbol", as_dict=True).items():
            g = g.sort("timestamp")
            X = g.select(FEATURES).to_numpy().astype(np.float32)
            Y = g.select(targets).to_numpy().astype(np.float32)
            ts = g["timestamp"].to_list()
            sym = g["symbol"][0]
            valid = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
            for i in range(seq_len - 1, len(g)):
                sl = slice(i-seq_len+1, i+1)
                if valid[sl].all():
                    self.items.append((X[sl].copy(), Y[i].copy(), sym, ts[i]))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        x,y,_,_ = self.items[i]
        return torch.from_numpy(x), torch.from_numpy(y)

class MarketTransformer(nn.Module):
    def __init__(self, n_features: int, n_targets: int, d_model=128, nhead=8,
                 layers=4, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.randn(1, 512, d_model) * 0.01)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_targets)
        )
    def forward(self, x):
        z = self.inp(x)
        z = z + self.pos[:, :z.size(1)]
        z = self.encoder(z)
        return self.head(self.norm(z[:, -1]))

def _slice(df, start=None, end=None):
    d = pl.col("timestamp").dt.date()
    if start: df = df.filter(d >= pl.lit(start).str.to_date())
    if end: df = df.filter(d <= pl.lit(end).str.to_date())
    return df

def _standardize(train, valid, test):
    # compute mean/std ONLY on training data
    stats = {}
    tr = train
    va = valid
    te = test
    for c in FEATURES:
        m = tr[c].mean()
        s = tr[c].std()
        s = 1.0 if s is None or not np.isfinite(s) or s < 1e-12 else float(s)
        m = 0.0 if m is None or not np.isfinite(m) else float(m)
        stats[c] = {"mean": m, "std": s}
        expr = ((pl.col(c)-m)/s).clip(-10,10).alias(c)
        tr = tr.with_columns(expr); va = va.with_columns(expr); te = te.with_columns(expr)
    return tr,va,te,stats

def train_transformer(
    data: str, outdir: str, train_end: str, valid_start: str, valid_end: str,
    test_start: str, seq_len: int=78, epochs: int=12, batch_size: int=512
):
    torch.manual_seed(7); np.random.seed(7); random.seed(7)
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(data).sort(["symbol","timestamp"])
    targets = sorted([c for c in df.columns if c.startswith("target_")],
                     key=lambda x:int(x.split("_")[-1]))
    train = _slice(df, end=train_end)
    valid = _slice(df, start=valid_start, end=valid_end)
    test  = _slice(df, start=test_start)
    train, valid, test, stats = _standardize(train,valid,test)
    (out/"normalization.json").write_text(json.dumps(stats, indent=2))

    tr_ds = SequenceDataset(train, seq_len, targets)
    va_ds = SequenceDataset(valid, seq_len, targets)
    te_ds = SequenceDataset(test, seq_len, targets)
    tr_dl = DataLoader(tr_ds,batch_size=batch_size,shuffle=True,num_workers=0)
    va_dl = DataLoader(va_ds,batch_size=batch_size,shuffle=False,num_workers=0)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = MarketTransformer(len(FEATURES),len(targets)).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-3)
    loss_fn = nn.HuberLoss(delta=1.0)
    best = math.inf
    history=[]
    for ep in range(epochs):
        model.train(); tr_loss=0; n=0
        for x,y in tr_dl:
            x=x.to(device); y=y.to(device)
            opt.zero_grad(set_to_none=True)
            pred=model(x)
            loss=loss_fn(pred,y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            tr_loss += loss.item()*len(x); n += len(x)
        model.eval(); va_loss=0; vn=0
        with torch.no_grad():
            for x,y in va_dl:
                x=x.to(device); y=y.to(device)
                l=loss_fn(model(x),y)
                va_loss += l.item()*len(x); vn += len(x)
        rec={"epoch":ep+1,"train_loss":tr_loss/max(n,1),"valid_loss":va_loss/max(vn,1)}
        history.append(rec)
        if rec["valid_loss"] < best:
            best=rec["valid_loss"]
            torch.save({"model":model.state_dict(),"targets":targets,"seq_len":seq_len}, out/"best.pt")
    (out/"history.json").write_text(json.dumps(history,indent=2))
    return history
