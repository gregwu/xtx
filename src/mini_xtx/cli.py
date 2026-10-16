from __future__ import annotations
import argparse, json
from pathlib import Path
from .features import build_dataset
from .lgbm_model import train_lgbm
from .transformer_model import train_transformer
from .backtest import backtest_ranked

def main():
    p=argparse.ArgumentParser("mini-xtx")
    sp=p.add_subparsers(dest="cmd",required=True)

    b=sp.add_parser("build")
    b.add_argument("--input",required=True)
    b.add_argument("--output",required=True)
    b.add_argument("--freq",choices=["5m","1d"],required=True)

    l=sp.add_parser("train-lgbm")
    l.add_argument("--data",required=True); l.add_argument("--outdir",required=True)
    l.add_argument("--train-end",required=True); l.add_argument("--valid-start",required=True)
    l.add_argument("--valid-end",required=True); l.add_argument("--test-start",required=True)

    t=sp.add_parser("train-transformer")
    t.add_argument("--data",required=True); t.add_argument("--outdir",required=True)
    t.add_argument("--train-end",required=True); t.add_argument("--valid-start",required=True)
    t.add_argument("--valid-end",required=True); t.add_argument("--test-start",required=True)
    t.add_argument("--seq-len",type=int,default=78)
    t.add_argument("--epochs",type=int,default=12); t.add_argument("--batch-size",type=int,default=512)

    bt=sp.add_parser("backtest")
    bt.add_argument("--predictions",required=True)
    bt.add_argument("--horizon",required=True)
    bt.add_argument("--top-frac",type=float,default=0.02)
    bt.add_argument("--cost-bps",type=float,default=2.0)

    a=p.parse_args()
    if a.cmd=="build":
        build_dataset(a.input,a.output,a.freq)
    elif a.cmd=="train-lgbm":
        m=train_lgbm(a.data,a.outdir,a.train_end,a.valid_start,a.valid_end,a.test_start)
        print(json.dumps(m,indent=2))
    elif a.cmd=="train-transformer":
        h=train_transformer(a.data,a.outdir,a.train_end,a.valid_start,a.valid_end,a.test_start,
                            a.seq_len,a.epochs,a.batch_size)
        print(json.dumps(h[-3:],indent=2))
    elif a.cmd=="backtest":
        s,pnl=backtest_ranked(a.predictions,a.horizon,a.top_frac,a.cost_bps)
        print(json.dumps(s,indent=2))
        out=Path(a.predictions).with_name("backtest_"+a.horizon+".parquet")
        pnl.write_parquet(out)
        print(f"wrote {out}")

if __name__=="__main__":
    main()
