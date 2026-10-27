from __future__ import annotations
import argparse
import json
from pathlib import Path

from .features import build_dataset
from .lgbm_model import train_lgbm
from .transformer_model import train_transformer
from .backtest import backtest_ranked
from .evaluation import evaluate_predictions
from .portfolio import simulate_portfolio
from .walkforward import run_walkforward


def main():
    p = argparse.ArgumentParser("mini-xtx")
    sp = p.add_subparsers(dest="cmd", required=True)

    b = sp.add_parser("build")
    b.add_argument("--input", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--freq", choices=["5m", "1d"], required=True)
    b.add_argument("--daily-input")

    l = sp.add_parser("train-lgbm")
    l.add_argument("--data", required=True)
    l.add_argument("--outdir", required=True)
    l.add_argument("--train-end", required=True)
    l.add_argument("--valid-start", required=True)
    l.add_argument("--valid-end", required=True)
    l.add_argument("--test-start", required=True)
    l.add_argument("--purge-bars", type=int, default=None)
    l.add_argument("--no-aux-targets", action="store_true")

    t = sp.add_parser("train-transformer")
    t.add_argument("--data", required=True)
    t.add_argument("--outdir", required=True)
    t.add_argument("--train-end", required=True)
    t.add_argument("--valid-start", required=True)
    t.add_argument("--valid-end", required=True)
    t.add_argument("--test-start", required=True)
    t.add_argument("--seq-len", type=int, default=78)
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--batch-size", type=int, default=512)

    wf = sp.add_parser("walkforward")
    wf.add_argument("--data", required=True)
    wf.add_argument("--outdir", required=True)
    wf.add_argument("--horizon", type=int, required=True)
    wf.add_argument("--start", required=True)
    wf.add_argument("--end")
    wf.add_argument("--models", default="ridge,lgbm")
    wf.add_argument("--purge-bars", type=int, default=None)
    wf.add_argument("--eval-stride", type=int, default=None)
    wf.add_argument("--ridge-alpha", type=float, default=10.0)

    ev = sp.add_parser("evaluate")
    ev.add_argument("--predictions", required=True)
    ev.add_argument("--outdir", required=True)
    ev.add_argument("--horizon", type=int, required=True)
    ev.add_argument("--pred-col")
    ev.add_argument("--top-frac", type=float, default=0.02)
    ev.add_argument("--cost-bps", type=float, default=2.0)
    ev.add_argument("--stride", type=int, default=None)
    ev.add_argument("--periods-per-year", type=float, default=None)
    ev.add_argument("--gross-exposure", type=float, default=1.0)
    ev.add_argument("--max-position", type=float, default=0.05)
    ev.add_argument("--max-turnover", type=float, default=1.0)

    pf = sp.add_parser("portfolio")
    pf.add_argument("--predictions", required=True)
    pf.add_argument("--output", required=True)
    pf.add_argument("--horizon", type=int, required=True)
    pf.add_argument("--score-col")
    pf.add_argument("--top-frac", type=float, default=0.02)
    pf.add_argument("--gross-exposure", type=float, default=1.0)
    pf.add_argument("--max-position", type=float, default=0.05)
    pf.add_argument("--cost-bps", type=float, default=2.0)
    pf.add_argument("--max-turnover", type=float, default=1.0)
    pf.add_argument("--stride", type=int, default=None)
    pf.add_argument("--periods-per-year", type=float, default=None)

    bt = sp.add_parser("backtest")
    bt.add_argument("--predictions", required=True)
    bt.add_argument("--horizon", required=True)
    bt.add_argument("--top-frac", type=float, default=0.02)
    bt.add_argument("--cost-bps", type=float, default=2.0)
    bt.add_argument("--stride", type=int, default=None)
    bt.add_argument("--same-close-entry", action="store_true")

    a = p.parse_args()

    if a.cmd == "build":
        build_dataset(a.input, a.output, a.freq, a.daily_input)

    elif a.cmd == "train-lgbm":
        m = train_lgbm(
            a.data, a.outdir, a.train_end, a.valid_start, a.valid_end,
            a.test_start, a.purge_bars,
            auxiliary_targets=not a.no_aux_targets,
        )
        print(json.dumps(m, indent=2))

    elif a.cmd == "train-transformer":
        h = train_transformer(
            a.data, a.outdir, a.train_end, a.valid_start, a.valid_end,
            a.test_start, a.seq_len, a.epochs, a.batch_size,
        )
        print(json.dumps(h[-3:], indent=2))

    elif a.cmd == "walkforward":
        result = run_walkforward(
            a.data,
            a.outdir,
            a.horizon,
            a.start,
            a.end,
            tuple(x.strip() for x in a.models.split(",") if x.strip()),
            a.purge_bars,
            a.ridge_alpha,
            a.eval_stride,
        )
        print(json.dumps(result, indent=2))

    elif a.cmd == "evaluate":
        result = evaluate_predictions(
            a.predictions,
            a.outdir,
            a.horizon,
            a.pred_col,
            a.top_frac,
            a.cost_bps,
            a.stride,
            a.periods_per_year,
            a.gross_exposure,
            a.max_position,
            a.max_turnover,
        )
        print(json.dumps(result, indent=2))

    elif a.cmd == "portfolio":
        result = simulate_portfolio(
            a.predictions,
            a.output,
            a.horizon,
            a.score_col,
            a.top_frac,
            a.gross_exposure,
            a.max_position,
            a.cost_bps,
            a.max_turnover,
            a.stride,
            a.periods_per_year,
        )
        print(json.dumps(result, indent=2))

    elif a.cmd == "backtest":
        s, pnl = backtest_ranked(
            a.predictions,
            a.horizon,
            a.top_frac,
            a.cost_bps,
            a.stride,
            executable=not a.same_close_entry,
        )
        print(json.dumps(s, indent=2))
        out = Path(a.predictions).with_name("backtest_" + a.horizon + ".parquet")
        pnl.write_parquet(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
