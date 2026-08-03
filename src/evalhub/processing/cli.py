"""Command-line entry point for the batch processing tool.

Usage examples::

    evalhub-process report data/samples.jsonl
    evalhub-process report data/samples.jsonl --benchmark
    evalhub-process gen 100000 --out data/big.jsonl

Installed as the ``evalhub-process`` console script (see pyproject.toml).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from .. import __version__
from ..core.models import Sample
from .fast_metrics import benchmark, corpus_report, iter_jsonl_samples


def _cmd_report(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 2

    if args.benchmark:
        result = benchmark(iter_jsonl_samples(path))
        payload = {
            "throughput_samples_per_sec": round(result.samples_per_second, 1),
            "elapsed_seconds": round(result.seconds, 4),
            "report": result.report.as_dict(),
        }
    else:
        payload = corpus_report(iter_jsonl_samples(path)).as_dict()

    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _cmd_gen(args: argparse.Namespace) -> int:
    """Generate a synthetic JSONL dataset for benchmarking."""
    rng = random.Random(args.seed)
    words = [
        "paris", "tokyo", "canberra", "the", "capital", "of", "is", "city",
        "answer", "model", "correct", "berlin", "rome", "madrid", "cairo",
    ]

    def sentence(n: int) -> str:
        return " ".join(rng.choice(words) for _ in range(n))

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as fh:
        for _ in range(args.count):
            ref = sentence(rng.randint(1, 6))
            # Predictions overlap the reference to varying degrees.
            pred = ref if rng.random() < 0.4 else sentence(rng.randint(1, 10))
            sample = Sample(input=sentence(5), prediction=pred, reference=ref)
            fh.write(sample.model_dump_json(exclude_none=True) + "\n")

    print(f"wrote {args.count} samples to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalhub-process",
        description="Streaming corpus statistics over evaluation samples.",
    )
    parser.add_argument("--version", action="version", version=f"evalhub {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    report = sub.add_parser("report", help="Summarize a JSONL sample file.")
    report.add_argument("path", help="Path to a .jsonl file of samples.")
    report.add_argument(
        "--benchmark",
        action="store_true",
        help="Also report processing throughput.",
    )
    report.set_defaults(func=_cmd_report)

    gen = sub.add_parser("gen", help="Generate a synthetic JSONL dataset.")
    gen.add_argument("count", type=int, help="Number of samples to generate.")
    gen.add_argument("--out", default="data/generated.jsonl", help="Output path.")
    gen.add_argument("--seed", type=int, default=0)
    gen.set_defaults(func=_cmd_gen)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
