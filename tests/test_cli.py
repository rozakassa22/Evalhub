"""Tests for the evalhub-process CLI."""
from __future__ import annotations

import json

from evalhub.processing.cli import main


def test_gen_then_report(tmp_path, capsys):
    out = tmp_path / "gen.jsonl"
    rc = main(["gen", "20", "--out", str(out), "--seed", "1"])
    assert rc == 0
    assert out.exists()
    assert sum(1 for _ in out.open()) == 20
    capsys.readouterr()  # clear the "wrote ..." line

    rc = main(["report", str(out)])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["samples"] == 20
    assert report["mean_token_f1"] is not None


def test_report_benchmark(tmp_path, capsys):
    out = tmp_path / "gen.jsonl"
    main(["gen", "10", "--out", str(out)])
    capsys.readouterr()

    rc = main(["report", str(out), "--benchmark"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "throughput_samples_per_sec" in payload
    assert payload["report"]["samples"] == 10


def test_report_missing_file(capsys):
    rc = main(["report", "/no/such/file.jsonl"])
    assert rc == 2
    assert "no such file" in capsys.readouterr().err
