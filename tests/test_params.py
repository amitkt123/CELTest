"""Parameter provenance (spec §10.6) and unit fixes (spec Appendix B)."""
import csv
import dataclasses

import pytest

from cel.params import CSV_PATH, Params, S0, SCENARIOS, load_params, scenario

VALID_CLASSES = {"A", "B", "C", "D", "N"}
VALID_GRADES = {"A", "B", "C", "-"}


def _rows():
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_csv_has_exactly_one_row_per_field():
    names = [r["name"] for r in _rows()]
    assert len(names) == len(set(names))
    assert set(names) == {f.name for f in dataclasses.fields(Params)}


def test_every_row_has_class_grade_and_source():
    for r in _rows():
        assert r["class"] in VALID_CLASSES, r["name"]
        assert r["grade"] in VALID_GRADES, r["name"]
        assert r["source"].strip(), r["name"]


def test_values_lie_inside_stated_ranges():
    for r in _rows():
        if r["lower"] and r["upper"]:
            lo, hi, v = float(r["lower"]), float(r["upper"]), float(r["value"])
            assert lo <= v <= hi, r["name"]


def test_s0_matches_csv_with_declared_types():
    rows = {r["name"]: r for r in _rows()}
    for f in dataclasses.fields(Params):
        v = getattr(S0, f.name)
        assert isinstance(v, int if f.type == "int" else float), f.name
        assert v == pytest.approx(float(rows[f.name]["value"])), f.name


def test_load_params_rejects_out_of_sync_csv(tmp_path):
    rows = _rows()[:-1]
    bad = tmp_path / "parameters.csv"
    with bad.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(KeyError, match="out of sync"):
        load_params(bad)


def test_scenarios_are_valid_diffs():
    for name in SCENARIOS:
        scenario(name)
    with pytest.raises(KeyError):
        S0.with_(not_a_param=1.0)


def test_unit_inconsistencies_are_fixed():
    assert 1.0 - 1.0 / S0.m_H == pytest.approx(0.75)                  # 75% gross margin
    assert S0.kappa0 * S0.eps0 * S0.unit_kW == pytest.approx(30_000.0)  # $ per 1 kW unit
    assert S0.e0 == pytest.approx(1e20)                                # worker-year anchor
