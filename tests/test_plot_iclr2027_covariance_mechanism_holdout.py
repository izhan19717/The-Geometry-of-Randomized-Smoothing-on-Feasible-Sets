from pathlib import Path

from experiments.plot_iclr2027_covariance_mechanism_holdout import load_frozen_rows


def test_frozen_covariance_figure_rows_match_reported_units():
    rows = load_frozen_rows(
        Path("outputs/iclr2027_covariance_mechanism_holdout_20260912/factor_rows.csv")
    )
    assert len(rows) == 1024
    exceedances = [row for row in rows if float(row["finite_kl_ratio"]) > 1]
    assert len(exceedances) == 27
    assert len({(row["dimension"], row["geometry_seed"]) for row in exceedances}) == 14
