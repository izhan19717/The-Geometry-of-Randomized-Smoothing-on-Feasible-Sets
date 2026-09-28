from fractions import Fraction

import mpmath as mp
import pytest

from feasible_robustness.rational_interval import (
    RationalInterval,
    gaussian_kernel_mass,
    gaussian_kernel_value,
    log_rational,
    pi_interval,
)
from experiments.certify_local_lshape_witness import certify, interval_payload


TOLERANCE = Fraction(1, 10**45)


def test_rational_gaussian_primitives_enclose_high_precision_regression_values():
    """mpmath is only an independent regression oracle, not part of the proof path."""

    mp.mp.dps = 80
    for point in (Fraction(0), Fraction(1, 15), Fraction(-13, 3), Fraction(17, 5)):
        interval = gaussian_kernel_value(point, TOLERANCE)
        assert interval.width <= TOLERANCE
        value = mp.exp(-mp.mpf(point.numerator) ** 2 / (2 * point.denominator**2))
        assert mp.mpf(interval.lo.numerator) / interval.lo.denominator <= value
        assert value <= mp.mpf(interval.hi.numerator) / interval.hi.denominator

    mass = gaussian_kernel_mass(Fraction(-13, 3), Fraction(7, 3), TOLERANCE)
    assert mass.width <= 2 * TOLERANCE
    expected = mp.quad(lambda x: mp.exp(-(x * x) / 2), [-mp.mpf(13) / 3, mp.mpf(7) / 3])
    assert mp.mpf(mass.lo.numerator) / mass.lo.denominator <= expected
    assert expected <= mp.mpf(mass.hi.numerator) / mass.hi.denominator


def test_rational_log_and_pi_enclosures_cover_high_precision_values():
    mp.mp.dps = 80
    log_bound = log_rational(Fraction(97, 100), TOLERANCE)
    assert log_bound.width <= TOLERANCE
    expected_log = mp.log(mp.mpf(97) / 100)
    assert mp.mpf(log_bound.lo.numerator) / log_bound.lo.denominator <= expected_log
    assert expected_log <= mp.mpf(log_bound.hi.numerator) / log_bound.hi.denominator

    pi_bound = pi_interval(TOLERANCE)
    assert pi_bound.width <= TOLERANCE
    assert mp.mpf(pi_bound.lo.numerator) / pi_bound.lo.denominator <= mp.pi
    assert mp.pi <= mp.mpf(pi_bound.hi.numerator) / pi_bound.hi.denominator


def test_local_lshape_has_strict_certified_kl_and_tv_excess():
    certificate = certify()
    intervals = certificate["intervals"]

    assert intervals["crossing_x"].lo > Fraction(1, 2)
    assert intervals["crossing_x"].hi < 1
    assert intervals["kl"].lo > intervals["gaussian_kl"].hi
    assert intervals["kl_excess"].lo > Fraction(1, 5000)
    assert intervals["exact_tv"].lo > intervals["gaussian_tv"].hi
    assert intervals["tv_excess"].lo > Fraction(1, 500)
    assert intervals["fixed_event_probability_gap"].lo > intervals["gaussian_tv"].hi
    assert intervals["fixed_event_excess_over_gaussian_tv"].lo > Fraction(1, 500)
    assert intervals["decision_event_probability_a"].lo > Fraction(1, 2)
    assert intervals["decision_event_probability_b"].hi < Fraction(1, 2)
    assert intervals["decision_event_probability_a"].lo > Fraction(527, 1000)
    assert intervals["decision_event_probability_b"].hi < Fraction(499, 1000)
    assert intervals["decision_event_margin_a"].lo > intervals["twice_gaussian_tv"].hi
    assert intervals["gaussian_rate_false_certificate_slack"].lo > Fraction(1, 1000)
    assert intervals["variance_over_sigma_squared"].lo > Fraction(109, 100)
    assert intervals["conditional_mad_over_gaussian_mad"].lo > Fraction(107, 100)
    assert intervals["mad_split_location"].lo > Fraction(1, 2)
    assert intervals["mad_split_location"].hi < 1
    assert (
        intervals["decision_event_probability_a"].lo
        > intervals["cohen_probability_threshold_phi_1_over_15"].hi
    )
    assert intervals["cohen_probability_slack"].lo > Fraction(9, 10000)


def test_published_decimal_intervals_are_outward_roundings():
    certificate = certify()
    for interval in certificate["intervals"].values():
        payload = interval_payload(interval)
        lower = Fraction(payload["lower_rational"])
        upper = Fraction(payload["upper_rational"])
        assert lower <= interval.lo <= interval.hi <= upper


def test_interval_primitive_preconditions_are_enforced():
    with pytest.raises(ValueError, match="lower integration endpoint"):
        gaussian_kernel_mass(Fraction(1), Fraction(0), TOLERANCE)
    with pytest.raises(ValueError, match="log argument"):
        log_rational(Fraction(0), TOLERANCE)
    with pytest.raises(ZeroDivisionError, match="containing zero"):
        RationalInterval(Fraction(-1), Fraction(1)).reciprocal()
