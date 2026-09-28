from fractions import Fraction

from experiments.certify_two_band_covariance_radius import certify, interval_payload


def test_two_band_example_has_the_published_strict_radius_ordering():
    certificate = certify()
    intervals = certificate["intervals"]

    assert intervals["joint_mass_radius"].hi < intervals["covariance_controlled_radius"].lo
    assert intervals["covariance_controlled_radius"].hi < Fraction(1, 10)
    assert Fraction(1, 10) < intervals["substituted_conditional_radius"].lo
    assert intervals["covariance_controlled_radius"].lo > Fraction(94, 1000)
    assert intervals["substituted_conditional_radius"].lo > Fraction(118, 1000)
    assert all(certificate["certified"].values())


def test_two_band_published_decimals_are_outward_roundings():
    certificate = certify()
    for interval in certificate["intervals"].values():
        payload = interval_payload(interval)
        lower = Fraction(payload["lower_rational"])
        upper = Fraction(payload["upper_rational"])
        assert lower <= interval.lo <= interval.hi <= upper
