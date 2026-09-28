"""Sound certificates for Gaussian smoothing followed by a fixed filter."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
import math

from scipy.stats import beta, norm


Label = Hashable


@dataclass(frozen=True)
class JointMassCertificate:
    """Finite-sample certificate based on joint retained-label masses."""

    selected_label: Label
    runner_up: Label
    selected_count: int
    proposal_count: int
    selected_lower: float
    runner_upper: float
    radius: float
    confidence: float
    selection_independent: bool


@dataclass(frozen=True)
class HybridMassCertificate:
    """Finite-sample certificate combining two valid runner bounds."""

    selected_label: Label
    runner_up: Label
    selected_count: int
    proposal_count: int
    selected_lower: float
    explicit_runner_upper: float
    complement_runner_upper: float
    runner_upper: float
    radius: float
    confidence: float


def _validate_probability(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return result


def _validate_scale(sigma: float) -> float:
    value = float(sigma)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("sigma must be finite and positive")
    return value


def one_sided_binomial_lower(
    successes: int, trials: int, error: float
) -> float:
    """Return the exact one-sided Clopper--Pearson lower endpoint."""

    if not isinstance(successes, int) or not isinstance(trials, int):
        raise TypeError("successes and trials must be integers")
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials and trials >= 1")
    error_value = float(error)
    if not math.isfinite(error_value) or not 0.0 < error_value < 1.0:
        raise ValueError("error must lie in (0, 1)")
    if successes == 0:
        return 0.0
    return float(beta.ppf(error_value, successes, trials - successes + 1))


def one_sided_binomial_upper(
    successes: int, trials: int, error: float
) -> float:
    """Return the exact one-sided Clopper--Pearson upper endpoint."""

    if not isinstance(successes, int) or not isinstance(trials, int):
        raise TypeError("successes and trials must be integers")
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials and trials >= 1")
    error_value = float(error)
    if not math.isfinite(error_value) or not 0.0 < error_value < 1.0:
        raise ValueError("error must lie in (0, 1)")
    if successes == trials:
        return 1.0
    return float(beta.ppf(1.0 - error_value, successes + 1, trials - successes))


def gaussian_joint_mass_radius(
    selected_mass_lower: float,
    runner_mass_upper: float,
    sigma: float,
) -> float:
    """Return the Gaussian event radius for two joint label masses."""

    selected = _validate_probability(selected_mass_lower, "selected_mass_lower")
    runner = _validate_probability(runner_mass_upper, "runner_mass_upper")
    scale = _validate_scale(sigma)
    if selected <= runner:
        return 0.0
    return float(0.5 * scale * (norm.ppf(selected) - norm.ppf(runner)))


def certify_filtered_joint_mass_from_counts(
    counts: Mapping[Label, int],
    proposal_count: int,
    sigma: float,
    *,
    label_universe: Iterable[Label],
    delta: float = 0.001,
    selected_label: Label | None = None,
    selection_independent: bool = False,
) -> JointMassCertificate:
    """Certify a fixed-filter conditional classifier from proposal counts.

    Each count records proposals that both pass the fixed filter and receive
    the corresponding task label. Filtered proposals remain in
    ``proposal_count`` but in no label count.

    With no supplied label, all label masses receive simultaneous two-sided
    Clopper--Pearson bounds and the empirical top label is selected. A supplied
    label may use the sharper independent-selection allocation only when it
    was chosen from a statistically independent proposal batch.
    """

    scale = _validate_scale(sigma)
    if not isinstance(proposal_count, int) or proposal_count < 1:
        raise ValueError("proposal_count must be a positive integer")
    delta_value = float(delta)
    if not math.isfinite(delta_value) or not 0.0 < delta_value < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    labels = tuple(label_universe)
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("label_universe must contain at least two unique labels")
    complete: dict[Label, int] = {}
    for label in labels:
        value = counts.get(label, 0)
        if not isinstance(value, int) or value < 0:
            raise ValueError("every count must be a nonnegative integer")
        complete[label] = value
    if any(label not in labels for label in counts):
        raise ValueError("counts contains a label outside label_universe")
    if sum(complete.values()) > proposal_count:
        raise ValueError("retained label counts cannot exceed proposal_count")

    if selected_label is None:
        selected = max(labels, key=complete.__getitem__)
        per_tail_error = delta_value / (2.0 * len(labels))
        lower = one_sided_binomial_lower(
            complete[selected], proposal_count, per_tail_error
        )
        upper_by_label = {
            label: one_sided_binomial_upper(
                complete[label], proposal_count, per_tail_error
            )
            for label in labels
            if label != selected
        }
        independent = False
    else:
        if selected_label not in complete:
            raise ValueError("selected_label must belong to label_universe")
        if not selection_independent:
            raise ValueError(
                "a supplied selected_label requires selection_independent=True"
            )
        selected = selected_label
        lower = one_sided_binomial_lower(
            complete[selected], proposal_count, delta_value / 2.0
        )
        competitor_error = delta_value / (2.0 * (len(labels) - 1))
        upper_by_label = {
            label: one_sided_binomial_upper(
                complete[label], proposal_count, competitor_error
            )
            for label in labels
            if label != selected
        }
        independent = True

    runner = max(upper_by_label, key=upper_by_label.__getitem__)
    runner_upper = upper_by_label[runner]
    radius = gaussian_joint_mass_radius(lower, runner_upper, scale)
    return JointMassCertificate(
        selected_label=selected,
        runner_up=runner,
        selected_count=complete[selected],
        proposal_count=proposal_count,
        selected_lower=lower,
        runner_upper=runner_upper,
        radius=radius,
        confidence=1.0 - delta_value,
        selection_independent=independent,
    )


def certify_filtered_hybrid_from_counts(
    counts: Mapping[Label, int],
    proposal_count: int,
    sigma: float,
    *,
    label_universe: Iterable[Label],
    selected_label: Label,
    delta: float = 0.001,
    selection_independent: bool = False,
) -> HybridMassCertificate:
    """Certify after independent selection using explicit and complement bounds.

    The error probability is divided equally among the selected-label lower
    bound, the selected-or-rejected lower bound, and the family of explicit
    competitor upper bounds.  The tighter simultaneous competitor bound is
    then used in the Gaussian event comparison.
    """

    scale = _validate_scale(sigma)
    if not isinstance(proposal_count, int) or proposal_count < 1:
        raise ValueError("proposal_count must be a positive integer")
    delta_value = float(delta)
    if not math.isfinite(delta_value) or not 0.0 < delta_value < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    labels = tuple(label_universe)
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("label_universe must contain at least two unique labels")
    if selected_label not in labels:
        raise ValueError("selected_label must belong to label_universe")
    if not selection_independent:
        raise ValueError("hybrid certification requires independent selection")
    complete: dict[Label, int] = {}
    for label in labels:
        value = counts.get(label, 0)
        if not isinstance(value, int) or value < 0:
            raise ValueError("every count must be a nonnegative integer")
        complete[label] = value
    if any(label not in labels for label in counts):
        raise ValueError("counts contains a label outside label_universe")
    retained = sum(complete.values())
    if retained > proposal_count:
        raise ValueError("retained label counts cannot exceed proposal_count")

    family_error = delta_value / 3.0
    selected_lower = one_sided_binomial_lower(
        complete[selected_label], proposal_count, family_error
    )
    competitor_error = family_error / (len(labels) - 1)
    explicit_by_label = {
        label: one_sided_binomial_upper(
            complete[label], proposal_count, competitor_error
        )
        for label in labels
        if label != selected_label
    }
    runner = max(explicit_by_label, key=explicit_by_label.__getitem__)
    explicit_upper = explicit_by_label[runner]

    rejected = proposal_count - retained
    selected_or_rejected_lower = one_sided_binomial_lower(
        complete[selected_label] + rejected,
        proposal_count,
        family_error,
    )
    complement_upper = 1.0 - selected_or_rejected_lower
    runner_upper = min(explicit_upper, complement_upper)
    radius = gaussian_joint_mass_radius(
        selected_lower, runner_upper, scale
    )
    return HybridMassCertificate(
        selected_label=selected_label,
        runner_up=runner,
        selected_count=complete[selected_label],
        proposal_count=proposal_count,
        selected_lower=selected_lower,
        explicit_runner_upper=explicit_upper,
        complement_runner_upper=complement_upper,
        runner_upper=runner_upper,
        radius=radius,
        confidence=1.0 - delta_value,
    )


def conditioned_diameter_pinsker_radius(
    selected_probability_lower: float,
    runner_probability_upper: float,
    sigma: float,
    diameter_upper: float,
) -> float:
    """Return the bounded-support KL--Pinsker classification radius."""

    selected = _validate_probability(
        selected_probability_lower, "selected_probability_lower"
    )
    runner = _validate_probability(
        runner_probability_upper, "runner_probability_upper"
    )
    scale = _validate_scale(sigma)
    diameter = float(diameter_upper)
    if not math.isfinite(diameter) or diameter <= 0.0:
        raise ValueError("diameter_upper must be finite and positive")
    return float(2.0 * scale**2 * max(0.0, selected - runner) / diameter)


def categorical_order_reversal_kl(
    selected_probability_lower: float,
    runner_probability_upper: float,
) -> float:
    """Return the least categorical KL needed to reverse two labels.

    The input probabilities are simultaneous bounds at the anchor.  When the
    selected-label lower bound exceeds the runner-up upper bound, the result is
    the KL distance from those two masses to their common arithmetic mean.
    The runner bound is first tightened by the remaining probability mass.
    Coarsening all remaining outcomes into one category proves that every
    distribution in which the runner catches the selected label must incur at
    least this much KL divergence.
    """

    selected = _validate_probability(
        selected_probability_lower, "selected_probability_lower"
    )
    runner = _validate_probability(
        runner_probability_upper, "runner_probability_upper"
    )
    runner = min(runner, max(0.0, 1.0 - selected))
    if selected <= runner:
        return 0.0
    if runner == 0.0:
        return float(selected * math.log(2.0))
    total = selected + runner
    contrast = (selected - runner) / total
    result = total * (
        contrast * math.atanh(contrast)
        + 0.5 * math.log1p(-(contrast * contrast))
    )
    return float(max(0.0, result))


def conditioned_covariance_kl_radius(
    selected_probability_lower: float,
    runner_probability_upper: float,
    sigma: float,
    covariance_factor_upper: float,
    *,
    certified_region_radius: float = math.inf,
) -> float:
    """Return a conditional-label radius from a uniform covariance bound.

    ``covariance_factor_upper`` certifies
    ``Cov(Q_c) <= covariance_factor_upper * sigma**2 * I`` at every center in
    the stated region.  The region radius is therefore part of the guarantee,
    not a numerical search limit.  A covariance measured only at the anchor
    does not satisfy this contract.
    """

    scale = _validate_scale(sigma)
    factor = float(covariance_factor_upper)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("covariance_factor_upper must be finite and positive")
    region_radius = float(certified_region_radius)
    if math.isnan(region_radius) or region_radius <= 0.0:
        raise ValueError("certified_region_radius must be positive")
    reversal_kl = categorical_order_reversal_kl(
        selected_probability_lower,
        runner_probability_upper,
    )
    if reversal_kl == 0.0:
        return 0.0
    radius = scale * math.sqrt(2.0 * reversal_kl / factor)
    return float(min(radius, region_radius))


def conditioned_diameter_log_odds_radius(
    selected_probability_lower: float,
    runner_probability_upper: float,
    sigma: float,
    diameter_upper: float,
) -> float:
    """Return the bounded-support log-odds classification radius."""

    selected = _validate_probability(
        selected_probability_lower, "selected_probability_lower"
    )
    runner = _validate_probability(
        runner_probability_upper, "runner_probability_upper"
    )
    scale = _validate_scale(sigma)
    diameter = float(diameter_upper)
    if not math.isfinite(diameter) or diameter <= 0.0:
        raise ValueError("diameter_upper must be finite and positive")
    if selected <= runner:
        return 0.0
    if runner == 0.0:
        return math.inf
    return float(scale**2 * math.log(selected / runner) / diameter)
