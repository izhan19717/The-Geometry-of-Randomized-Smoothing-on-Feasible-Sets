"""Symbolic proof expressions used in the theory notes.

SymPy is the canonical symbolic algebra library for this project. These
helpers keep algebraic identities in executable form and exportable to LaTeX.
They do not replace formal proof assistants; they make the paper algebra
auditable and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class SmoothingSymbols:
    sigma: sp.Symbol
    p_a: sp.Symbol
    p_b: sp.Symbol
    tv: sp.Symbol
    margin: sp.Symbol
    norm_ab: sp.Symbol
    z_a: sp.Symbol
    z_b: sp.Symbol


def symbols() -> SmoothingSymbols:
    """Return common positive symbols for smoothing derivations."""

    sigma = sp.Symbol("sigma", positive=True)
    p_a = sp.Symbol("p_A", positive=True)
    p_b = sp.Symbol("p_B", positive=True)
    tv = sp.Symbol("tau", nonnegative=True)
    margin = sp.Symbol("Delta", positive=True)
    norm_ab = sp.Symbol("d_ab", nonnegative=True)
    z_a = sp.Symbol("Z_a", positive=True)
    z_b = sp.Symbol("Z_b", positive=True)
    return SmoothingSymbols(sigma, p_a, p_b, tv, margin, norm_ab, z_a, z_b)


def normal_quantile(name: str) -> sp.Function:
    """Symbolic placeholder for the standard-normal inverse CDF."""

    return sp.Function(name)


def cohen_radius_expr() -> sp.Expr:
    """Cohen Gaussian smoothing radius expression."""

    s = symbols()
    phi_inv = normal_quantile("Phi^{-1}")
    return sp.Rational(1, 2) * s.sigma * (phi_inv(s.p_a) - phi_inv(s.p_b))


def tv_margin_condition_expr() -> sp.StrictLessThan:
    """Condition ensuring a top-two probability margin survives TV drift."""

    s = symbols()
    return sp.StrictLessThan(2 * s.tv, s.margin)


def pinsker_tv_bound_expr() -> sp.Expr:
    """Pinsker upper bound on total variation from KL divergence."""

    kl = sp.Symbol("KL", nonnegative=True)
    return sp.sqrt(kl / 2)


def truncated_gaussian_kl_identity_expr() -> sp.Expr:
    """Symbolic KL identity for two truncated Gaussians on the same set.

    The returned expression is the KL divergence after bounding the expectation
    term by ``d_ab^2 / (2 sigma^2)`` and leaving the normalizer ratio explicit.
    The exact paper statement keeps the expectation form before applying this
    conservative bound.
    """

    s = symbols()
    return s.norm_ab**2 / (2 * s.sigma**2) + sp.log(s.z_b / s.z_a)


def dirichlet_kl_expr(k: int = 3) -> sp.Expr:
    """Closed-form KL divergence between two k-dimensional Dirichlet laws."""

    if k < 2:
        raise ValueError("k must be at least 2")
    alpha = sp.symbols(f"alpha_1:{k + 1}", positive=True)
    beta = sp.symbols(f"beta_1:{k + 1}", positive=True)
    alpha_0 = sum(alpha)
    beta_0 = sum(beta)
    expr = sp.loggamma(alpha_0) - sp.loggamma(beta_0)
    expr -= sum(sp.loggamma(a) for a in alpha)
    expr += sum(sp.loggamma(b) for b in beta)
    expr += sum((a - b) * (sp.digamma(a) - sp.digamma(alpha_0)) for a, b in zip(alpha, beta))
    return sp.simplify(expr)


def latex_summary() -> dict[str, str]:
    """Return LaTeX snippets for core symbolic expressions."""

    return {
        "cohen_radius": sp.latex(cohen_radius_expr()),
        "tv_margin_condition": sp.latex(tv_margin_condition_expr()),
        "pinsker_tv_bound": sp.latex(pinsker_tv_bound_expr()),
        "truncated_gaussian_kl_bound": sp.latex(truncated_gaussian_kl_identity_expr()),
        "dirichlet_kl_3d": sp.latex(dirichlet_kl_expr(3)),
    }

