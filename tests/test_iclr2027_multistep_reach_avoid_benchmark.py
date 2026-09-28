import json
import math

import numpy as np

from experiments.iclr2027_multistep_reach_avoid_benchmark import (
    METHODS,
    Stage,
    audit_state,
    build_payload,
    coarse_partition,
    controller_center,
    frozen_conditioned_upper_probability,
    generate_layouts,
    load_protocol,
    moving_conditioned_kl,
    moving_upper_probability,
    normal_interval_logmass,
    partition_weights,
    perturbation_centers,
    point_is_feasible,
    projected_rectangle_upper_probability,
    projection_rectangles,
    rectangle_logmass,
    refined_partition,
    run_episode,
    sample_kernel,
    support_stats,
    wilson_interval,
)


def protocol_payload():
    return {
        "protocol_version": 1,
        "splits": {
            "development": {
                "layout_seed": 27182818,
                "episode_seed": 14142135,
                "layout_count": 2,
                "episodes_per_layout": 2,
            },
            "validation": {
                "layout_seed": 31415926,
                "episode_seed": 17320508,
                "layout_count": 2,
                "episodes_per_layout": 2,
            },
            "confirmation": {
                "layout_seed": 16180339,
                "episode_seed": 22360679,
                "layout_count": 2,
                "episodes_per_layout": 2,
            },
        },
        "bootstrap_seed": 24494897,
        "bootstrap_resamples": 200,
        "environment": {
            "horizon": 7,
            "stage_spacing": 1.0,
            "stage_box_half_width_x": 0.38,
            "stage_box_y_bounds": [-1.35, 1.35],
            "repair_clearance": 0.025,
        },
        "controller": {"name": "test controller"},
        "explicit_nonclaims": ["test protocol"],
    }


def test_protocol_loader_accepts_v1_and_disclosed_v2(tmp_path):
    for version in (1, 2):
        payload = protocol_payload()
        payload["protocol_version"] = version
        path = tmp_path / f"protocol_v{version}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert load_protocol(path)["protocol_version"] == version


def test_layout_generation_is_deterministic_nonsymmetric_and_feasible():
    protocol = protocol_payload()
    first = generate_layouts("development", 27182818, 3, protocol)
    second = generate_layouts("development", 27182818, 3, protocol)
    assert first == second
    centers = [stage.obstacle_center_y for layout in first for stage in layout.stages]
    assert any(abs(value) > 0.05 for value in centers)
    for layout in first:
        state = layout.start
        for stage in layout.stages:
            center = controller_center(state, layout, stage)
            assert point_is_feasible(center, stage)
            state = center


def test_log_interval_arithmetic_stays_finite_in_both_tails():
    left_tail = normal_interval_logmass(-10.0, -9.0, 0.0, 1.0)
    right_tail = normal_interval_logmass(9.0, 10.0, 0.0, 1.0)
    assert math.isfinite(left_tail)
    assert math.isfinite(right_tail)
    assert np.isclose(left_tail, right_tail, atol=2e-13, rtol=0.0)


def test_both_partitions_reconstruct_the_same_support_mass():
    protocol = protocol_payload()
    layout = generate_layouts("development", 17, 1, protocol)[0]
    for stage in layout.stages:
        anchor = controller_center(layout.start, layout, stage)
        support_logmass = math.log(support_stats(stage, anchor).mass)
        for partition in (coarse_partition(stage), refined_partition(stage)):
            component_logmass = np.asarray(
                [rectangle_logmass(rectangle, anchor, stage.sigma) for rectangle in partition]
            )
            maximum = float(np.max(component_logmass))
            union_logmass = maximum + math.log(
                float(np.sum(np.exp(component_logmass - maximum)))
            )
            assert np.isclose(union_logmass, support_logmass, atol=5e-12, rtol=5e-11)
            weights = partition_weights(stage, anchor, partition)
            assert np.all(weights > 0.0)
            assert np.isclose(np.sum(weights), 1.0, atol=3e-15, rtol=0.0)


def test_fixed_conditioned_partitions_match_whole_law_at_the_anchor():
    protocol = protocol_payload()
    layout = generate_layouts("development", 19, 1, protocol)[0]
    state = layout.start
    for stage in layout.stages:
        anchor = controller_center(state, layout, stage)
        whole = moving_upper_probability(stage, anchor)
        for partition in (coarse_partition(stage), refined_partition(stage)):
            weights = partition_weights(stage, anchor, partition)
            fixed = frozen_conditioned_upper_probability(
                stage, anchor, partition, weights
            )
            assert np.isclose(fixed, whole, atol=2e-13, rtol=2e-12)
        state = anchor


def test_projection_audit_uses_clipped_not_conditioned_gaussian_law():
    protocol = protocol_payload()
    layout = generate_layouts("development", 21, 1, protocol)[0]
    stage = layout.stages[0]
    center = controller_center(layout.start, layout, stage)
    threshold = stage.obstacle_center_y
    left, right, bottom, top = projection_rectangles(stage)
    expected_tail = float(
        0.5 * math.erfc((threshold - float(center[1])) / (math.sqrt(2.0) * stage.sigma))
    )
    assert np.isclose(
        projected_rectangle_upper_probability(
            left, center, stage.sigma, threshold
        ),
        expected_tail,
        atol=2e-15,
        rtol=0.0,
    )
    assert np.isclose(
        projected_rectangle_upper_probability(
            right, center, stage.sigma, threshold
        ),
        expected_tail,
        atol=2e-15,
        rtol=0.0,
    )
    assert projected_rectangle_upper_probability(
        bottom, center, stage.sigma, threshold
    ) == 0.0
    assert projected_rectangle_upper_probability(
        top, center, stage.sigma, threshold
    ) == 1.0


def test_every_deployed_sampler_returns_a_feasible_waypoint():
    protocol = protocol_payload()
    layout = generate_layouts("development", 23, 1, protocol)[0]
    state = layout.start
    for stage_index, stage in enumerate(layout.stages):
        anchor = controller_center(state, layout, stage)
        for method_index, method in enumerate(METHODS):
            rng = np.random.default_rng([stage_index, method_index, 29])
            for _ in range(30):
                sample, cost = sample_kernel(method, stage, anchor, rng)
                assert point_is_feasible(sample, stage)
                direct_methods = {
                    "native_controller",
                    "fixed_uniform_conditioned_coarse4",
                    "anchor_frozen_conditioned_coarse4",
                    "anchor_frozen_conditioned_refined8",
                }
                minimum = 0 if method in direct_methods else 1
                assert cost.actual_gaussian_proposals >= minimum
                assert cost.naive_rejection_equivalent_proposals >= float(minimum)
        state = anchor


def test_fixed_mixture_and_convex_full_kl_ratios_do_not_exceed_one():
    protocol = protocol_payload()
    layout = generate_layouts("development", 31, 1, protocol)[0]
    stage = layout.stages[0]
    anchor = controller_center(layout.start, layout, stage)
    rows = {row["method"]: row for row in audit_state(layout, stage, anchor)}
    for method in (
        "anchor_frozen_conditioned_coarse4",
        "anchor_frozen_conditioned_refined8",
        "convex_box_conditioned_control",
    ):
        assert rows[method]["maximum_full_kl_ratio_on_declared_grid"] <= 1.0 + 1e-9
    for second in perturbation_centers(stage, anchor):
        assert moving_conditioned_kl(stage, anchor, second) >= 0.0


def test_episode_randomness_is_reproducible_apart_from_runtime():
    protocol = protocol_payload()
    layout = generate_layouts("development", 37, 1, protocol)[0]
    first = run_episode(layout, 0, 41, METHODS[3])
    second = run_episode(layout, 0, 41, METHODS[3])
    first.pop("runtime_seconds")
    first.pop("runtime_ms")
    second.pop("runtime_seconds")
    second.pop("runtime_ms")
    assert first == second
    assert first["forbidden_waypoint_count"] == 0


def test_small_payload_is_deterministic_and_clustered_by_layout(tmp_path):
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol_payload()), encoding="utf-8")
    first, _ = build_payload(protocol_path, ["development"])
    second, _ = build_payload(protocol_path, ["development"])
    assert (
        first["deterministic_result_sha256_excluding_runtime"]
        == second["deterministic_result_sha256_excluding_runtime"]
    )
    assert all(
        row["cluster_unit"] == "layout" and row["independent_layout_count"] == 2
        for row in first["episode_summary"]
    )


def test_wilson_interval_is_non_degenerate_at_boundary_counts():
    lower_all, upper_all = wilson_interval(64, 64)
    lower_none, upper_none = wilson_interval(0, 64)
    assert 0.94 < lower_all < 1.0
    assert upper_all == 1.0
    assert lower_none == 0.0
    assert 0.0 < upper_none < 0.06
