# Reproduction guide

This repository contains the source, study protocols, tests, and retained tables
used by **The Geometry of Randomized Smoothing on Feasible Sets**. Historical
experiment paths are unchanged so that recorded protocols and checksums remain
meaningful.

## Environment

Use Python 3.11 or newer. A clean installation uses the following commands.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,validated]'
```

The final local verification used NumPy 2.5.0, SciPy 1.18.0, Matplotlib 3.11.0,
mpmath 1.3.0, SymPy 1.14.0, python-flint 0.9.0, and pytest 9.1.1.

This base environment covers the mathematical tests, retained-result audits,
and figure generation. Learned-image reruns additionally require the pinned
AuditVotes checkout, its checkpoint, and the image environment.

```sh
python -m pip install -r requirements-image.txt
```

These pins match the CIFAR-10.1 and CIFAR-10.2 confirmations. The earlier
CIFAR-10 evaluation used PyTorch 2.3.1 and torchvision 0.18.1, as recorded in
its metadata.

The learned-controller rerun uses a separate Python 3.10.12 environment with
PyTorch 2.2.2, Safety-Gymnasium 1.0.0, MuJoCo 2.3.3, Gymnasium 0.28.1,
Gymnasium-Robotics 1.2.2, Joblib 1.5.3, and SafePO 1.0.1 at commit
`8d0fa763b5e9091db8bed502b7efe76f81b1d263`. Complete versions and source
checksums are recorded in the controller training protocol. Create that
environment separately because the core package targets Python 3.11 or newer.

```sh
python3.10 -m venv .venv-controller
. .venv-controller/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-controller.txt
```

Controller commands use `PYTHONPATH=.:src` and do not require installing the
core package into this Python 3.10 environment.

PyTorch and Joblib checkpoints can execute serialized code. The experiment
loaders verify the declared file checksums before deserialization. Use only the
pinned external checkpoint and the controller files distributed here.

`ANONYMIZATION_PROVENANCE.json` records every JSON file whose
machine-specific project root was replaced by a project-relative path. It gives
the original and archived SHA-256 values and every changed JSON pointer. No
numeric result, model parameter, count, seed, or decision field was changed.
The terminal controller checkpoints, normalizers, progress tables, final
CIFAR-10.2 row file, and final run manifest are exact copies.
`MANIFEST.sha256` authenticates every file distributed in this archive.

## Fast verification

Run the fast verification from the archive root.

```sh
PYTHONPATH=.:src python -m pytest -q
```

The expected result for this archive is `248 passed, 1 skipped`. The skipped
test binds the original machine-path-bearing controller configuration bytes.
The other controller integration tests remain active.

The archive manifest records the exact files used for final verification.

Recompute the exact interval witness with the following command.

```sh
PYTHONPATH=src python experiments/certify_local_lshape_witness.py \
  --output outputs/local_lshape_interval_certificate_recomputed.json
```

Recompute the nonconvex two-band covariance certificate with the following
command.

```sh
PYTHONPATH=src python experiments/certify_two_band_covariance_radius.py \
  --output outputs/two_band_covariance_radius_certificate_recomputed.json
```

Regenerate the paper figures from the retained tables with the following
command.

```sh
PYTHONPATH=.:src python experiments/plot_iclr2027_filtered_smoothing.py \
  --output reproduced_figures
PYTHONPATH=.:src python experiments/plot_iclr2027_covariance_mechanism_holdout.py \
  --rows outputs/iclr2027_covariance_mechanism_holdout_20260912/factor_rows.csv \
  --output reproduced_figures/covariance_mechanism_holdout
PYTHONPATH=.:src python experiments/plot_iclr2027_conditional_renyi.py \
  --output-directory reproduced_figures
```

## Released-model evaluation

The source audit is pinned to AuditVotes commit
`52b6a0db53947c815884ad62927d6af6dc584705`. The upstream repository is
`https://github.com/Yuni-Lai/AuditVotes-Certified-Robustness`. It has no license
file, so its source and checkpoint are not redistributed. The released
CIFAR-10 ResNet-110 checkpoint has SHA-256
`420333fe0380cc437218c9b67c20bacf687932957da47340f900d4af5e05bd7c`.
The audited claim is Theorem 2 in Section 5.2, with its proof in Appendix A.2,
of the retained AuditVotes v3 manuscript. That PDF has SHA-256
`27ed62c8de1518c71de0fc585ce126796fe49aa243fd97db279290618f3505fe`.

### CIFAR-10.2 projected-band confirmation

The filter, random streams, analysis, and full 2,000-image census were specified
before CIFAR-10.2 access. The protocol, runner, figure source, run manifest,
per-image rows, and summary are included. The archived summary differs only in
project-path fields, as recorded in `ANONYMIZATION_PROVENANCE.json`.

Recompute every plotted value from the retained rows and regenerate both
figures with the following command.

```sh
PYTHONPATH=.:src python experiments/plot_iclr2027_global_band_cifar102_confirmation.py \
  --summary outputs/iclr2027_global_band_cifar102_confirmation_v2/summary.json \
  --rows outputs/iclr2027_global_band_cifar102_confirmation_v2/per_image.jsonl \
  --manifest outputs/iclr2027_global_band_cifar102_confirmation_v2/manifest.json \
  --output-directory reproduced_figures
```

The loader rejects partial, reordered, incomplete, or hash-mismatched rows. At
radius `0.2`, it recovers correct-certified counts of `829`, `725`, and `948`
for the original forward-KL, joint-mass, and unfiltered calculations. It also recovers
`329` correctly selected filtered images whose simultaneous covariance lower
radius exceeds the finite joint-mass upper radius by at least `0.01`. The mean
estimation retention is `0.427570395`.

The original V1 confirmation protocol and its pre-outcome loader failure are
included. The loader rejected additional metadata arrays in the official
dataset before loading the model or generating image outcomes. V2 corrected
that schema check. These records are distinct from the earlier training-side
filter development, whose records are also retained.

### Conditional Rényi reanalysis

The subsequent reanalysis uses exactly the original CIFAR-10.2 conditional
probability bounds. It makes no new model calls and is not an independent
confirmation. The categorical bound is from Li et al., NeurIPS 2019, Lemma 1.
The paper transfers it through the conditional covariance bound and requires
control out to `order * radius` when that bound is local.

```sh
PYTHONPATH=.:src python experiments/iclr2027_conditional_renyi_reanalysis.py \
  --output-directory outputs/recomputed_conditional_renyi
PYTHONPATH=.:src python experiments/plot_iclr2027_conditional_renyi.py \
  --output-directory reproduced_figures
```

At radius `0.2`, reverse KL and Rényi certify `888` and `892` correct images.
At radius `0.5`, they certify `329` and `381`, while the original forward-KL
and joint-mass calculations certify zero. Unfiltered smoothing certifies
`529`. The count-level matched-call comparison certifies `522` with the same
`94,064,290` model calls as the filtered streams. It uses one seeded
hypergeometric subset of each unfiltered count vector and the original
per-tail error probability. Its separate confidence family has error at most
`0.0003125`. It is not a new network inference or runtime measurement.
The combined error bound for the original and matched-call families is
`0.0013125`; the Rényi analysis alone reuses the original `0.001` family.

### Outward count-to-radius validation

The full-count CIFAR-10.2 conditional Rényi, reverse-KL, joint-mass, and
unfiltered calculations have also been recomputed with Arb real-ball
arithmetic at 128 bits. This step uses the original integer counts. It does
not validate the floating-point noise sampler, filter, neural inference,
matched-call baseline, or other studies.

```sh
PYTHONPATH=.:src python experiments/iclr2027_validate_cifar102_certificates.py \
  --workers 6 --output-directory outputs/recomputed_validated_cifar102
```

The exact tail allocation is `1/64000000`. SciPy supplies a candidate
binomial endpoint, which is returned only after a rigorous tail inequality
is proved. The recurrence bounds uncomputed terms by a geometric series.
Gaussian quantiles and divergence expressions are rounded outward.
All tabulated counts at 0.2, 0.3, and 0.5 remain unchanged, as do the 877
separation statements. The largest conditional endpoint change is less than
`3.51e-12`. The tests include exact-rational binomial sums, endpoint
inequalities, outward rounding, a large-count regression case, and
independent higher-precision radius checks. The summary records the source
hashes, order set, precision, and arithmetic-library version. Elapsed time
will differ on reproduction.

The `validated` extra is optional for the other experiments. Without it,
the validated-numerics tests are skipped and the command above is unavailable.

### Applying the conditional certificate

For a new application, obtain conditional probability bounds using an
independent estimation batch, then supply a proved covariance bound.

```python
from feasible_robustness.conditional_renyi import conditioned_covariance_renyi_radius

certificate = conditioned_covariance_renyi_radius(
    selected_probability_lower=0.90,
    runner_probability_upper=0.08,
    sigma=0.25,
    covariance_factor_upper=1.0,
    certified_region_radius=float("inf"),
)
print(certificate.radius, certificate.order)
```

The covariance factor must bound `Cov(Q_c) / sigma**2` at every center in the
declared region. A sample covariance at one center does not meet this premise.
The implementation evaluates ordinary floating-point expressions, not
outward-rounded interval certificates. The exact-rational two-band calculation
in the preceding section remains a separate forward-KL verification.

### Repeating the original CIFAR-10.2 inference

The official CIFAR-10.2 repository is pinned at commit
`521467a52ad494b78ff9c70051099c0d9ba74d40`. The official test NPZ has
SHA-256
`e4fd6462bd5141ed293acd52ea570952f65d1bfd4ec9b068f87742f75a7a632e`.
Neither the dataset nor the AuditVotes checkpoint is redistributed. After
obtaining both pinned repositories and the released checkpoint, run a fresh
reproduction with the following command.

```sh
PYTHONPATH=.:src python experiments/iclr2027_global_band_cifar102_confirmation.py \
  --protocol outputs/ICLR2027_GLOBAL_BAND_CIFAR102_PROTOCOL_V2_20260920.anonymous_rerun.json \
  --development-result outputs/iclr2027_global_band_precision_replication_v1.json \
  --cifar102-repo-root PATH_TO_PINNED_CIFAR102 \
  --dataset PATH_TO_CIFAR102_TEST_NPZ \
  --auditvotes-repo-root PATH_TO_PINNED_AUDITVOTES \
  --checkpoint PATH_TO_RELEASED_RESNET110 \
  --output outputs/recomputed_cifar102_confirmation
```

The portable protocol uses project-relative paths and updates the checksum of
the path-normalized development prerequisite. Its numeric settings are
unchanged. CUDA execution is not expected to be byte identical across hardware.

### CIFAR-10 full evaluation

The retained full-test records can be summarized without reevaluating the
network.

```sh
PYTHONPATH=.:src python experiments/summarize_iclr2027_auditvotes_recertification.py \
  --rows outputs/auditvotes_recertification_full_20260906/per_image.jsonl \
  --output outputs/auditvotes_recertification_full_20260906/recomputed_summary.json
```

The retained rows store the explicit-runner Clopper-Pearson endpoint rather
than the full ten-class count vector. At fixed proposal count and error level,
that endpoint is strictly increasing in the integer count. The postprocessor
recovers the unique maximum competitor count, checks the reconstructed endpoint
to numerical tolerance, and recomputes the combined bounds in Algorithm 1. It
returns a simultaneous positive-radius fraction of `0.8875`, median radius
`0.3330324`, and correct fractions `0.5153`, `0.3202`, and `0.1394` above radii
`0.3`, `0.5`, and `0.7`.

The network run itself requires the upstream CUDA environment and checkpoint.

```sh
PYTHONPATH=.:src python experiments/iclr2027_auditvotes_recertification.py \
  --repo-root PATH_TO_PINNED_AUDITVOTES \
  --checkpoint PATH_TO_RELEASED_RESNET110 \
  --output outputs/recomputed_auditvotes \
  --max-images 10000 --n0 100 --n 10000 --batch-size 1024 \
  --sigma 0.25 --confidence-threshold 0.9 --alpha 0.001 \
  --seed 26090501
```

This run used PyTorch 2.3.1, torchvision 0.18.1, and one NVIDIA RTX 4070 Ti.
The per-image output is included in the archive.

The confidence-threshold protocol was specified before execution and is included
at `_ICLR_2027__Feasibility_Breaks_Smoothing/research/AUDITVOTES_THRESHOLD_SELECTION_PROTOCOL_V1_20260906.json`.
Its retained per-image and summary outputs are included.
The declared objective averages explicit-joint certified accuracy over radii
`0.25`, `0.5`, and `0.75`. The first 1,000 ordered test indices form the
development set, and the remaining 9,000 form the untouched confirmation set.
Threshold zero is selected on development and has the highest value on
confirmation among all nine declared thresholds.

### CIFAR-10.1 full evaluation

The CIFAR-10.1 v4 protocol was specified before model inference and is included
at `_ICLR_2027__Feasibility_Breaks_Smoothing/research/AUDITVOTES_CIFAR101_FIXED_COHORT_PROTOCOL_V1_20260909.json`.
The dataset repository is pinned at commit
`d9982abb0bfc4846b8d13a11e66b887d946205d0`. The v4 image and label files have
SHA-256 digests
`5aaedc268df8f4cffca2ac0cf9e608ae1897a8cddb630c7a850a2ea8f6558e75`
and
`1f9544300ded199539b6662364ce7fe6c2e1ea9f87408f06db4b1a6184cffb18`.
The dataset files are not redistributed.

All 2,021 images use the same checkpoint, filter, noise scale, proposal counts,
and no-clipping proposal law as the CIFAR-10 evaluation. The per-image records,
summary, and evaluation manifest are stored under
`outputs/auditvotes_cifar101_v4_fixed_cohort_20260909/evaluation`.
The simultaneous joint calculation returns positive radius on `0.8362` of the
images, median radius `0.2122`, and correct fractions `0.3449`, `0.1752`, and
`0.0643` above radii `0.3`, `0.5`, and `0.7`. The filtered selected-label
accuracy is `0.6319`, the unfiltered selected-label accuracy is `0.6502`, and
the mean retained proposal fraction is `0.351061`. The run took 1,636 seconds
with PyTorch 2.2.2, torchvision 0.17.2, and one NVIDIA RTX 4070 Ti.

The full evaluation can be rerun after obtaining the pinned repository,
checkpoint, and official CIFAR-10.1 files.

```sh
PYTHONPATH=.:src python experiments/iclr2027_auditvotes_cifar101_fixed_cohort.py evaluate \
  --protocol _ICLR_2027__Feasibility_Breaks_Smoothing/research/AUDITVOTES_CIFAR101_FIXED_COHORT_PROTOCOL_V1_20260909.json \
  --repo-root PATH_TO_PINNED_AUDITVOTES \
  --checkpoint PATH_TO_RELEASED_RESNET110 \
  --images PATH_TO_CIFAR101_V4_IMAGES \
  --labels PATH_TO_CIFAR101_V4_LABELS \
  --output outputs/recomputed_cifar101_evaluation
```

## Fixed CIFAR-10.1 cohort confirmation

The same protocol specifies a hash-ranked cohort of 128 images without
using model output. It fixes all random streams, the alternative-label rule,
four search restarts, 150 optimization steps, 256 proposals per gradient
estimate, and three surrogate temperatures. The best and final centers from
each restart give eight candidates per positive-radius image. Each candidate
receives a disjoint fresh 50,000-proposal screen, after which the center with
the largest target margin is retained.
The released calculation assigns positive radius to 107 cohort members. The
other 21 members remain in the primary denominator. Twelve pairs pass the
screen.

Confirmation uses 250,000 fresh proposals at each endpoint. The error
allocation reserves nine winner-versus-competitor tests at each endpoint and
one nominal conditional-probability lower bound for every cohort member. This
gives 2,432 reserved inferences and familywise error at most `0.001`. All twelve
screen-positive pairs verify opposite population labels with shifts below
fresh lower bounds on the substituted radii. The fixed-cohort attack yield is
`12/128`. Nine confirmations occur among the 80 images with a correct original
filtered label and positive released radius. The 95 unsuccessful screens do
not establish robustness.

The search summary, attempt records, confirmation file, and confirmation rows
are retained under
`outputs/auditvotes_cifar101_v4_fixed_cohort_20260909`.
Search and confirmation took 8,852 and 466 seconds on the same GPU. Candidate
image tensors are not redistributed. Reproducing the confirmation from model
execution therefore requires rerunning the search with the pinned external
assets.

These results describe the fixed cohort and the fixed search procedure. They
do not estimate a population frequency for other models, filters, datasets,
noise scales, or attacks.

## Earlier output-selected boundary confirmation

The earlier search protocol and preconfirmation record are included at
`_ICLR_2027__Feasibility_Breaks_Smoothing/research/AUDITVOTES_BOUNDARY_SEARCH_PROTOCOL_V1_20260906.json`
and
`_ICLR_2027__Feasibility_Breaks_Smoothing/research/AUDITVOTES_BOUNDARY_CONFIRMATION_FREEZE_V1_20260907.json`.
The selected order and retained count outputs are included. Candidate
image tensors are not redistributed.

The confirmation used one million fresh proposals at each endpoint. The raw
output and retained analysis are included. It verifies four population
substitution violations with familywise error at
most `0.001`. Candidate selection used opened full-test output. This result is
a separate existence check and does not estimate a population frequency.

## Reach-avoid reanalysis

The source, test, protocol, and confirmation metadata are recorded in
`_ICLR_2027__Feasibility_Breaks_Smoothing/research/REACH_AVOID_CONFIRMATION_RESULTS_V2_20260831.json`.
The original opened population can be rerun under a new output name with the
following command.

```sh
PYTHONPATH=.:src python experiments/iclr2027_multistep_reach_avoid_benchmark.py \
  --protocol _ICLR_2027__Feasibility_Breaks_Smoothing/research/REACH_AVOID_CONTROLLER_PROTOCOL_V2_20260830.json \
  --freeze-manifest _ICLR_2027__Feasibility_Breaks_Smoothing/research/REACH_AVOID_CONFIRMATION_FREEZE_V2_1_20260831.json \
  --splits confirmation \
  --output-stem outputs/reproduced_reach_avoid_confirmation_v2
```

Then reproduce the analytic joint-mass reanalysis with the following command.

```sh
PYTHONPATH=.:src python experiments/iclr2027_reach_avoid_joint_mass_recertification.py \
  --protocol _ICLR_2027__Feasibility_Breaks_Smoothing/research/REACH_AVOID_CONTROLLER_PROTOCOL_V2_20260830.json \
  --output outputs/recomputed_reach_avoid_joint_mass.json
```

## Held-out covariance-mechanism replication

The protocol and result manifest are included at
`_ICLR_2027__Feasibility_Breaks_Smoothing/research/COVARIANCE_MECHANISM_HOLDOUT_PROTOCOL_V1_20260912.json`
and
`_ICLR_2027__Feasibility_Breaks_Smoothing/research/COVARIANCE_MECHANISM_HOLDOUT_RESULTS_V1_20260912.json`.
It fixes 128 independently seeded coherent product-box geometries before the
new geometry and direction seeds are instantiated. The retained factor rows
and summary are included.

The archived protocol binds the runner revision used for this study. The
retained rows are authenticated by `MANIFEST.sha256`, and the tests recompute
the reported statistics from those rows. Regenerate the figure with the
following command.

```sh
PYTHONPATH=.:src python experiments/plot_iclr2027_covariance_mechanism_holdout.py \
  --rows outputs/iclr2027_covariance_mechanism_holdout_20260912/factor_rows.csv \
  --output reproduced_figures/covariance_mechanism_holdout
```

The path-integrated identity is a numerical implementation check. The study is
a targeted replication in a family chosen from earlier results. It does not
estimate failure prevalence over arbitrary feasible sets, and a nominal sample
covariance is not a uniform covariance certificate.

## Certified nonconvex covariance radius

The two-band verifier certifies the strict ordering among the joint-mass,
covariance-controlled, exact-boundary, and substituted conditional radii using
exact fractions and outward rational intervals. The support is nonconvex and
has infinite diameter. Its first coordinate remains Gaussian, while
Popoviciu's inequality bounds the conditioned second-coordinate variance by
one at every center. The output is stored at
`outputs/two_band_covariance_radius_certificate.json`.

## Capped causal selector

`src/feasible_robustness/ccfs.py` implements the correlated Gaussian proposal
block and its exact comparison multiplier. Its older trajectory helper is
retained byte for byte because development protocols record its source hash.
`src/feasible_robustness/totalized_ccfs.py` inspects a finite proposal block,
then a finite fallback library, and returns an explicit abstention without a
physical action when both are exhausted.
`src/feasible_robustness/trajectory_certificate.py` supplies the one-sided
binomial lower bound, count-to-radius calculation, numerically hardened
`trajectory_shift_distance`, and a pre-block energy limiter.
`src/feasible_robustness/trajectory_rollout.py` runs a finite schedule, spawns
separate proposal, attack, and transition random streams, commits and limits
each shift before drawing its proposal block, records public and private data
separately, and never calls the transition after abstention. Import these
modules directly. A certified adaptive comparison requires a uniform energy
bound over every reachable history and attack seed. An energy computed on one
observed trace is only a diagnostic.

## Learned-controller trajectory study

The supplement includes all three terminal PPO-Lagrangian actors, their
normalizers, and 500-epoch progress tables. Each
configuration differs from its original only in the training-output path.
`ANONYMIZATION_PROVENANCE.json` records the original and archived
configuration hashes.

Reaudit the retained 3,072 trajectory records with the following command.

```sh
PYTHONPATH=.:src python experiments/iclr2027_goal2_safe_trajectory_audit.py \
  --protocol _ICLR_2027__Feasibility_Breaks_Smoothing/research/GOAL2_SAFE_TRAJECTORY_CONFIRMATION_V1_20260919.json \
  --integrity _ICLR_2027__Feasibility_Breaks_Smoothing/research/GOAL2_SAFE_TRAJECTORY_POSTRUN_INTEGRITY_V1_20260919.json \
  --result outputs/iclr2027_goal2_safe_trajectory_confirmation_v1.json \
  --output outputs/recomputed_goal2_safe_trajectory_audit.json
```

The expected report has status `strict_postrun_audit_passed`, exactly 3,072
records, six nominal controller-method arms, and
`tested_budget_within_certificate=true` for every arm.

The integrity file at the command path is an anonymous derivative. It updates
only the training-protocol hash after path normalization and adds a disclosure
field. The exact original integrity record is retained beside it with the
suffix `.original.json`. The original training protocol and controller
configuration byte sequences are not redistributed because their path strings
identify the execution account. The archive therefore supports a scientifically
equivalent rerun rather than a byte-for-byte recreation of machine-specific
metadata.

A fresh simulator run can use the exact model weights and a path-normalized
protocol.

```sh
PYTHONPATH=.:src python experiments/iclr2027_goal2_safe_trajectory_confirmation.py \
  --protocol _ICLR_2027__Feasibility_Breaks_Smoothing/research/GOAL2_SAFE_TRAJECTORY_CONFIRMATION_V1_20260919.anonymous_rerun.json \
  --output outputs/recomputed_goal2_safe_trajectory_confirmation.json
```

This run requires Python 3.10.12, Safety-Gymnasium 1.0.0, MuJoCo 2.3.3, and a
PyTorch 2.2.2 environment compatible with the retained checkpoints. The new
result is a scientifically equivalent reproduction rather than a
byte-identical replay of the original protocol.

## Appendix study

The random product-box command and study metadata are in
`outputs/e_random_box_union_ensemble_holdout/HOLDOUT_MANIFEST.json`.

## Claim boundary

The primary learned-model confirmation uses a fixed output-independent cohort
of 128 CIFAR-10.1 images. It verifies twelve population substitution violations
under one fixed search procedure. Its `12/128` value is a fixed-cohort attack
yield and does not estimate a population frequency. The earlier
output-selected confirmation establishes four additional violations among six
selected pairs. It is an existence check.

The reach-avoid study is a specified synthetic controller benchmark with layout
as the independent unit. It is not a learned-policy, observation-space,
continuous-time, physical-system, or real-world frequency claim. Soundness
of the fixed rows follows from the paper's theorems. Empirical absence of an
attack is only a diagnostic.
