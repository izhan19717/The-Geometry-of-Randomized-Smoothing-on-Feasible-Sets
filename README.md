# The Geometry of Randomized Smoothing on Feasible Sets

Research code and retained experimental results accompanying:

**The Geometry of Randomized Smoothing on Feasible Sets**  
Syed Izhan Khilji, Alireza Furutanpey, and Schahram Dustdar  
arXiv preprint, 2026

**[Read the paper](https://arxiv.org/abs/2609.39497)** · **[Reproduction guide](docs/REPRODUCING.md)** · **[Citation](#citation)**

## Overview

This repository studies robustness certificates for Gaussian randomized smoothing when a fixed feasibility or confidence rule rejects some noisy proposals.

Ordinary smoothing controls the Gaussian probability of a label event. With filtering, the reported label probability is the joint retention-and-label probability divided by the probability of retention. Because that denominator changes with the Gaussian center, substituting conditional label probabilities into the ordinary Gaussian certificate can produce an invalid radius.

The repository provides:

- A **joint-mass certificate** for the unchanged filtered predictor, using joint retention-and-label counts.
- A **conditional Rényi certificate** for filters satisfying a proved covariance bound.
- **Outward numerical calculations** for binomial bounds and certified radii.
- An **exact nonconvex witness**, retained image-classification analyses, and finite-horizon trajectory comparisons.

<p align="center">
  <img src="docs/figures/certificate_map.svg" alt="Gaussian proposals pass through a filter. Joint event masses support a Gaussian certificate; conditional votes require a proved geometric condition." width="100%">
</p>

## Certificate assumptions

| Approach | Quantities used | Applicability |
| --- | --- | --- |
| Joint-mass certificate | Joint retention-and-label counts | Certifies the filtered predictor under the method's stated assumptions |
| Conditional Rényi certificate | Conditional label counts and a proved covariance bound | Requires the filter to satisfy the stated geometric condition |
| Ordinary Gaussian formula applied to conditional votes | Conditional label probabilities alone | Not valid in general; the repository includes counterexamples |

A fixed filter does not imply a constant retention probability: moving the Gaussian center can change how often proposals are retained. The paper and implementations specify the conditions required by each certificate.

## Quick start

Run the following commands from the repository root using **Python 3.11 or newer**:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,validated]'
python scripts/prepare_recorded_paths.py
PYTHONPATH=.:src python -m pytest -q
shasum -a 256 -c MANIFEST.sha256
```

The recorded verification run returned **248 passed, 1 skipped**. The documented skip concerns an original controller configuration containing a machine path.

Most mathematical and count-level checks run without a dataset or neural checkpoint. The setup script creates local compatibility links for paths stored in the original protocols; these generated links are excluded from the repository and file manifest.

For complete dependency information, external data requirements, and experiment commands, see [the reproduction guide](docs/REPRODUCING.md).

## Reproduce analyses from retained results

The commands below regenerate the exact witness and the two CIFAR-10.2 analyses from retained rows. They write to a separate directory to preserve the recorded outputs.

```sh
mkdir -p reproduced

PYTHONPATH=.:src python experiments/certify_local_lshape_witness.py \
  --output reproduced/local_lshape.json

PYTHONPATH=.:src python scripts/reanalyze_cifar102.py \
  --output-directory reproduced/renyi

PYTHONPATH=.:src python scripts/validate_cifar102_certificates.py \
  --output-directory reproduced/validated
```

### Reproduction scope

| Task | Requirements |
| --- | --- |
| Run most mathematical and count-level checks | Installed repository dependencies |
| Regenerate the exact nonconvex witness | Installed repository dependencies |
| Reanalyze retained CIFAR-10.2 counts | Retained rows and the dependencies specified in the reproduction guide |
| Check count-to-radius calculations with outward arithmetic | Retained rows and validated-numerics dependencies |
| Rerun image-model or learned-controller experiments | Additional software and external data specified in the reproduction guide |

Reanalyzing stored counts does not rerun model inference or regenerate the original noisy proposals.

The AuditVotes source and checkpoint and the CIFAR datasets are **not redistributed** in this repository. See [the reproduction guide](docs/REPRODUCING.md) for acquisition and setup instructions.

## Experimental results

### Projected-band study

The projected-band study evaluates all **2,000 CIFAR-10.2 test images** using one checkpoint, one noise scale, and a filter selected on training data.

The table reports correctly classified images certified at **radius 0.2**:

| Certificate | Correctly certified images |
| --- | ---: |
| Conditional Rényi for the filtered predictor | 892 |
| Joint mass for the same filtered predictor | 725 |
| Unfiltered Gaussian smoothing | 948 |

The conditional Rényi calculation reuses stored counts. The outward-arithmetic run checks the count-to-radius calculations and leaves the reported totals unchanged.

The conditional Rényi certificate certifies more images than the joint-mass certificate for the same filtered predictor in this evaluation. The unfiltered predictor certifies more images than either filtered-predictor certificate.

### Released-model study

A separate released-model study confirms **12 label changes inside substituted conditional radii** among **128 images selected without model output**.

This count is the yield of the fixed search procedure. It is **not an estimate of population prevalence**.

## Figures and interpretation

![A counterexample to substituting conditional votes into the Gaussian formula, and a learned-model comparison of valid certificates](docs/figures/conditioning_failures.png)

- **Left panel:** A decision boundary lies inside the radius obtained by substituting conditional votes into the ordinary Gaussian formula.
- **Right panel:** Valid certificates are compared on a separate projected-band filter.

Additional figures:

| Figure | What it shows |
| --- | --- |
| [Released-model evaluation and endpoint tests](docs/figures/auditvotes_recertification.png) | Released-model results and fresh endpoint tests |
| [Supplementary certificate comparison](docs/figures/conditional_renyi_reanalysis.png) | Conditional Rényi, reverse-KL, forward-KL, joint-mass, and unfiltered calculations |

## Repository guide

| Component | Location |
| --- | --- |
| Installation, full reproduction commands, dependencies, and external data | [docs/REPRODUCING.md](docs/REPRODUCING.md) |
| Joint-mass certificate for the unchanged filtered predictor | [src/feasible_robustness/filtered_certificate.py](src/feasible_robustness/filtered_certificate.py) |
| Conditional covariance and Rényi certificate | [src/feasible_robustness/conditional_renyi.py](src/feasible_robustness/conditional_renyi.py) |
| Outward binomial and radius calculations | [src/feasible_robustness/validated_numerics.py](src/feasible_robustness/validated_numerics.py) |
| Finite-horizon trajectory comparison | [src/feasible_robustness/trajectory_certificate.py](src/feasible_robustness/trajectory_certificate.py) |
| Exact nonconvex witness | [experiments/certify_local_lshape_witness.py](experiments/certify_local_lshape_witness.py) |
| Retained CIFAR-10.2 count reanalysis | [scripts/reanalyze_cifar102.py](scripts/reanalyze_cifar102.py) |
| Outward-arithmetic CIFAR-10.2 validation | [scripts/validate_cifar102_certificates.py](scripts/validate_cifar102_certificates.py) |
| Recorded study protocols | [research/](research/) |
| Retained results | [outputs/](outputs/) |
| Retained controller checkpoints | [controllers/](controllers/) |
| File integrity checks | [MANIFEST.sha256](MANIFEST.sha256) |

## License

No license is currently included in this repository. Permissions for reuse, modification, and redistribution have not been specified.

## Citation

If you use this code, the retained results, or the methods in your research, please cite the accompanying paper.

[Preprint on arXiv](https://arxiv.org/abs/2609.39497)

```bibtex
@misc{khilji2026geometryrandomizedsmoothingfeasible,
  title={The Geometry of Randomized Smoothing on Feasible Sets},
  author={Syed Izhan Khilji and Alireza Furutanpey and Schahram Dustdar},
  year={2026},
  eprint={2609.39497},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2609.39497}
}
```
