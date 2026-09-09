# Early Detection of Toxic Language through Latent Probing of Large Language Models

## Research objective

Investigate whether toxicity- and harm-related information is encoded in the internal
hidden representations of a Gemma 1B-class language model **before** response generation.

The claim under test is one of *decodability*: if a lightweight linear probe can recover
harm labels from frozen hidden states at layer *k*, that information is present in the
representation at that depth. Linear probing establishes correlation and decodability,
not causal use by the model — Phase 13 keeps that distinction explicit.

## Datasets

Four sources, deliberately heterogeneous in language, provenance and annotation scheme.
Status below reflects a preliminary look at `data/raw/`; **Phase 2 is the formal inspection**
and is the authority on shapes, label vocabularies and provenance.

| Source | Location | Status |
| --- | --- | --- |
| HateXplain | `data/raw/HateXplain-master/Data/dataset.json` | Present. 20,148 posts, multi-annotator, with official `post_id_divisions.json` splits (15,383 / 1,922 / 1,924). |
| Ubuntu (EN/AM) | `data/raw/*.csv` | Present. 4,000-row and 500-row files. **Template-generated — see the warning below.** |
| AfriHate | `data/raw/AfriHate-main/` | Repo clone only: README and assets, **no data**. Lives on the Hub at `afrihate/afrihate`. |
| ToxiGen | `data/raw/TOXIGEN-main/` | Repo clone only: generation prompts and scripts, **no labels**. Labelled data is gated on the Hub at `toxigen/toxigen-data`. |

### Provenance warning — Ubuntu CSVs

Both Ubuntu files draw from **33 distinct `text_en` values**, shared identically between them
(4,000 rows and 500 rows respectively). Labels and metadata are perfectly balanced across
those 33 sentences, which is the signature of template generation, not human annotation.

Consequences carried through the whole pipeline:

- These are **synthetic** data (`is_synthetic = True` in the canonical schema) and are never
  reported as human-annotated ground truth.
- A random row-level split would place the same sentence in train and test. Ubuntu data must
  be split by **template group**, not by row (Phase 6).
- English and Amharic rows are **translation pairs**. They are kept as separate language
  samples sharing a `pair_id`, never concatenated into one model input, and never separated
  across a split boundary.

The Ubuntu harm taxonomy (`exclusion`, `belonging denial`, `dehumanization`, `threat`,
`silencing`, `stereotyping`) is treated as a culturally grounded ethical perspective —
one lens among several, not universal ground truth.

## Methodology

Phases run in order; each is reviewed before the next begins.

| Phase | Output |
| --- | --- |
| 1 | Environment and project setup |
| 2 | Dataset discovery and inspection reports |
| 3 | Per-dataset loaders with preserved source metadata |
| 4 | Canonical unified schema |
| 5 | Data quality and leakage control |
| 6 | Leakage-safe splits (group-aware) |
| 7 | TF-IDF + Logistic Regression baselines |
| 8 | Gemma frozen hidden-state extraction |
| 9 | Layer-wise linear probes |
| 10 | Probe controls (shuffled labels, random embeddings, pooling variants) |
| 11 | Cross-dataset and multilingual generalization |
| 12 | Ubuntu / African ethical analysis |
| 13 | Interpretability and error analysis |
| 14 | Statistical validation across seeds |
| 15 | Final outputs and reproducibility documentation |

## Project structure

```
.
├── configs/            # experiment configuration (no hard-coded paths in src/)
├── data/
│   ├── raw/            # source datasets; third-party repos are git-ignored
│   └── processed/      # generated canonical dataset — reproducible, git-ignored
├── external/           # third-party code kept outside the import path
├── notebooks/          # exploration and visualization only
├── reports/            # generated inspection and quality reports (tracked)
├── src/
│   ├── data/           # loaders, canonical schema, cleaning, splitting
│   ├── probing/        # extraction and linear probes
│   └── analysis/       # metrics, plots, statistical validation
├── check_env.py        # Phase 1 environment verification
├── requirements.txt        # Colab / CUDA runtime
└── requirements-local.txt  # macOS x86_64 development
```

Reusable logic lives in `src/`. Notebooks call into it; they do not contain the pipeline.

## How to run

### Execution model

Authoring happens locally in this repository; **Gemma extraction (Phase 8) runs on Colab**,
where a CUDA GPU is available. The notebook is a thin driver, not the codebase.

### Colab

```python
!git clone https://github.com/<user>/<repo>.git
%cd <repo>
!pip install -q -r requirements.txt
!python check_env.py
```

Gated assets (Gemma weights, AfriHate, ToxiGen) need a HuggingFace token. Store it in the
Colab **Secrets** panel as `HF_TOKEN` — never in a notebook cell, never in this repository.

```python
from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

To pick up later changes, `!git pull` — do not re-paste code into cells.

### Local

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements-local.txt
./.venv/bin/python check_env.py
```

Local runs cover inspection, cleaning, splitting and TF-IDF baselines. Phase 8 is not
expected to complete here; see the environment note below.

## Environment notes

Verified 2026-09-09 on the development machine:

- Python 3.9.6, macOS x86_64 (Intel i7-9750H, 16 GB), AMD Radeon Pro 5300M (4 GB).
- No CUDA. `torch.backends.mps` reports available, but on this hardware it is unverified
  for Gemma and should be treated as CPU-class.
- PyTorch published no macOS x86_64 wheel after **2.2.2**, which is compiled against the
  NumPy 1.x C API. Under NumPy 2.x, `torch` imports but `tensor.numpy()` fails — the exact
  path Phase 8 feeds into Phase 9. `requirements-local.txt` therefore pins `numpy<2`.
- `datasets` is capped below 4.x locally because 4.x requires Python ≥3.10. Colab runs
  Python 3.12 and is unaffected; this divergence is recorded so results stay traceable.

Run `python check_env.py` in either environment; it exits non-zero on a blocking problem.

## Reproducibility

- Random seeds are fixed and recorded; Phase 14 reports mean ± standard deviation across
  seeds rather than a single run.
- `label_original` is preserved for every sample so no source annotation is ever lost to a
  binary mapping.
- Every removal in cleaning is recorded with a reason (Phase 5); nothing is dropped silently.
- Split construction is documented per phase and is group-aware wherever templates,
  translation pairs or duplicates exist.
