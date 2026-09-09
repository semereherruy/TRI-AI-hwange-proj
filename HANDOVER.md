# Project handover — status as of 2026-09-09

Early Detection of Toxic Language through Latent Probing of Large Language Models.

**One sentence:** the entire pipeline is built, tested and pushed; every stage
through Gemma hidden-state extraction has been verified on real hardware; the
layer-wise probe run is the only thing standing between the current state and the
core scientific result.

---

## 1. Resuming on a new machine

Everything needed is in git. Nothing important lives only on the old laptop.

```bash
git clone https://github.com/semereherruy/TRI-AI-hwange-proj.git
cd TRI-AI-hwange-proj
python3 -m venv .venv
./.venv/bin/pip install -r requirements-local.txt
./.venv/bin/python check_env.py
```

Then authenticate once, so gated datasets and Gemma resolve:

```bash
./.venv/bin/hf auth login      # paste a read token from huggingface.co/settings/tokens
```

The token is stored machine-wide in `~/.cache/huggingface/`. It is deliberately
**not** in the repository, and must be re-created on each new machine.

**Two things that do not travel with the repo** and must be re-done:

| Item | Why | Fix |
| --- | --- | --- |
| HuggingFace token | machine-local cache, never committed | `hf auth login` |
| `data/raw/HateXplain-master/` | vendored third-party repo, git-ignored | `git clone --depth 1 https://github.com/hate-alert/HateXplain.git data/raw/HateXplain-master` |

AfriHate and ToxiGen are fetched from the Hub at run time and need no local copy.
The account `semere27` already has accepted licences for `afrihate/afrihate` and
`google/gemma-3-1b-pt`.

### Running the pipeline

```bash
python -m src.data.inspection     # Phase 2  -> reports/phase2_inspection/
python -m src.data.canonical      # Phase 4  -> data/processed/canonical.parquet
python -m src.data.quality        # Phase 5  -> reports/phase5_quality/
python -m src.data.splits         # Phase 6  -> data/processed/canonical_split.parquet
python -m src.analysis.baselines  # Phase 7  -> reports/phase7_baselines/
python -m src.probing.extraction  # Phase 8  -> data/embeddings/    (needs a CUDA GPU)
python -m src.probing.probes      # Phase 9-10 -> reports/phase9_probes/
```

Everything up to and including baselines runs on a laptop in a few minutes.
Phase 8 needs a GPU and runs on Colab via `notebooks/colab_run.ipynb`.

---

## 2. What is built

Fourteen modules across four packages, plus a Colab driver notebook.

| Area | Modules | Status |
| --- | --- | --- |
| Config | `src/paths.py`, `src/config.py`, `configs/data.yaml`, `configs/model.yaml` | done |
| Inspection | `src/data/profiling.py`, `src/data/inspection.py` | done, reports committed |
| Loading | `src/data/loaders.py`, `src/data/schema.py` | done, all four sources |
| Canonical | `src/data/canonical.py` | done, 43,087 rows |
| Quality | `src/data/quality.py` | done, nothing removed silently |
| Splitting | `src/data/splits.py` | done, leakage check passes |
| Baselines | `src/analysis/baselines.py`, `src/analysis/metrics.py` | done, results committed |
| Extraction | `src/probing/extraction.py` | done, verified on real Gemma |
| Probing | `src/probing/probes.py` | done, awaiting a GPU run |

---

## 3. Data

43,087 rows, four sources, two languages, zero leakage.

| Source | Rows | Language | Provenance |
| --- | --- | --- | --- |
| HateXplain | 19,229 | English | human posts, 3 human annotators each |
| ToxiGen | 9,900 | English | machine-generated, human-rated; 148 rows human-written |
| Ubuntu | 9,000 | EN + Amharic | template-generated, synthetic |
| AfriHate | 4,958 | Amharic | human tweets, native-speaker annotation |

Splits: **27,639 train / 2,666 val / 3,782 test / 9,000 Ubuntu probe.**
Leakage verification: **0 groups and 0 texts span split boundaries.**

Splitting is group-level, never row-level. A `group_id` derived from normalized
text keeps duplicates, templates and translation pairs together; Ubuntu's
Amharic rows inherit the group of their English pivot sentence.

---

## 4. Results so far — the TF-IDF baseline

This is the yardstick the Gemma probe must beat. Word 1–2 grams + logistic
regression, trained on `train`, evaluated on `test` (n = 3,782).

| Metric | Value |
| --- | --- |
| F1 | **0.781** |
| Balanced accuracy | 0.745 |
| Precision / Recall | 0.801 / 0.761 |
| ROC-AUC / PR-AUC | 0.825 / 0.866 |
| False-positive rate | 0.272 |
| False-negative rate | 0.239 |

By language: **Amharic 0.840** (n=745), English 0.759 (n=3,037).
By source: AfriHate 0.840, HateXplain 0.794, ToxiGen 0.678.
By toxicity type: abuse 0.940, hate speech 0.874, offensive language 0.804.

### Three findings that already stand on their own

**Legitimate ethnic-group discussion is over-flagged.** 96 of 411 non-toxic test
rows that name an ethnic group are classified toxic — a **23.4% false-positive
rate** on exactly the speech both project cards warn about suppressing.

**No keyword shortcut in the lexical baseline.** Masking neutral group names
across 1,272 test texts costs only **0.018 F1**. The baseline is not simply
keyword-spotting, which sets a demanding bar for the probe.

**Ubuntu generalisation, out of domain.** On the held-out probe set the baseline
reaches F1 0.854 in English and 0.712 in Amharic. Per relational harm type:
dehumanization 1.000, threat 0.905, silencing 0.894, stereotyping 0.806,
belonging denial 0.685, exclusion 0.667.

---

## 5. Research decisions

All labelling decisions live in `configs/data.yaml`, are documented with the
reasoning, and are fingerprinted into `canonical_metadata.json` so any result
traces to the policy that produced it. Changing one and re-running regenerates
everything downstream.

| Decision | Setting | Basis |
| --- | --- | --- |
| HateXplain `offensive` | **toxic** | Data Card §2/§5: the toxic class covers hate speech, harassment, abuse and offensive language |
| HateXplain no-majority posts (919) | **excluded** | matches the official split, which already omits exactly these |
| ToxiGen binarization | **`toxicity_human` > 2.5** | published convention; ~15% of ratings sit in the 3.0–3.7 band, so this materially moves class balance |
| AfriHate `Abuse` | **toxic** | same Data Card reasoning as `offensive`; confirmed by the researcher |
| Ubuntu role | **probe only, never trained on** | 33 unique sentences cannot support training; also satisfies "evaluate synthetic data separately" |

### Alignment between the two cards

The Data Card (v2) and the Impact Assessment Card disagree on scope: the Data
Card is broad-toxicity and English-predominant with no Ubuntu framing, while the
Impact Card is titled "Ubuntu-Aligned" and lists relational harm, relational
target and severity as predicted outputs. Resolution in force: **Data Card governs
dataset composition, Impact Card governs evaluation metrics.** Ubuntu-as-held-out
probe satisfies both.

**Open gap:** the Impact Card's predicted outputs include relational harm, target
and severity. The pipeline currently predicts only the binary label and
*evaluates* against the relational categories. Predicting them would need
training data, and 33 template sentences cannot support a 6-way classifier. A few
hundred genuinely distinct Ubuntu sentences would close this.

**Also unresolved, from Data Card §4:** HateXplain instances are social-media
statements, not user prompts, while the SOP pipeline takes prompts. The project
currently treats them as prompts (the "Reframe" option). Choosing prompt-native
data instead would change the corpus, not the code.

---

## 6. Bugs found and fixed

Recorded because several would have produced **plausible but meaningless
results** rather than errors. These are the real risk in this kind of study.

| # | Bug | Why it mattered |
| --- | --- | --- |
| 1 | `AutoModel` matched **zero** of Gemma's weights | Returned a fully randomly-initialized model with no error. Would have yielded a complete, believable layer curve computed from noise. Fixed by loading `AutoModelForCausalLM`; a guard now refuses to extract if any weight is unloaded. |
| 2 | Gemma pads **left**; last-token pooling assumed right | Every padded sequence returned a padding vector instead of its final token. Half the study (`last_token`) would have been noise. Caught by comparing against an unbatched reference: `mean` matched to 9.5e-4, `last_token` was off by 3.05. |
| 3 | **float16** overflow | Gemma-3 is trained in bfloat16; float16 caps at 65,504 and Gemma's activations exceed it, silently becoming `inf`. Now bfloat16 compute, float32 storage, with a finiteness assert per batch. |
| 4 | ~10.4 GB of RAM buffering | Would have died deep into a long Colab run. Now streams into memory-mapped arrays; RAM is flat regardless of corpus size. |
| 5 | `check_env.py` imported packages to read versions | Importing `umap-learn` on Colab pulls numba/llvmlite and killed the interpreter with no traceback. Now reads `importlib.metadata`. |
| 6 | Notebook `!cmd` cells don't fail | Non-zero exits rendered green, so the first visible symptom was a `FileNotFoundError` several cells later. Now every step is exit-code checked and prints captured output. |
| 7 | Token not reaching subprocesses | `os.environ` alone was insufficient; now `login()` writes to the runtime cache, and `auth_check` verifies both gated repos up front. |
| 8 | Nested clone | Re-running the clone cell produced `.../TRI-AI-hwange-proj/TRI-AI-hwange-proj`. Clone is now idempotent. |

Bugs 1, 2 and 3 are the significant ones: each produces output that looks fine.

---

## 7. What is left

| Phase | State | Needs |
| --- | --- | --- |
| 8 extraction | code done, verified | one GPU run |
| 9 layer-wise probes | code done | the embeddings from Phase 8 |
| 10 controls | code done | same run |
| 11 cross-source / multilingual | partly — per-language breakdowns come free | a module for explicit train-on-X-test-on-Y |
| 12 Ubuntu analysis | partly — probe set is evaluated | a per-harm-type module over probe results |
| 13 interpretability, PCA, error analysis | **not built** | ~1 module |
| 14 statistical validation | **not built** | must use bootstrap, not seeds — see below |
| 15 final report | **not built** | the above |

Estimated: one GPU run, then roughly four modules and one further CPU-only pass
over the saved embeddings.

**If time runs out, the Phase 8–10 run alone is a defensible submission:** layer
curve, controls, baselines, leakage-safe splits and full provenance. Phases 13–15
are analysis polish on a result that would already exist.

---

## 8. Known risks and cautions

**Seeds do nothing.** `LogisticRegression` with lbfgs is deterministic on fixed
data, so the three-seed loop reports std = 0.0 by construction. This is recorded
as `seed_variance_note` in the results. Phase 14 must draw variance from bootstrap
resampling or resampled group-level splits — a seed loop there would be theatre.

**Version drift.** Colab runs Python 3.13 / transformers 5.16 / hub 1.28; local
development is Python 3.9 / transformers 4.57 / hub 0.36. Model loading now tries
both API variants, but this gap is the most likely source of new breakage.

**How to read the layer curve.** If it is flat across all layers, or `last_token`
and `mean` disagree sharply, treat that as a probable extraction bug rather than a
finding — that is the signature of the failures above.

**What counts as a result.** Probe F1 above 0.781 overall (0.840 Amharic) with
controls near chance means Gemma encodes toxicity before generation. Matching the
baseline is an honest negative result. Beating it specifically on Amharic would be
evidence of cross-lingual transfer inside the model, and is the most interesting
of the three outcomes. Per Impact Card §10, failure is itself a useful result.
