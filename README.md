# Market-Impact Activation Axis

This repository tests whether a pretrained language model contains an internal
activation direction associated with market-impactful financial text. A Ridge
model maps document activations to a normalized post/pre Parkinson-volatility
expansion label; its coefficient defines the activation axis. The resulting
document score is compared with a direct LLM rating and a BGE-M3 embedding
score, then added to AR, HAR, HAR-X, and MIDAS volatility forecasts.

The current main experiments use:

- **News:** FinTexTS target-company news, 2019-2021 train / 2022 validation /
  2023 exploratory test.
- **Filings:** SEC 8-K disclosure text, 2022-2023 train / 2024 validation /
  2025 test.
- **LLM:** Qwen2.5-7B-Instruct intermediate activations.
- **Label:** firm Post-5/Pre-20 Parkinson-volatility expansion minus the same
  market expansion. The News main result additionally removes train-period
  ticker means.

## Repository map

- [`plan.md`](plan.md): research overview, figures, results, and TODOs.
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md): required artifacts and exact run
  order.
- [`scripts/`](scripts): the minimal scripts used by the current News and 8-K
  main results.
- [`joint_impact_axis/`](joint_impact_axis): prepared-text activation extraction
  for long 8-K documents.
- [`results/reference/`](results/reference): small CSV/JSON outputs used as
  reference checks. Raw text, prices, embeddings, and activations are excluded.

## Important limitation

The 2023 News split predates the Qwen2.5 release and may overlap the model's
pretraining data. It is therefore exploratory rather than a clean confirmatory
holdout. The highest-priority next experiment is a frozen, cutoff-safe temporal
replication. See the TODO section in [`plan.md`](plan.md#7-todo).

## Quick replay

After placing the external artifacts at the paths listed in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export PYTHONPATH="$PWD/scripts:$PWD"
bash scripts/replay_main_results.sh
python scripts/verify_main_results.py
```

The four main replay commands are CPU-only. GPU is needed only to regenerate
the activation and embedding arrays.

## Cutoff-safe Llama 2 News replication

The repository also contains a staged News-only replication that replaces
Qwen2.5 with the base Llama 2 7B checkpoint.  Llama 2 pretraining ends in
September 2022; the Chat checkpoint is deliberately not used because its
tuning data can extend into 2023.  The existing BGE-M3 model remains the
dedicated-embedding baseline.

The pipeline preserves the main News sample and normalized Parkinson label,
but does not allow 2023 representations to be extracted until the 2022
validation selection has been written:

```bash
export PYTHONPATH="$PWD/scripts:$PWD"
bash scripts/run_news_cutoff_safe.sh
```

The explicit stages are implemented in
[`scripts/news_cutoff_safe_llama2.py`](scripts/news_cutoff_safe_llama2.py):

1. rebuild the canonical 25,767 / 12,051 / 12,478 News split;
2. extract BGE and all 32 Llama layers for train and validation;
3. tune BGE alpha and Llama layer/alpha on validation, then freeze them;
4. extract and evaluate the 2023 test representations;
5. compare baseline, BGE, and Llama scores in Parkinson forecasting.

This removes the Qwen knowledge-cutoff overlap, but the 2023 test period was
already inspected in earlier research.  It is therefore a cutoff-safety
replication rather than a fresh confirmatory holdout.

The completed pinned run selected Llama layer 17 with Ridge alpha 1.0 before
opening test representations.  On the 12,478-document 2023 test, Llama
activation reached 0.1628 Spearman / 0.2057 Pearson versus 0.0904 / 0.1215 for
BGE-M3.  The Spearman difference was +0.0723 (date-block 95% CI
[0.0536, 0.0924]; ticker-block [0.0494, 0.0950], 2,000 resamples).  Adding the
Llama score improved forecast QLIKE over the no-text baseline by 4.38--4.86%
at one day and 4.52--5.28% at five days across AR(5), HAR, HAR-X, and MIDAS.

Only the Llama activation checkpoint is cutoff-safe for this protocol.  BGE-M3
is intentionally retained as the existing small-embedding baseline and is not
claimed to be a cutoff-matched historical checkpoint.

## Uncentered capacity and 8-K controls (July 2026)

The latest requested analysis does **not** subtract train-ticker means from
labels or representations.  It compares BGE-M3 (0.57B), Qwen3-Embedding-8B,
and all 32 layers of base Llama 2 7B.  Qwen3 is a capacity control only; it is
not cutoff-safe for either test period.  A matched 64-unit MLP head is evaluated
alongside Ridge, and all six resulting scores are added to the same AR(5), HAR,
HAR-X, and MIDAS forecasts.

For News, the uncentered 2023 test Spearman correlations are 0.0829 (BGE),
0.0893 (Qwen3-8B), and 0.1511 (Llama activation, validation-selected layer 16).
Average QLIKE improvements over the price-only baselines are respectively
1.57%, 3.07%, and 4.20% at one day and 1.38%, 1.64%, and 4.60% at five days.

The new public-data 8-K panel contains 863 / 1,232 / 890 documents in
2022-2023 train / 2024 validation / 2025 test.  Its test Spearman correlations
are 0.3239 (BGE), 0.3409 (Qwen3-8B), and 0.3301 (Llama activation,
validation-selected layer 32).  Activation therefore does not win the direct
8-K impact-correlation comparison.  It does win the downstream forecast
comparison: its Ridge score reduces one-day QLIKE by 8.65% on average versus
the price-only baselines and by 9.24--10.36% versus the Qwen3 Ridge score.

The public 8-K CSV lacks EDGAR acceptance timestamps.  This run consequently
uses the first ticker trading session strictly after `file_date`, not a claimed
exact acceptance-time event mapping.  The limitation is recorded in
`outputs/sec8k_uncentered/data_manifest.json`.

Run the complete 8-K protocol with:

```bash
export PYTHONPATH="$PWD/scripts:$PWD"
bash scripts/run_sec8k_uncentered.sh
```
