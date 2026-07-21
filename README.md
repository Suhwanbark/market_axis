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
