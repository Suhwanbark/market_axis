# Reproducibility Guide

## 1. Scope

This package reproduces the current normalized-Parkinson experiments:

1. News: tune the BGE Ridge alpha and the Qwen activation layer/Ridge alpha on
   2022 validation, then evaluate 2023.
2. News: compare panel AR(5), HAR, HAR-X, and MIDAS with no text score, one BGE
   score, or one activation score.
3. 8-K: sweep the activation layer for market-adjusted volatility-expansion
   labels using 2024 validation.
4. 8-K: compare the same forecast families on the 2025 event panel.

Large and potentially licensed artifacts are deliberately not committed. Exact
numeric replay requires the prepared data and representation arrays below.

## 2. Tested environment

The recorded run used Python 3.10.19, NumPy 1.26.4, pandas 2.3.3, SciPy 1.15.3,
scikit-learn 1.7.2, PyArrow 22.0.0, PyTorch 2.9.1, and Transformers 4.52.4.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/scripts:$PWD"
```

For GPU extraction, set the local Qwen snapshot explicitly:

```bash
export QWEN25_MODEL=/absolute/path/to/Qwen2.5-7B-Instruct
```

The original Qwen snapshot was revision
`a09a35458c702b33eeacc393d103063234e8bc28`. New confirmatory runs should also
record an explicit embedding-model revision rather than relying on a moving
model name.

## 3. Required external artifacts

Place the files under the repository root with these exact relative paths:

| Artifact | Expected shape or role |
|---|---|
| `outputs/fintexts_news_axis_e2e/news_layers1-28.npy` | `(51729, 28, 3584)`, float16 News activations |
| `outputs/fintexts_news_axis_e2e/hf_cache/` | Pinned FinTexTS parquet snapshot used for OHLC and text |
| `outputs/fintexts_news_axis_calendar_corrected/news_records.parquet` | Calendar-corrected News rows and split |
| `outputs/fintexts_news_axis_calendar_corrected/data_audit.json` | Removed carry-forward closure dates |
| `outputs/layer_axis_robustness/qwen25_current_news/strategy_scores.parquet` | News activation-array row map |
| `outputs/joint_impact_axis_v1/filing_documents.parquet` | Prepared 8-K documents and split |
| `outputs/joint_impact_axis_v1/activations/qwen25/filing_layers.npy` | `(2417, 28, 3584)`, float16 8-K activations |
| `outputs/sec8k_eventtime_impact/prices_with_volume.parquet` | Adjusted 8-K OHLCV panel including SPY |
| `outputs/sec8k_harx_activation_forecast/daily_panel.parquet` | 8-K event-session panel |
| `outputs/ohlc_volatility_forecast_comparison/sec8k_adjusted_open.parquet` | Adjusted open-price cache |
| `outputs/impact_ratio_representation_benchmark/documents.parquet` | Alignment table for News and 8-K |
| `outputs/impact_ratio_representation_benchmark/embedding_vectors.npy` | `(52750, 1024)`, float16 BGE-M3 embeddings |
| `outputs/impact_ratio_representation_benchmark/direct_scores.parquet` | Direct Qwen 1-9 expected scores |

The SHA-256 hashes of the largest canonical artifacts are in
[`artifacts.sha256`](artifacts.sha256). Hugging Face and market-data sources can
change over time, so downloading them again is not guaranteed to reproduce the
same numbers.

## 4. Main CPU replay

Run from the repository root:

```bash
export PYTHONPATH="$PWD/scripts:$PWD"

python scripts/news_ticker_centered_parkinson_axis_tuned.py
python scripts/news_ticker_centered_parkinson_forecast_tuned.py
python scripts/sec8k_simple_label_family_benchmark.py
python scripts/sec8k_market_adjusted_parkinson_forecast.py
```

Or run all four with:

```bash
bash scripts/replay_main_results.sh
```

Then verify the key frozen values:

```bash
python tests/test_core_formulas.py
python scripts/verify_main_results.py
```

Generated outputs are written to:

```text
outputs/news_ticker_centered_parkinson_axis_tuned/
outputs/news_ticker_centered_parkinson_forecast_tuned/
outputs/sec8k_simple_label_family_benchmark/
outputs/sec8k_market_adjusted_parkinson_forecast/
```

Small outputs from the original run are retained under `results/reference/`.

## 5. Regenerating representations

### 5.1 News activations

The historical main run used 512 tokens and two shards. On a one-GPU server,
run shard 0 and shard 1 sequentially.

```bash
export QWEN25_MODEL=/absolute/path/to/Qwen2.5-7B-Instruct
export PYTHONPATH="$PWD/scripts:$PWD"

CUDA_VISIBLE_DEVICES=0 python scripts/fintexts_news_axis_pipeline.py \
  --stage extract-shard --shard-id 0 --num-shards 2 \
  --batch-size 8 --max-length 512

CUDA_VISIBLE_DEVICES=0 python scripts/fintexts_news_axis_pipeline.py \
  --stage extract-shard --shard-id 1 --num-shards 2 \
  --batch-size 8 --max-length 512

python scripts/fintexts_news_axis_pipeline.py --stage merge --num-shards 2
```

### 5.2 8-K activations

The main 8-K result uses 2,048-token body-only representations with 256-token
overlap for long documents.

```bash
export QWEN25_MODEL=/absolute/path/to/Qwen2.5-7B-Instruct
export PYTHONPATH="$PWD/scripts:$PWD"

CUDA_VISIBLE_DEVICES=0 python -m joint_impact_axis.pipeline \
  --stage extract-shard --out-dir outputs/joint_impact_axis_v1 \
  --model qwen25 --domain filing --shard-id 0 --num-shards 2 \
  --document-batch-size 1 --chunk-batch-size 1 \
  --max-length 2048 --overlap 256 --disk-floor-gib 20

CUDA_VISIBLE_DEVICES=0 python -m joint_impact_axis.pipeline \
  --stage extract-shard --out-dir outputs/joint_impact_axis_v1 \
  --model qwen25 --domain filing --shard-id 1 --num-shards 2 \
  --document-batch-size 1 --chunk-batch-size 1 \
  --max-length 2048 --overlap 256 --disk-floor-gib 20

python -m joint_impact_axis.pipeline \
  --stage merge --out-dir outputs/joint_impact_axis_v1 \
  --model qwen25 --domain filing --num-shards 2 --disk-floor-gib 20
```

### 5.3 BGE-M3 and Direct Qwen scores

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/impact_ratio_representation_benchmark.py \
  extract-embedding --model BAAI/bge-m3 --batch-news 16 --batch-filing 4

CUDA_VISIBLE_DEVICES=0 python scripts/impact_ratio_representation_benchmark.py \
  extract-direct --batch-news 8 --batch-filing 2
```

The exact BGE model revision was not pinned in the original artifact. Use the
provided embedding array for exact replay, and pin a revision in all new runs.

## 6. Data and timing conventions

- Daily Parkinson variance is `log(High / Low)^2 / (4 log 2)`.
- The axis label is the log Post-5/Pre-20 firm volatility expansion minus the
  corresponding market expansion.
- News uses the next five actual trading sessions after the dated news row.
- 8-K uses the mapped reaction session based on SEC acceptance time.
- News train/validation/test is 2019-2021 / 2022 / 2023.
- 8-K train/validation/test is 2022-2023 / 2024 / 2025.
- Forecast regressions fit train and validation together after all layer and
  hyperparameter choices have been frozen on validation.
- Forecasting uses panel OLS with shared slopes and ticker fixed effects.

## 7. Audit limitations

1. The 2023 News test predates the Qwen2.5 release and may be represented in
   pretraining. Treat it as exploratory until a cutoff-safe holdout is run.
2. The latest News axis tunes both layer and Ridge alpha on validation; the
   latest 8-K axis selects layer on validation but keeps Ridge alpha fixed at 10.
3. The 8-K market-average label code should explicitly exclude SPY in the final
   protocol; the current historical artifact may include it in the equal-weight
   date average.
4. The original test periods were inspected repeatedly during exploration.
   Future confirmatory runs must freeze all choices before opening a fresh test.

## 8. Cutoff-safe Llama 2 News run

The cutoff-safe replication uses these pinned model revisions:

```text
NousResearch/Llama-2-7b-hf
  8efe6c9b93655b934e27bd9981e3ec13e55aee9d

BAAI/bge-m3
  5617a9f61b028005a4858fdac845db406aefb181
```

The NousResearch repository is used as an ungated mirror of the base Llama 2
weights.  No Chat/instruction-tuned checkpoint is used.  Place the snapshots at
`models/llama2-7b-base` and `models/bge-m3`, then run:

```bash
export PYTHONPATH="$PWD/scripts:$PWD"
bash scripts/run_news_cutoff_safe.sh
```

The run order is enforced in code:

```text
prepare data
  -> BGE train/validation inference
  -> Llama train/validation inference
  -> validation-only layer/alpha selection and freeze
  -> BGE/Llama 2023 test inference
  -> document ranking and block bootstrap
  -> AR/HAR/HAR-X/MIDAS forecasting
```

Primary outputs are written to:

```text
outputs/news_cutoff_safe_llama2/
outputs/news_cutoff_safe_llama2_forecast/
```

The Llama arrays are split by train/validation/test.  Each has 32 layers and a
4,096-dimensional hidden state.  GPU forward uses BF16 and arrays are stored as
FP16.  BGE-M3 uses its 1,024-dimensional CLS state followed by L2
normalization.  Both supervised comparisons use the same train-ticker
centering, label, alpha grid, validation split, and test rows.

## 9. No-ticker-centering News and 8-K runs

The later capacity controls intentionally supersede the centering statement
above for their own outputs: neither representation centroids nor label means
are subtracted by ticker.  The raw target remains the market-adjusted
Post-5/Pre-20 Parkinson log expansion.

News outputs:

```text
outputs/news_uncentered_linear_control/
outputs/news_qwen3_embedding_8b_control/
outputs/news_uncentered_mlp_control/
outputs/news_capacity_matched_forecast/
outputs/news_uncentered_mlp_forecast/
```

The 8-K source is the public
`Disclosures-SSRC/8k_disclosure_dataset` CSV pinned to revision
`2370739fda5983b658556692b38a3b9bdbc5a0cb`.  Place it at:

```text
data/sec8k_disclosure_dataset/8k_all_data_with_text.csv
```

The 8-K runner enforces train/validation extraction, selection freeze, and only
then test extraction:

```bash
export PYTHONPATH="$PWD/scripts:$PWD"
bash scripts/run_sec8k_uncentered.sh
```

Primary 8-K outputs are:

```text
outputs/sec8k_uncentered/
outputs/sec8k_uncentered_mlp/
outputs/sec8k_uncentered_forecast/
```

Pinned model revisions are BGE-M3
`5617a9f61b028005a4858fdac845db406aefb181`, Qwen3-Embedding-8B
`1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`, and base Llama 2 7B
`8efe6c9b93655b934e27bd9981e3ec13e55aee9d`.  8-K inference uses the same
cleaned first 5,000 characters for all models with a 2,048-token cap.  BGE and
Qwen produce one embedding; Llama saves every layer as FP16.

The public CSV contains filing dates but not SEC acceptance times.  The new
8-K run maps each filing to the first ticker session strictly after its filing
date.  It must not be described as an exact before/after-close event-time test.
