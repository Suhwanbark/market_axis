# Ticker-conditioned parametric surprise axis: end-to-end experiment

## 한국어 핵심 요약

이 실험에서 찾는 것은 주가 방향이나 변동성 축이 아니다. 먼저 시장
데이터를 전혀 사용하지 않고, 모델이 회사별로 원래 기대하던 결과와 실제
통제 보고서의 결과가 얼마나 어긋나는지를
\(-\log P_M(\text{realized outcome}\mid\text{firm, relation})\)으로 정의했다.
같은 relation과 같은 favorable/continuity/adverse 결과 안에서 target과
activation update를 각각 중심화한 뒤, `post - pre` hidden-state delta로
Ridge와 pair-ranking MLP를 학습했다. 따라서 단순한 긍정/부정 문체나 사건
종류를 읽어서 맞히는 probe가 되지 않도록 설계했다.

Held-out firm에서 MLP는 Qwen2.5와 Qwen3 모두 유의한 controlled decoding을
보였다. MLP는 한 개의 축이 아니므로, 실제 기사 단일 forward에서 직접
투영하기 위해 validation에서 고른 linear-delta Ridge를 raw hidden
coordinate의 unit vector로 변환했다. 이 방향은 Qwen2.5 layer 20,
Qwen3 layer 24에 고정했으며 기사나 시장 label은 전혀 보지 않았다.

첫 실제 뉴스 검증에서는 383개 July 기사에 어떠한 prefix, ticker 지시,
질문, 선택지도 붙이지 않고 `title + analysis_body`만 2,048 token까지
forward했다. 사전 primary인 last-token projection은 Qwen2.5
\(\rho=0.0024\), Qwen3 \(\rho=-0.0472\)로 둘 다 실패했다. Qwen2.5
mean-pooling sensitivity는 \(\rho=0.1831\), date-block 95% CI
[0.0663, 0.3226], \(p=0.0006\)이지만 Qwen3에서 반대로 나타났으므로
primary 성공이나 cross-model axis로 승격하지 않는다.

동일한 전체 기사로 수행한 전용 embedding+Ridge와 explicit Direct-LM
0--9 rating 비교도 완료했다. Qwen2.5 mean-axis는 Direct Qwen2.5보다
point estimate가 높지만 차이가 확정적이지 않았고, Embedding-8B+Ridge의
\(\rho=0.3062\)보다 낮았다.

Frozen axis의 2026년 전체 9,376개 기사 평가도 완료했다. 사전 primary인
last projection은 Qwen3에서 prefixless \(\rho=0.1118\), company/ticker
prefix \(\rho=0.1036\)으로 날짜·issuer·within-date 검정을 모두 통과했지만,
Qwen2.5에서는 0이었다. 따라서 primary의 cross-model replication은
실패했다. 사전에 sensitivity로 분리해 둔 Qwen2.5 mean projection은
1일/2일 절대 시장조정수익률, Parkinson 변동성 확장, 비정상 거래량
모두에서 \(\rho=0.1669\)--\(0.2480\)이고, full 기간에는 Direct Qwen2.5도
유의하게 이겼다. 다만 Qwen3에는 같은 우위가 없고, untouched July의
Embedding-8B보다도 낮았으므로 “일반적이고 embedding보다 우월한 axis”는
아니다. 현재 가장 정확한 결론은 **Qwen3 primary transfer 한 건과,
Qwen2.5의 강하지만 secondary/model-specific한 mean-pooled downstream
signal**이다.

## 0. Document status

This document is a live, append-only research record. It separates three
questions that are easy to conflate:

1. Can a model-internal representation predict how unexpected an outcome is
   relative to the model's own firm-specific parametric memory?
2. Does that representation generalize from controlled reports to untouched
   real news without using market labels?
3. Is the resulting signal useful for real market outcomes, and does it add
   anything beyond dedicated text embeddings or an explicit direct-LLM
   surprise rating?

The controlled benchmark, nonlinear/linear probe comparisons, first
prefixless July real-news projection, fair exact-article embedding/direct
baselines, and the complete eligible-2026 prefixless/company-prefix
evaluation are complete as of 2026-07-25 KST.

Important terminology:

- **Parametric memory** means information already retrievable from the frozen
  model weights before a report is supplied.
- **Ticker-conditioned surprise** means low model probability assigned to the
  subsequently realized firm-relation outcome.
- **Controlled readout** means a Ridge or MLP function learned only from
  controlled reports and firm-memory measurements.
- **Linear axis** means a one-dimensional raw activation direction obtained
  from the controlled Ridge coefficient.
- **Market axis** is deliberately *not* what is learned here. No return,
  volatility, volume, or price label is used to fit the controlled readout.

## 1. Research claim and falsification criteria

### 1.1 Intended claim

For a company-relation pair for which a frozen LLM has a strong directional
expectation, the model's internal state update after reading a controlled
report contains a decodable signal proportional to:

\[
S_i^{\mathrm{param}}
  = -\log P_M(Y_i=y_i \mid F_i, R_i),
\]

where:

- \(M\) is the frozen LLM;
- \(F_i\) identifies the firm;
- \(R_i\) is a relation such as product demand, litigation, guidance, or
  operational performance;
- \(Y_i\) is the report outcome;
- \(y_i\) is the realized outcome among favorable, continuity, and adverse.

This is a model-relative quantity. It does not assert that the model's prior
is normatively correct and it is not defined from market prices.

### 1.2 What would count as success

Evidence is accumulated in increasing order of difficulty:

1. **Controlled decoding:** held-out firms show positive within-group
   Spearman correlation and pair concordance between predicted and target
   surprise.
2. **Nuisance selectivity:** the signal survives conditioning on relation and
   realized outcome, so it cannot be explained only by favorable/adverse
   valence or report template.
3. **Model replication:** the controlled finding appears in both Qwen2.5-7B
   and Qwen3-4B, with independently fit readouts.
4. **Raw-article transfer:** a frozen direction produces a useful ordering
   after an actual article is forwarded with no task prefix or question.
5. **Market external validity:** the article score correlates with
   out-of-sample absolute abnormal return and volatility/attention outcomes.
6. **Baseline gain:** the signal is stronger than, or incrementally useful
   beyond, dedicated text-embedding and direct-LLM rating baselines under the
   same article and outcome protocol.

### 1.3 What would *not* count as success

- High train accuracy without held-out-firm performance.
- A probe that predicts favorable versus adverse outcome but not surprise
  differences within the same relation/outcome group.
- Selecting a layer or score after inspecting the market test labels.
- Calling a nonlinear MLP a literal one-dimensional axis.
- Calling a real-news correlation causal.
- Reporting only the best pooling rule after inspecting test results.
- Treating a post-hoc article-quality filter as confirmatory.
- Retuning the controlled axis on 2026 market outcomes.

If controlled decoding succeeds but raw-article transfer fails, the correct
name is **controlled parametric-surprise representation**. If real-news
correlation succeeds but incremental predictive gain fails, the correct
conclusion is **externally aligned but not incrementally useful**.

## 2. Frozen models

Two non-Llama models are analyzed independently:

| Internal name | Model | Hidden width | Controlled-selected layer |
|---|---|---:|---:|
| `qwen25_7b` | Qwen2.5-7B-Instruct | 3,584 | 20 |
| `qwen3_4b` | Qwen3-4B | 2,560 | 24 |

No vector is transferred directly between model activation spaces. Each model
has a separately fitted target normalizer, activation centering package,
probe, and raw-space direction.

Local inference uses validated snapshots under:

- `/tmp/axis_qwen25_7b`
- `/tmp/axis_qwen3_4b`

The runtime bootstrap must complete with `RUNTIME_BOOTSTRAP_OK` before any GPU
job is started:

```bash
bash scripts/bootstrap_tmp_runtime.sh
```

## 3. Controlled data construction

### 3.1 Unit of observation

A controlled row contains:

- a synthetic or templated report about a real firm-relation prompt;
- the model's pre-report categorical expectation over three outcomes;
- the report's realized outcome;
- pre- and post-report hidden states at a fixed answer location;
- a firm-disjoint split label;
- metadata used to measure parametric-memory strength.

The three outcomes are:

| Index | Letter | Meaning |
|---:|---|---|
| 0 | A | favorable |
| 1 | B | continuity |
| 2 | C | adverse |

The report must be parsed correctly by the model after observation. Rows are
eligible only when:

```text
strong_memory_benchmark == True
and post_correct == True
```

This prevents the surprise target from being interpreted when the model did
not have a sufficiently identified prior or failed to understand the report.
All eligible favorable, continuity, and adverse outcomes are retained.

### 3.2 Firm-disjoint split

The split unit is the firm, not an individual paraphrase. All controlled rows
derived from the same firm stay in one split.

| Model | Train rows | Validation rows | Held-out test rows |
|---|---:|---:|---:|
| Qwen2.5-7B | 2,682 | 912 | 888 |
| Qwen3-4B | 2,835 | 1,101 | 1,074 |

- Train firms fit target normalization, nuisance centers, feature scaling,
  and readout parameters.
- Validation firms select layer, Ridge regularization, or MLP capacity.
- Test firms are opened only for the final controlled evaluation.
- No market label participates in any of these operations.

### 3.3 Firm-specific prior

For row \(i\), the frozen model produces:

\[
\mathbf p_i =
\left[
p_i(\mathrm{fav}),
p_i(\mathrm{cont}),
p_i(\mathrm{adv})
\right].
\]

After renormalizing the three probabilities, the raw target is:

\[
s_i = -\log \max(\epsilon, p_i(y_i)), \qquad \epsilon=10^{-8}.
\]

Example:

- the model assigns a company a 0.75 favorable, 0.20 continuity, and 0.05
  adverse expectation for product demand;
- the controlled report realizes an adverse outcome;
- raw parametric surprise is \(-\log(0.05)=2.996\);
- if another firm had assigned 0.40 to adverse for the same relation and
  outcome, its raw surprise would be \(-\log(0.40)=0.916\).

The first report is therefore more surprising to that model even though both
reports are equally adverse at the text level.

### 3.4 Why relation-by-outcome normalization is required

Raw surprisal distributions differ mechanically by relation and realized
outcome. To prevent the probe from winning by learning only those group
differences, define:

\[
g_i = R_i \times y_i.
\]

For every group \(g\), fit mean \(\mu_g^s\) and standard deviation
\(\sigma_g^s\) on train firms only:

\[
t_i = \frac{s_i-\mu_{g_i}^s}{\sigma_{g_i}^s}.
\]

The supervised target \(t_i\) therefore means:

> How unexpectedly low was the model's probability of this outcome,
> compared with train examples having the same relation and the same realized
> outcome?

This is the central identification device. The probe cannot obtain a high
within-group score simply because an article is adverse, favorable, about
litigation, or about demand.

## 4. Hidden-state representation

### 4.1 Pre/post extraction

At candidate layer \(\ell\), the model is run twice at the same answer
position:

\[
h^-_{i\ell}
  = H_{\ell,r}(\text{firm expectation prompt}),
\]

\[
h^+_{i\ell}
  = H_{\ell,r}(\text{firm expectation prompt + controlled report}).
\]

The primary representation is the update:

\[
d_{i\ell} = h^+_{i\ell} - h^-_{i\ell}.
\]

Using a delta reduces fixed company-name, prompt-format, and baseline
firm-memory components. It asks what changed internally because the report was
observed.

### 4.2 Removing relation/outcome mean updates

Even \(d_{i\ell}\) contains a generic response to a favorable or adverse
report. For each relation/outcome group, fit a train-only activation center:

\[
\bar d_{\ell,g}
  = \frac{1}{|\mathcal D_{\mathrm{train},g}|}
    \sum_{i\in\mathcal D_{\mathrm{train},g}} d_{i\ell}.
\]

The centered update is:

\[
x_{i\ell}=d_{i\ell}-\bar d_{\ell,g_i}.
\]

The resulting readout is forced to use deviations from the ordinary hidden
update for the same relation and outcome.

### 4.3 Feature standardization

For each hidden coordinate \(j\), train-only mean \(m_{\ell j}\) and standard
deviation \(a_{\ell j}\) are fitted:

\[
z_{i\ell j} =
  \frac{x_{i\ell j}-m_{\ell j}}{a_{\ell j}}.
\]

Coordinates with non-finite or near-zero variance use scale 1. Validation and
test statistics never enter this transform.

## 5. Readouts considered

### 5.1 Linear Ridge readout

For each candidate layer and
\(\lambda\in\{0.1,1,10,100,1000\}\), fit:

\[
(\hat w_{\ell,\lambda},\hat b)
=
\arg\min_{w,b}
\sum_{i\in\mathrm{train}}
(t_i-w^\top z_{i\ell}-b)^2
+\lambda\|w\|_2^2.
\]

This produces an actual one-dimensional direction and is the readout used for
the raw-article projection experiment.

### 5.2 Absolute-delta Ridge diagnostic

An otherwise identical Ridge model is fit to \(|z_{i\ell}|\). This tests the
hypothesis that surprise is encoded only as unsigned activation magnitude.
It performs substantially below the signed delta readout on validation.

### 5.3 Pair-ranking MLP

Linearity is not assumed for the general representation claim. A one-hidden
layer MLP is fit with:

- hidden width 32 or 64;
- weight decay \(10^{-3}\) or \(10^{-2}\);
- Smooth-L1 pointwise regression loss;
- pair-ranking loss within the same relation/outcome group;
- three fixed seeds, ensembled by their mean prediction.

For a training pair \((a,b)\) in the same group with \(t_b>t_a\), the ranking
term penalizes failure to produce \(f_\theta(z_b)>f_\theta(z_a)\).

The validation selection criterion is lexicographic:

1. higher within-group pair concordance;
2. higher within-group Spearman correlation;
3. fixed configuration order as deterministic tie-break.

The MLP was the controlled winner for both models. It is called a nonlinear
readout, not an axis.

### 5.4 Post-only MLP diagnostic

A fixed-capacity MLP is also trained from \(h^+\) rather than \(h^+-h^-\).
This is a nuisance diagnostic and is never eligible as the frozen real-news
delta representation.

## 6. Controlled validation and held-out test

### 6.1 Metrics

Four complementary metrics are reported:

1. **Global Spearman:** rank association across all eligible rows.
2. **Within-group Spearman:** row-count-weighted Spearman calculated
   independently inside each relation/outcome group.
3. **Pair concordance:** Kendall-\(\tau_b\) transformed to \([0,1]\) and
   pair-count weighted across groups.
4. **Same-outcome conflict AUC:** confirm-versus-contradict discrimination
   while holding realized outcome group fixed.

Within-group metrics are primary because global metrics can be inflated by
relation or valence.

### 6.2 Validation comparison

| Model | Method | Layer | Pair concordance | Within-group Spearman |
|---|---|---:|---:|---:|
| Qwen2.5 | linear delta | 20 | 0.7113 | 0.5400 |
| Qwen2.5 | absolute delta | 20 | 0.5782 | 0.1817 |
| Qwen2.5 | pair-MLP delta | 20 | **0.7497** | **0.6095** |
| Qwen2.5 | pair-MLP post | 20 | 0.7495 | 0.6239 |
| Qwen3 | linear delta | 24 | 0.6070 | 0.2897 |
| Qwen3 | absolute delta | 24 | 0.5321 | 0.0775 |
| Qwen3 | pair-MLP delta | 24 | **0.6238** | **0.3290** |
| Qwen3 | pair-MLP post | 24 | 0.6202 | 0.3331 |

Post-only is not eligible despite a high diagnostic correlation because the
construct is defined as a report-induced belief update.

### 6.3 Held-out-firm result

| Model | Readout | Rows | Firms | Within-group Spearman | Pair concordance | Same-outcome conflict AUC |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5 | pair-MLP delta | 888 | 162 | **0.5829** | **0.7220** | 0.8519 |
| Qwen3 | pair-MLP delta | 1,074 | 153 | **0.3028** | **0.6127** | 0.7000 |

The linear delta controls remain positive:

| Model | Linear within-group Spearman | Linear pair concordance |
|---|---:|---:|
| Qwen2.5 | 0.5476 | 0.7058 |
| Qwen3 | 0.2709 | 0.5989 |

Firm-block bootstrap and stratified permutation results:

- Qwen2.5 MLP pair concordance 95% CI: [0.7010, 0.7430].
- Qwen2.5 MLP within-group Spearman 95% CI: [0.5174, 0.6317].
- Qwen3 MLP pair concordance 95% CI: [0.5935, 0.6371].
- Qwen3 MLP within-group Spearman 95% CI: [0.2499, 0.3599].
- All four one-sided stratified permutation tests have \(p=0.0005\) with
  2,000 repeats.

This supports a stable controlled decoding result in both models. It does not
by itself prove raw-news transfer or causal control.

## 7. Converting the controlled Ridge to a raw activation axis

### 7.1 Why a separate linear package is needed

The original validation winner is an MLP and cannot be represented by a
single vector. The new downstream question explicitly asks whether a news
article can be forwarded once and scored by projection onto a frozen
direction. Therefore, the best validation-selected **linear-delta** candidate
is reproduced without changing any controlled selection.

Frozen choices:

| Model | Layer | Ridge \(\lambda\) | Validation pair concordance |
|---|---:|---:|---:|
| Qwen2.5 | 20 | 1000 | 0.7113 |
| Qwen3 | 24 | 100 | 0.6070 |

### 7.2 Standardized-to-raw coefficient conversion

The Ridge model was trained in standardized coordinates:

\[
\hat t = w_{\mathrm{std}}^\top
\frac{x-m}{a}+b.
\]

Its coefficient in raw hidden coordinates is:

\[
w_{\mathrm{raw},j}
=\frac{w_{\mathrm{std},j}}{a_j}.
\]

The unit direction is:

\[
u=\frac{w_{\mathrm{raw}}}
        {\|w_{\mathrm{raw}}\|_2}.
\]

The saved package includes:

- `unit_axis`;
- raw and standardized coefficients;
- train-only feature mean and scale;
- train-only relation/outcome activation centers;
- Ridge intercept;
- layer and regularization metadata.

The axis freeze manifest explicitly records:

```json
{
  "market_labels_used": false,
  "test_articles_used": false,
  "status": "PREFIXLESS_TEST_AXES_FROZEN"
}
```

### 7.3 Important transfer stress-test caveat

The controlled readout was learned from a centered pre/post delta, whereas the
requested test projects a raw article-only hidden state:

\[
\mathrm{score}_{i}^{\mathrm{raw}}
  = u^\top h_{\ell}^{\mathrm{article}}.
\]

No pre-article state, relation label, outcome label, controlled group center,
feature mean, or Ridge intercept is available in this test.

This is intentionally a severe zero-shot test, but it is not algebraically
equivalent to applying the controlled Ridge predictor. A positive result would
show that the controlled direction is also present in ordinary article states.
A null result would not negate controlled delta decoding; it would show that
raw-state transfer requires conditioning, a delta construction, or a nonlinear
readout.

## 8. Real-news cohort

### 8.1 Source and row identity

The frozen v9 retrieval-conditioned Yahoo news super-cohort contains 13,735
one-article ticker-session rows from 3,321 issuers. A label-blind atomic-claim
audit removes recap, investment commentary, market-reaction-title,
speculative-event, and relation-homonym articles.

The resulting eligible cohort contains:

- 9,376 independent articles;
- one selected article per ticker-session;
- dates from 2026-01-02 through 2026-07-21;
- 9,714 relation rows before article-level aggregation;
- 383 July confirmation articles.

Both activation scoring and all semantic/text controls use the identical
`analysis_keep` mask.

### 8.2 Cleaning

`analysis_body` removes:

- sentences that explicitly describe the observed market reaction;
- analyst price-target statements;
- boilerplate that does not describe the event.

The raw body remains preserved for audit. The title is retained. Cleaning is
label-blind: no return, OHLC, volume, or volatility value is used.

### 8.3 Exact prefixless input

The article-only text is exactly:

```text
[title]

[analysis_body]
```

There is no:

- `Target company:` prefix;
- company or ticker instruction;
- surprise question;
- A/B/C/D option list;
- answer delimiter;
- chat template;
- market information.

Tokenization uses right padding and right truncation at 2,048 tokens. The
selected layer is captured during the ordinary causal-LM forward pass.

### 8.4 Pooling and frozen scores

Four scores are materialized without labels:

\[
s_{\mathrm{last}} = u^\top h_{\mathrm{last}},
\]

\[
c_{\mathrm{last}} =
\frac{u^\top h_{\mathrm{last}}}
{\|h_{\mathrm{last}}\|_2},
\]

\[
s_{\mathrm{mean}} = u^\top
\left(\frac{1}{T}\sum_{t=1}^{T}h_t\right),
\]

\[
c_{\mathrm{mean}} =
\frac{u^\top h_{\mathrm{mean}}}
{\|h_{\mathrm{mean}}\|_2}.
\]

`last_projection` is the predeclared primary score. The other three are
sensitivity analyses and cannot replace the primary result after labels are
opened.

## 9. Initial July market test

### 9.1 Outcome

Let the ticker's close-to-close log return on the article event session be
\(r_{i,1}\), and the SPY return be \(r_{m,1}\). The primary external outcome
is:

\[
Y_i^{\mathrm{AR1}}=|r_{i,1}-r_{m,1}|.
\]

The event-session rule is:

- before the market opens: same trading session;
- after the market closes: next trading session;
- during trading hours: the event session is retained, while lagged controls
  use the last fully observed session.

The July cohort contains 383 articles across 9 event dates. This is a
retrospective test because the parent retrieval-conditioned study previously
opened July values. The axis itself still uses no article or market label.

### 9.2 Statistical test

For each frozen score:

1. compute article-level Spearman \(\rho\);
2. resample event dates as blocks 5,000 times for a 95% percentile interval;
3. permute scores within event date 5,000 times for a one-sided randomization
   \(p\)-value.

The within-date permutation removes day-level market conditions from the null
comparison. Date-block resampling preserves cross-sectional dependence within
each event date.

### 9.3 July result

| Model | Pooling/score | Status | Spearman | Date-block 95% CI | Within-date one-sided \(p\) |
|---|---|---|---:|---:|---:|
| Qwen2.5 | last projection | **Primary** | 0.0024 | [-0.1079, 0.1148] | 0.6225 |
| Qwen2.5 | last cosine | Sensitivity | 0.0120 | [-0.0989, 0.1165] | 0.5781 |
| Qwen2.5 | mean projection | Sensitivity | **0.1831** | **[0.0663, 0.3226]** | **0.0006** |
| Qwen2.5 | mean cosine | Sensitivity | **0.1972** | **[0.0814, 0.3354]** | **0.0004** |
| Qwen3 | last projection | **Primary** | -0.0472 | [-0.1388, 0.0399] | 0.8218 |
| Qwen3 | last cosine | Sensitivity | -0.0431 | [-0.1369, 0.0495] | 0.8004 |
| Qwen3 | mean projection | Sensitivity | -0.1171 | [-0.1553, -0.0735] | 0.9900 |
| Qwen3 | mean cosine | Sensitivity | -0.1175 | [-0.1759, -0.0685] | 0.9894 |

The predeclared primary raw-state transfer test **fails** in both models.
Neither last-token projection is positively associated with the July
one-session absolute market-adjusted return.

Qwen2.5 mean pooling is a positive, statistically strong sensitivity result.
It cannot replace the failed primary result because pooling was not selected
on validation market data. The correct interpretation at this checkpoint is:

> Controlled delta decoding is replicated, but a model-replicated
> prefixless raw-state market signal has not been established. Qwen2.5
> contains an exploratory distributed/mean-pooled transfer signal that must be
> checked on the full 2026 sample and against strong baselines.

Qwen3 mean pooling moves in the opposite direction, so the Qwen2.5 sensitivity
does not replicate cross-model. The disagreement also argues against treating
the mean-pooled result as a generic market-surprise axis.

The article token audit is identical across models:

- mean retained tokens: 1,030.87;
- median retained tokens: 884;
- fraction reaching the 2,048-token ceiling: 16.97%.

No market-label mapping, sign flip, absolute-value transform, pooling
selection, or score calibration was fitted for this result.

The terminal source will be:

```text
outputs/prefixless_article_axis_test/evaluation/
  july_prefixless_axis_results.csv
  july_prefixless_axis_scores.parquet
  manifest.json
```

## 10. Dedicated embedding and direct-LLM baselines

This comparison is complete and is reported even though the axis does not win.

### 10.1 Fair-input requirement

The baseline must receive the same cleaned title and article body and the same
2,048-token ceiling. Older embedding artifacts based on a company-prefixed
`classifier_text` truncated to 2,500 characters are retained only as audit
controls and are not the final fair comparison.

### 10.2 Embedding baseline

Two dedicated embedding models are evaluated:

- Qwen3-Embedding-0.6B;
- Qwen3-Embedding-8B.

The embedding forward itself is label-free. A Ridge market head must be fit
only on a development period and selected on a later validation period before
the July confirmation rows are evaluated. It is invalid to call all 2026 rows
an untouched test set for an embedding+Ridge model if any of their market
labels fit the Ridge head.

The embedding vectors are L2-normalized final-token representations. A
64-component randomized PCA is fit on the 3,522 development-train articles
only. Ridge alpha is selected on 4,477 chronologically later development
articles, after which the head is refit on both development splits and
evaluated on the 383 July rows.

The comparison therefore distinguishes:

- zero-shot scalar embedding novelty on the whole eligible cohort;
- embedding+Ridge on a genuinely chronological held-out period;
- raw frozen-axis correlation, which requires no market head.

### 10.3 Direct-LLM baseline

The direct baseline receives the article and a fixed 0--9 question:

- 0 means fully expected and consistent with prior parametric expectations;
- 9 means extremely unexpected and inconsistent;
- score is the expected value under next-token logits for digits 0--9.

The question and target company are appended after the article. The article
token budget is fixed before label access so that the question cannot be
truncated. The direct score is computed without market labels.

### 10.4 Required comparison statistics

For the identical held-out rows:

- univariate Spearman correlation with every market outcome;
- date-block confidence interval;
- within-date permutation \(p\)-value;
- paired date-block interval for
  \(\rho_{\mathrm{axis}}-\rho_{\mathrm{baseline}}\);
- chronological held-out \(R^2\) and QLIKE/MAE where defined;
- incremental performance after pre-event volatility and lagged absolute
  return controls;
- false-discovery-rate adjusted values across secondary outcomes.

### 10.5 Input and score audit

All forwards cover the same 9,376 eligible articles.

| Forward | Mean retained article tokens | Median | Truncation/max-length rate |
|---|---:|---:|---:|
| Axis and embedding article input | 1,183.54 | 1,058 | 28.54% at 2,048 tokens |
| Direct Qwen2.5 article portion | 1,156.03 | — | 29.58% article clipping |
| Direct Qwen3 article portion | 1,154.85 | — | 29.62% article clipping |

The direct prompt reserves room for the complete rating question. Its whole
rendered chat reaches 2,048 tokens for 25.74% of Qwen2.5 rows and 25.87% of
Qwen3 rows.

The direct ratings are non-degenerate:

| Model | Rating mean | Rating standard deviation |
|---|---:|---:|
| Qwen2.5 | 1.9215 | 1.4614 |
| Qwen3 | 4.2379 | 1.0061 |

Different means are not calibrated across models; only within-model article
rankings are interpreted.

### 10.6 Zero-shot July comparison

Axis and Direct-LM scores in this table use no market-label head:

| Score | Spearman | Date-block 95% CI | Within-date one-sided \(p\) |
|---|---:|---:|---:|
| Qwen2.5 last-axis | 0.0024 | [-0.1079, 0.1163] | 0.6327 |
| Qwen3 last-axis | -0.0472 | [-0.1393, 0.0408] | 0.8231 |
| Qwen2.5 mean-axis | **0.1831** | **[0.0687, 0.3261]** | **0.0005** |
| Qwen3 mean-axis | -0.1171 | [-0.1554, -0.0764] | 0.9885 |
| Direct Qwen2.5 | 0.1066 | [-0.0312, 0.2368] | 0.0225 |
| Direct Qwen3 | **0.1400** | **[0.0039, 0.2714]** | **0.0100** |

The within-date permutation \(p\) and date-block interval answer different
questions. Direct Qwen2.5 has a positive within-date permutation result but
its date-block interval crosses zero, so it is not robust to the small
nine-date block sample.

Paired date-block comparison:

| Axis candidate | Baseline | \(\Delta\rho\) | 95% CI |
|---|---|---:|---:|
| Qwen2.5 last-axis | Direct Qwen2.5 | -0.1043 | [-0.2543, 0.0562] |
| Qwen3 last-axis | Direct Qwen3 | -0.1872 | **[-0.3339, -0.0308]** |
| Qwen2.5 mean-axis | Direct Qwen2.5 | +0.0764 | [-0.0459, 0.1972] |
| Qwen3 mean-axis | Direct Qwen3 | -0.2570 | **[-0.3995, -0.1335]** |

Qwen2.5 mean pooling has a higher point estimate than its direct rating, but
the paired interval crosses zero. There is no established gain over Direct LM.

### 10.7 Dedicated embedding+Ridge comparison

The market Ridge predictions use the same chronological development split and
the same 383 July rows:

| Feature set | Features | July log-\(R^2\) | Spearman | Large-reaction AUC |
|---|---:|---:|---:|---:|
| Qwen2.5 last-axis Ridge | 1 | -0.1567 | -0.0024 | 0.5136 |
| Qwen2.5 mean-axis Ridge | 1 | -0.0754 | 0.1831 | 0.7008 |
| Direct Qwen3 Ridge | 1 | -0.0736 | 0.1400 | 0.5996 |
| Embedding-0.6B Ridge | 64 | -0.0049 | 0.2456 | 0.7491 |
| Embedding-8B Ridge | 64 | **0.0329** | **0.3062** | **0.8117** |

The best dedicated text baseline is Embedding-8B. Relative to it:

| Axis prediction | \(\Delta\rho\) versus Embedding-8B | 95% CI |
|---|---:|---:|
| Qwen2.5 last-axis Ridge | -0.3085 | **[-0.4812, -0.1755]** |
| Qwen3 last-axis Ridge | -0.3534 | **[-0.4820, -0.2438]** |
| Qwen2.5 mean-axis Ridge | -0.1231 | [-0.2741, 0.0119] |
| Qwen3 mean-axis Ridge | -0.4232 | **[-0.5765, -0.2758]** |

Thus the primary axes clearly lose to Embedding-8B, and the favorable
Qwen2.5 mean sensitivity still does not establish superiority.

### 10.8 Incremental downstream tests

Pre-event controls are trailing volatility and lagged absolute
market-adjusted return.

| Candidate minus baseline | \(\Delta R^2_{\log}\) | Date-block 95% CI | \(\Delta\)AUC | AUC 95% CI |
|---|---:|---:|---:|---:|
| price + last axes minus price | -0.0059 | [-0.0231, 0.0144] | -0.0401 | [-0.1095, 0.0151] |
| price + mean axes minus price | **+0.0558** | **[0.0044, 0.1138]** | -0.0223 | [-0.1193, 0.0491] |
| price + Direct ratings minus price | **+0.1937** | **[0.0076, 0.5044]** | -0.0285 | [-0.1675, 0.0697] |
| price + Embedding-8B + last axes minus price + Embedding-8B | -0.0043 | **[-0.0070, -0.0015]** | -0.0036 | [-0.0131, 0.0021] |
| price + Embedding-8B + mean axes minus price + Embedding-8B | +0.0177 | [-0.0006, 0.0433] | -0.0056 | [-0.0182, 0.0029] |

Mean axes improve log-\(R^2\) relative to price controls alone, although both
models remain poorly calibrated in absolute \(R^2\) and the AUC does not
improve. Once the stronger Embedding-8B representation is included, the
two-sided 95% interval for mean-axis incremental \(R^2\) crosses zero. Last
axes significantly reduce \(R^2\) beyond Embedding-8B.

### 10.9 Baseline conclusion

The downstream success criterion is not met:

1. the primary last-token axis is null or negative;
2. Qwen2.5 mean pooling is the only favorable axis sensitivity;
3. it does not replicate in Qwen3;
4. it does not significantly beat Direct Qwen2.5 in paired rank correlation;
5. it is weaker than the dedicated Embedding-8B Ridge baseline;
6. its incremental gain beyond Embedding-8B is not established at a
   two-sided 95% level.

The accurate current label is:

> A replicated controlled parametric-surprise representation with one
> model-specific mean-pooled real-news sensitivity, but no demonstrated
> general or embedding-superior downstream axis.

## 11. Full available 2026 expansion

All 9,376 currently eligible articles have event sessions in 2026. Because no
real article was used to fit the controlled axis, the frozen raw projection
can be evaluated across the complete available period from 2026-01-02 through
2026-07-21.

The underlying label-blind super-cohort contains 13,735 articles. Before any
market outcome was opened, the frozen atomic-claim audit removed 4,359 recap,
investment-commentary, market-reaction-title, speculative-event, and
relation-homonym articles. “Full 2026” below therefore means every one of the
9,376 **pre-frozen eligible** articles, not a post-outcome subset and not the
13,735-row raw super-cohort.

This is broader and higher powered than the 383-row July analysis, but it is
retrospective. It must not be described as a preregistered untouched
confirmation.

### 11.1 Two frozen input variants

**Prefixless**

```text
[title]

[analysis_body]
```

**Ticker/company prefix**

```text
Target company: [company_name] ([market_ticker])

[title]

[analysis_body]
```

The prefix version contains attribution only. It contains no event keyword,
surprise instruction, answer options, price, or outcome.

### 11.2 Full-period market outcomes

The planned outcomes are:

1. one-session absolute market-adjusted return;
2. two-session absolute market-adjusted return;
3. Parkinson range-volatility expansion relative to its trailing 20-session
   level;
4. abnormal volume relative to its trailing 20-session median.

Definitions:

\[
Y_i^{\mathrm{AR1}}
  = |r_{i,1}-r_{m,1}|,
\]

\[
Y_i^{\mathrm{AR2}}
  = |(r_{i,1}-r_{m,1})+(r_{i,2}-r_{m,2})|,
\]

\[
PK_{it}
  = \frac{\log(H_{it}/L_{it})^2}{4\log 2},
\]

\[
Y_i^{\mathrm{PK}}
  = \log\frac{PK_{i,t}+10^{-12}}
  {\operatorname{mean}(PK_{i,t-20:t-1})+10^{-12}},
\]

\[
Y_i^{\mathrm{VOL}}
  = \log\frac{V_{i,t}+1}
  {\operatorname{median}(V_{i,t-20:t-1})+1}.
\]

Raw OHLC is audited for mechanically split-like overnight gaps before returns
are formed. Intraday publications use lag-2 rather than lag-1 pre-event
controls to prevent the current partially observed session from entering the
baseline.

### 11.3 Primary and secondary decisions

- Primary score: `last_projection`.
- Primary outcome: `abs_market_adjusted_return_h1`.
- Co-primary model replication: Qwen2.5 and Qwen3, corrected across the two
  model tests.
- Secondary scores: last cosine, mean projection, mean cosine.
- Secondary outcomes: two-session absolute return, Parkinson expansion,
  abnormal volume.
- Input-variant comparison: prefixless versus attribution prefix, paired by
  article and event date.
- Dependence checks: event-date block bootstrap, issuer-block bootstrap, and
  a within-date permutation test for the primary score/outcome.

The success criterion "the axis must be better" is treated as a falsifiable
criterion, not permission for test-set tuning. If it is not better, the
negative result will be committed unchanged.

### 11.4 Cohort and outcome availability

The 9,376 eligible rows are distributed as 3,545 `development_train`, 4,525
`development_test`, 923 `boundary_excluded`, and 383 `july_confirmation`
rows. Every row is a different ticker-session. The controlled axis was fit on
none of these articles. Outcome availability after price materialization is:

| Outcome | Finite rows |
|---|---:|
| one-session absolute market-adjusted return | 9,335 |
| two-session absolute market-adjusted return | 9,271 |
| Parkinson range-volatility expansion | 9,304 |
| abnormal volume | 9,304 |

The period contains 137 event dates and 3,042 tickers. Five mechanically
split-like price rows are excluded by the frozen price audit.

### 11.5 Full-period primary result

| Model | Input | Spearman | Date-block 95% CI | Issuer-block 95% CI | model-BH \(q\), date / issuer | Within-date \(p\) |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5 | prefixless | -0.0040 | [-0.0287, 0.0209] | [-0.0259, 0.0174] | 0.6187 / 0.6457 | 0.7326 |
| Qwen2.5 | company prefix | -0.0004 | [-0.0266, 0.0266] | [-0.0225, 0.0205] | 0.5097 / 0.5132 | 0.5667 |
| Qwen3 | prefixless | **0.1118** | **[0.0798, 0.1382]** | **[0.0892, 0.1340]** | **0.0010 / 0.0010** | **0.0005** |
| Qwen3 | company prefix | **0.1036** | **[0.0756, 0.1272]** | **[0.0815, 0.1253]** | **0.0010 / 0.0010** | **0.0005** |

Thus Qwen3 passes the primary transfer test in both input variants. Qwen2.5
does not, so the required cross-model primary replication fails. Adding the
company/ticker attribution does not improve the Qwen3 primary score:
\(\Delta\rho=-0.0082\), date-block 95% CI [-0.0183, 0.0038].

### 11.6 Downstream outcome coverage

The table below shows the two useful but differently qualified frozen
signals: Qwen3 `last_projection` is the primary score, while Qwen2.5
`mean_projection` is the sensitivity specified before the full-period test.

| Model / score | Input | \(|MAR_1|\) | \(|MAR_2|\) | Parkinson expansion | Abnormal volume |
|---|---|---:|---:|---:|---:|
| Qwen3 last | prefixless | 0.1118 | 0.0986 | 0.2025 | 0.1661 |
| Qwen3 last | company prefix | 0.1036 | 0.0898 | 0.2048 | 0.1598 |
| Qwen2.5 mean | prefixless | 0.1755 | 0.1669 | 0.1848 | 0.2308 |
| Qwen2.5 mean | company prefix | 0.1869 | 0.1773 | 0.2084 | 0.2480 |

All 16 correlations in this table have positive date-block and issuer-block
95% intervals and remain significant under the 64-comparison exploratory BH
correction. In contrast, Qwen2.5 `last_projection` is null across all four
outcomes. This is why the Qwen2.5 mean result is useful as a downstream
feature but cannot retroactively replace the primary endpoint.

### 11.7 Direct-LM comparison and practical utility

The zero-shot Direct-LM rating is significantly correlated with every
full-period outcome:

| Direct model | \(|MAR_1|\) | \(|MAR_2|\) | Parkinson expansion | Abnormal volume |
|---|---:|---:|---:|---:|
| Qwen2.5 | 0.1279 | 0.1121 | 0.0737 | 0.1187 |
| Qwen3 | 0.2117 | 0.1919 | 0.2727 | 0.2857 |

Every primary last-axis variant loses to its corresponding Direct LM on all
four outcomes; every paired date-block and issuer-block interval for
\(\rho_{\text{axis}}-\rho_{\text{Direct}}\) is strictly negative. For the
primary one-session outcome:

| Model | Axis input | Axis \(\rho\) | Direct \(\rho\) | Paired \(\Delta\rho\), date 95% CI |
|---|---|---:|---:|---:|
| Qwen2.5 | prefixless | -0.0040 | 0.1279 | -0.1320 [-0.1624, -0.1038] |
| Qwen2.5 | company prefix | -0.0004 | 0.1279 | -0.1283 [-0.1582, -0.0993] |
| Qwen3 | prefixless | 0.1118 | 0.2117 | -0.0999 [-0.1347, -0.0707] |
| Qwen3 | company prefix | 0.1036 | 0.2117 | -0.1080 [-0.1375, -0.0780] |

There is one qualified practical win. Qwen2.5 mean projection exceeds Direct
Qwen2.5 for every outcome. With the company prefix, the paired gains are
+0.0589 for one-session return, +0.0651 for two-session return, +0.1347 for
Parkinson expansion, and +0.1293 for abnormal volume; both date- and
issuer-block intervals are strictly positive. Qwen3 does not reproduce this
advantage.

The dedicated embedding Ridge cannot be fairly re-evaluated on “all 2026 as
test,” because it was fit using development-period market labels. Its fair
untouched comparison therefore remains the 383-row July result, where
Embedding-8B reaches \(\rho=0.3062\) and beats every axis score. Accordingly,
the overall requirement to beat both Direct LM and the strong embedding
baseline is not met.

### 11.8 What the attribution prefix changes

| Model / score | Prefixless-prefix score Spearman | Mean absolute change in prefixless SD |
|---|---:|---:|
| Qwen2.5 last | 0.7471 | 0.3765 |
| Qwen2.5 mean | 0.9906 | 0.1196 |
| Qwen3 last | 0.8788 | 0.3457 |
| Qwen3 mean | 0.9795 | 0.1638 |

The prefix adds a median 11 total tokens. The 2,048-token cap affects 2,673
prefixless rows and 2,683 prefixed rows, so only ten additional rows hit the
cap. Mean pooling is much more stable than last-token pooling. The prefix
significantly improves mean-pooling correlation for both models and all four
outcomes, but it does not reliably improve the primary last projection.

### 11.9 Post-open audit disclosure

The primary table, all 64 fixed axis/outcome combinations, prefix comparison,
and Direct-LM primary comparison were fixed before the full-period result was
opened. After seeing that first table, Direct-LM comparison was extended to
all four outcomes and both mean sensitivities. This audit extension does not
change the primary score, outcome, layer, direction, pooling decision, or
success rule. Its results are reported as comparative diagnostics, not as a
new preregistered endpoint.

## 12. Leakage and audit matrix

| Possible leakage | Protection |
|---|---|
| Market label enters controlled axis | Axis manifest asserts no market labels; source target is model probability |
| Same firm in controlled train and test | Firm-disjoint `recall_split` |
| Relation/valence shortcut | Train-only relation × realized-outcome centering |
| Test statistics enter standardization | Train-only mean and scale |
| Test selects layer or Ridge alpha | Validation-only lexicographic selection |
| News article trains controlled readout | Axis manifest asserts `test_articles_used=false` |
| Market-reaction sentence leaks outcome | Label-blind sentence removal and atomic-claim audit |
| Different article text benefits one baseline | Exact-text, exact-row, exact-token-budget comparison |
| Multiple articles inflate one ticker-session | One article per ticker-session |
| Same-day dependence narrows interval | Event-date block bootstrap |
| Market-wide day effect drives rank result | Within-date permutation |
| Multiple secondary endpoints | FDR adjustment; fixed primary endpoint |
| Failed run is silently replaced | Checkpoints and terminal manifests are preserved |
| Negative result is hidden | Every checkpoint is committed regardless of sign |

## 13. Interpretation boundaries

### 13.1 What a positive market correlation would mean

A positive correlation means that a controlled, model-relative surprise
direction ranks some real articles in a way associated with larger subsequent
market belief revision or attention. This is external predictive validity.

It does not prove:

- that LLM surprise causes market movement;
- that the model has the market's exact prior;
- that an individual activation coordinate has a human-interpretable causal
  meaning;
- that the result generalizes beyond the frozen models, dates, sources, and
  cleaning rule.

### 13.2 What baseline gain would mean

Outperforming a large dedicated embedding+Ridge baseline would show that the
controlled construction contributes information not easily recovered from a
generic semantic representation under the same sample and protocol.

Outperforming direct-LLM ratings would show that hidden-state access adds value
beyond asking the same model explicitly. Failure to beat either baseline
would narrow the contribution to representation analysis rather than
downstream utility.

### 13.3 Why both linear and nonlinear results are retained

- The MLP tests whether the construct is decodable without assuming
  one-dimensional linear geometry.
- The Ridge direction tests the stronger, more interpretable claim that a
  reusable linear direction exists.
- Raw article projection tests an even stronger transfer claim.

These are nested claims. A stronger failure does not erase a weaker success.

## 14. Relation to prior methodology

The design follows established probing lessons rather than treating probe
accuracy alone as representation proof:

- [Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/)
  motivates explicit control tasks and selectivity checks.
- [The Geometry of Truth](https://arxiv.org/abs/2310.06824) motivates
  held-out transfer, simple linear directions, and separating decoding from
  causal evidence.
- [Inference-Time Intervention](https://arxiv.org/abs/2306.03341) provides a
  precedent for manipulating behavior along activation directions; this
  experiment does not claim such causal evidence until intervention succeeds.
- [AxBench](https://arxiv.org/abs/2501.17148) motivates retaining strong simple
  baselines and reporting detection separately from steering.

The current experiment adds a finance-specific identification step:
conditioning both target and hidden update on relation × realized outcome so
that surprise is not reducible to event valence or semantic category.

## 15. Reproduction map

### 15.1 Controlled nonlinear and linear comparison

```bash
/tmp/axis_conda_sh/bin/python \
  scripts/ticker_surprise_nonlinear_finder.py \
  --model qwen25_7b --device 0

/tmp/axis_conda_sh/bin/python \
  scripts/ticker_surprise_nonlinear_finder.py \
  --model qwen3_4b --device 1
```

### 15.2 Freeze raw-space linear axes

```bash
/tmp/axis_conda_sh/bin/python \
  scripts/build_prefixless_test_axes.py
```

Expected terminal status:

```text
PREFIXLESS_TEST_AXES_FROZEN
```

### 15.3 Prefixless article extraction

Four resumable shards are used:

```bash
CUDA_VISIBLE_DEVICES=0 /tmp/axis_conda_sh/bin/python \
  scripts/extract_prefixless_article_axis_scores.py \
  --model qwen25_7b --device 0 \
  --shard-index 0 --num-shards 2 \
  --batch-size 2 --max-length 2048

CUDA_VISIBLE_DEVICES=1 /tmp/axis_conda_sh/bin/python \
  scripts/extract_prefixless_article_axis_scores.py \
  --model qwen25_7b --device 0 \
  --shard-index 1 --num-shards 2 \
  --batch-size 2 --max-length 2048

CUDA_VISIBLE_DEVICES=2 /tmp/axis_conda_sh/bin/python \
  scripts/extract_prefixless_article_axis_scores.py \
  --model qwen3_4b --device 0 \
  --shard-index 0 --num-shards 2 \
  --batch-size 4 --max-length 2048

CUDA_VISIBLE_DEVICES=3 /tmp/axis_conda_sh/bin/python \
  scripts/extract_prefixless_article_axis_scores.py \
  --model qwen3_4b --device 0 \
  --shard-index 1 --num-shards 2 \
  --batch-size 4 --max-length 2048
```

Each shard ends with:

```text
PREFIXLESS_ARTICLE_AXIS_SHARD_COMPLETE
```

### 15.4 July evaluation

```bash
/tmp/axis_conda_sh/bin/python \
  scripts/evaluate_prefixless_article_axis_test.py \
  --repeats 5000
```

Expected terminal status:

```text
PREFIXLESS_ARTICLE_AXIS_TEST_COMPLETE
```

### 15.5 Exact-article baselines

Direct-LM and embedding jobs are sharded identically to the axis extraction.
The coordinator waits for all direct manifests, revalidates the local runtime,
launches both embedding models, then runs the fair evaluator:

```bash
tmux new-session -d -s exact_baseline_coordinator \
  "bash scripts/run_exact_article_baseline_coordinator.sh"
```

The component commands are implemented in:

```text
scripts/extract_exact_article_direct_rating.py
scripts/extract_exact_article_embeddings.py
scripts/evaluate_exact_article_baselines.py
scripts/evaluate_exact_article_incremental_gain.py
```

Expected terminal statuses:

```text
EXACT_ARTICLE_DIRECT_RATING_SHARD_COMPLETE
EXACT_ARTICLE_EMBEDDING_SHARD_COMPLETE
EXACT_ARTICLE_BASELINE_EVALUATION_COMPLETE
EXACT_ARTICLE_INCREMENTAL_GAIN_COMPLETE
```

### 15.6 Full-2026 attribution-prefix and market evaluation

Before any GPU launch, restore and validate the local model copies:

```bash
bash scripts/bootstrap_tmp_runtime.sh
```

The command must print `RUNTIME_BOOTSTRAP_OK`. The four company-prefix shards
use the same frozen axis, row order, layers, pooling definitions, and
2,048-token limit as the prefixless run. Each command was run in its own named
`tmux` session (`full2026_prefix_q25_s0`, `full2026_prefix_q25_s1`,
`full2026_prefix_q3_s0`, and `full2026_prefix_q3_s1`):

```bash
CUDA_VISIBLE_DEVICES=0 /tmp/axis_conda_sh/bin/python \
  scripts/extract_company_prefixed_article_axis_scores.py \
  --model qwen25_7b --device 0 \
  --shard-index 0 --num-shards 2 --batch-size 2 --max-length 2048

CUDA_VISIBLE_DEVICES=1 /tmp/axis_conda_sh/bin/python \
  scripts/extract_company_prefixed_article_axis_scores.py \
  --model qwen25_7b --device 0 \
  --shard-index 1 --num-shards 2 --batch-size 2 --max-length 2048

CUDA_VISIBLE_DEVICES=2 /tmp/axis_conda_sh/bin/python \
  scripts/extract_company_prefixed_article_axis_scores.py \
  --model qwen3_4b --device 0 \
  --shard-index 0 --num-shards 2 --batch-size 4 --max-length 2048

CUDA_VISIBLE_DEVICES=3 /tmp/axis_conda_sh/bin/python \
  scripts/extract_company_prefixed_article_axis_scores.py \
  --model qwen3_4b --device 0 \
  --shard-index 1 --num-shards 2 --batch-size 4 --max-length 2048
```

Only after the controlled axis and article scores are frozen, materialize the
market outcomes and run the fixed evaluator:

```bash
/tmp/axis_conda_sh/bin/python \
  scripts/build_ticker_surprise_full_2026_outcomes.py

/tmp/axis_conda_sh/bin/python \
  scripts/evaluate_ticker_surprise_full_2026.py \
  --repeats 2000 --permutation-repeats 2000
```

Expected terminal statuses:

```text
COMPANY_PREFIXED_AXIS_SHARD_COMPLETE
TICKER_SURPRISE_FULL_2026_OUTCOMES_COMPLETE
TICKER_SURPRISE_FULL_2026_EVALUATION_COMPLETE
```

## 16. Artifact locations

Controlled readouts:

```text
outputs/ticker_surprise_nonlinear/
```

Frozen linear axes:

```text
outputs/prefixless_article_axis_test/axes/
```

Prefixless hidden states and projections:

```text
outputs/prefixless_article_axis_test/extractions/
```

July evaluation:

```text
outputs/prefixless_article_axis_test/evaluation/
```

Company/ticker-prefix scores and the full-2026 retrospective evaluation:

```text
outputs/company_prefixed_article_axis_test/extractions/
outputs/ticker_surprise_full_2026/
```

Curated, small reference tables intended for Git will be copied under:

```text
results/reference/ticker_parametric_surprise_axis/
```

Large hidden arrays, model weights, source articles, and market data are not
committed. Their manifests, row counts, configuration, hashes, and small
result tables are committed.

## 17. Running conclusions

1. A model-relative, ticker-conditioned surprise target is decodable from
   controlled activation updates for both Qwen2.5-7B and Qwen3-4B.
2. The pair-ranking MLP is the strongest controlled readout, so the broad
   representation claim does not require linearity.
3. A simpler linear delta readout remains positive on held-out firms and
   supplies a reusable raw-space direction.
4. The first July primary transfer test fails for both models. A Qwen2.5
   mean-pooled sensitivity is positive, but is post-primary and does not
   replicate in Qwen3; it is not promoted to the main claim.
5. In the untouched July comparison, the primary axes lose to exact
   Direct-LM and dedicated Embedding-8B baselines. Qwen2.5 mean pooling exceeds
   its direct-model point estimate there but not with a paired interval
   excluding zero, and it remains below Embedding-8B.
6. Mean axes improve log-\(R^2\) beyond price controls alone, but their
   incremental gain beyond Embedding-8B is not established at a two-sided 95%
   level. Last axes reduce performance beyond Embedding-8B.
7. In the full eligible-2026 retrospective test, Qwen3 last projection
   transfers to all four market outcomes and passes the primary one-session
   test under date, issuer, and within-date controls. Qwen2.5 last projection
   remains null, so primary cross-model replication fails.
8. Qwen2.5 mean projection is a strong model-specific secondary signal across
   returns, range volatility, and volume. It beats Direct Qwen2.5 across the
   full period, but Qwen3 does not reproduce that relative advantage.
9. A company/ticker attribution prefix significantly improves both models'
   mean-pooling correlations but does not improve the primary last projection.
10. The full primary axes remain weaker than Direct LM, and the fair July
    result remains weaker than Embedding-8B. The claim of a general,
    embedding-superior downstream axis is therefore rejected.
